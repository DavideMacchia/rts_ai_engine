# Design Decisions & Diagnostic Log

> Why the training system is built the way it is. This is the hard-won knowledge:
> each item cost real debugging and is easy to re-break if the reasoning is lost.
> Written as a decision log so future changes can see the rationale.

---

## The core problem we were solving

For a long time the agent **refused to build any military** — it learned an
economy-only policy and either idled or lost. Fixing this uncovered a chain of
independent issues, each hiding the next. This log walks the chain in order.

---

## D1 — Use MaskablePPO, not plain PPO

**Symptom:** the deterministic policy did `ATTACK` ~3000×/game and built nothing,
even though training "win rate" looked ok.

**Cause:** stable-baselines3 `PPO` does **not** call the environment's
`action_masks()`. Invalid actions (attack with no army, train with no barracks)
were selectable; the argmax collapsed to ATTACK. The training win rate was an
illusion of stochastic sampling.

**Decision:** switch to `MaskablePPO` (sb3-contrib). It consumes `action_masks()`
in both training and evaluation, so invalid actions are impossible and the
deterministic policy is coherent.

**Guard:** if you ever see the policy fixate on one invalid-looking action, check
you're using MaskablePPO and that `action_masks()` is correct.

---

## D2 — Reward *changes*, not *state* (delta rewards)

**Symptom:** with masking, the policy collapsed to `DO_NOTHING`. It built a bit of
economy then idled to a timeout win.

**Cause:** rewards were paid for *absolute state* every step (having population,
having resources). So sitting still with the starting economy already accrued
positive reward — an "idle and collect" exploit.

**Decision:** delta-based rewards. Pay once for each *positive change* — population
growth (+5), building completed (+3), unit trained (+12), strength gained (+6).
Doing nothing → ~0 reward. See `RewardCalculator.calculate_rewards` and the
`_get_snapshot` / `prev_state` delta tracking.

---

## D3 — The opponent must actually attack

**Symptom:** even with masking + delta rewards, still zero military. The
deterministic policy built ~15 economy buildings and won 100%.

**Cause:** the `NormalBot` opponent **never builds military and never attacks**.
In that environment, military is genuinely useless — a rational agent correctly
ignores it. No reward tweak can fix this without "bribing" the agent to do
something pointless.

**Decision:** create `AggressiveBot` — a military rusher that reliably builds
barracks + soldiers and attacks. Military must be *instrumentally necessary*, not
rewarded for its own sake. This is the right way to induce a behavior: make the
environment require it.

**Sub-lessons while building the bot:**
- Behavior-tree branches must **count in-progress buildings**, else the bot spams
  build orders during the multi-step construction delay (it built 13 lumberyards).
- The bot must build a sustainable minimal economy **before** military, or it goes
  broke (0 wood) and can never train.
- Aggression must **ramp**: a one-shot rush that destroys the warehouse at 300 s
  makes the game unwinnable before the agent can possibly react. The bot now waits
  `min_attack_time` (≈500 s) and an army threshold before attacking.

---

## D4 — It's a hard-exploration problem; use behavioral cloning

**Symptom:** vs the aggressive bot the agent lost ~99% and *still* never built
military (barracks in ~1/10 stochastic episodes).

**Cause:** the military build is a long action chain (barracks → wait 240 s →
train several soldiers) and **partial military is punished** (1–2 soldiers still
lose the game + you paid the resource cost). There's no local gradient toward
military — it's an all-or-nothing cliff. Higher entropy and bigger milestone
bonuses did not help.

**Key reframing:** the timeout win is scored with **military weighted 100×**, so
the agent mostly *reaches* timeout and loses the tiebreak for having zero
military — it doesn't need to survive a rush, just to *have* an army. And we
already had a scripted policy (AggressiveBot) that wins **9/10** vs the hard bot.

**Decision:** **behavioral cloning warm-start** (`agents/district/pretrain_bc.py`). Record the
scripted expert's `(obs, action)` decisions, supervised-train the policy to
imitate them, then RL fine-tune. This injects the winning strategy and bypasses
exploration entirely. Result: BC policy builds barracks 15/15, army ~6, wins
**100%** deterministically.

---

## D5 — Disable the diversity/monotony penalty

**Symptom:** RL fine-tuning from the perfect BC policy **degraded** it (military
4.8 → 0.6, win 100% → 21%). Crucially, a 100%-winning game scored **−2060**.

**Cause:** the reward had a monotony penalty (−1.0/step when the last 5 actions
match). But `DO_NOTHING` is the correct action ~99% of steps (only ~15 build
actions per 3600-step game). So the penalty dominated: `~3500 × −1.0 ≈ −3500`,
drowning the +100 victory. RL correctly followed the (inverted) gradient and
dismantled the winning policy.

**Decision:** zero the diversity/monotony rewards (config v5). Action collapse is
now prevented by masking + delta rewards, not by punishing repetition. After this,
a winning game scores **+1164**.

**Guard / general rule:** if the agent optimizes reward but *loses*, sum the
per-step reward of a KNOWN-good policy. If winning scores negative, the shaping is
inverted — fix the reward, don't fight the agent.

---

## D6 — Shorten the horizon (helped, but was NOT the cause)

**Symptom:** even with the reward sign fixed, RL fine-tuning still decayed military
(4.8 → 0.6) and drifted toward endless economy expansion.

**Hypothesis (partly right, insufficient):** `decision_interval = 0.5 s` →
**3600 steps/game**; with `gamma = 0.99` the terminal victory (+100) discounts as
`0.99^3600 ≈ 0` — invisible to earlier actions. Also, delta rewards favor
*continuous* building (economy expands forever at +3/building; military saturates
at ~6 units, collected once).

**Change made (kept):** `decision_interval` 0.5 s → **2.5 s** (720 steps/game) and
`gamma` 0.99 → **0.999** (`0.999^720 ≈ 0.49`, so the win signal survives). Both are
now CLI flags on `agents/district/train.py` and `agents/district/pretrain_bc.py`.

> Reducing `decision_interval` from 60 s → 0.5 s (earlier, "for more control") was
> a mistake for a build-order game. 2.5 s is the right scale.

**Outcome:** did NOT stop the decay (military still 5.5 → ~1, win 100% → 34%).
The real cause was D7. The horizon change is still correct and was kept.

---

## D7 — BC trains the actor but NOT the critic (the real cause)

**Symptom:** fine-tuning from a perfect BC policy always degraded it, no matter how
the reward or horizon was tuned.

**Measurement (the smoking gun):** on the BC policy, the critic predicted
`V ≈ 1.1` while the actual discounted return was `≈ 554`. The value function was
off by ~500×, sitting at random init.

**Cause:** `behavioral_clone()` minimizes `-log π(a_expert | s)` — a loss on the
**actor** only. The value head never receives a target, so it stays random. PPO then
computes advantages `A = return − V(s) ≈ 554 − 1 ≈ +553` for *every* state: huge,
un-baselined, uniformly positive. The policy gradient reinforces whatever was
sampled (including exploration noise), washing out the sharp BC military sequence.
This is the classic BC→RL pitfall: warm-start the actor, forget the critic.

**Fix (two parts, both required):**
1. **Critic warm-up** (`warmup_critic()` in `agents/common/bc.py`): roll out the BC
   policy, compute discounted returns-to-go, and regress the value head to them.
2. **Separate actor/critic feature extractors** (`share_features_extractor=False`).
   With a *shared, frozen* extractor the critic only reached `explained_var ≈ 0.29`
   (it had to predict values from features optimized to pick actions). With its own
   `vf_features_extractor` it reaches **0.886** — and the actor is provably untouched
   because only critic-branch params are optimized.

**Result:** critic predicts 552.9 vs actual 554.3 (corr 0.994); actor still 12/12 wins.

---

## D8 — Ship the BC policy: there is no headroom to fine-tune into

**Symptom:** with a calibrated critic, fine-tuning finally *retained* military
(2–5 instead of collapsing to ~0.1) — but win rate still fell 100% → 44%.

**Realization:** the BC policy already wins **100%**. Against this opponent the
scripted expert is optimal, so **the only direction RL can move is down**. The
entropy bonus (0.03) actively pushes the policy off a sharp, fragile optimum, and
the reward landscape has no gradient pulling it back.

**Decision:** the **BC policy is the Phase-1 deliverable**
(`checkpoints/district/district_agent_bc_v1.zip` — 10/10 deterministic wins vs `normal`,
`aggressive_medium`, and `aggressive_hard`; military 6–7).

RL fine-tuning only adds value when BC is *mediocre* and there is room to improve.
To create that headroom we need a stronger opponent — i.e. **self-play** (roadmap
Phase 2), where the scripted expert is no longer the ceiling.

---

## D9 — What we built is the DISTRICT tier, not the macro brain

**Claim corrected.** Earlier docs (and my own summaries) called the trained agent
"the macro brain". Wrong.

The simulator models a faction as one undifferentiated settlement. **In a
one-district world the three tiers collapse into a single agent.** The agent's
*action space* (`BUILD_FARM`, `TRAIN_SOLDIER`, `ATTACK`) is district-tier; only its
*objective* (win the game) is macro-tier. The macro's characteristic actions — found
a district, allocate sims, transfer forces — are all no-ops with one district, so the
macro brain was never "postponed": **it was not buildable.**

**Consequences:**
- `NormalBot` / `AggressiveBot` and the BC policy are **district-tier prototypes**.
  `AggressiveBot` is literally a scripted settlement policy — which is what we decided
  civilian districts should be (utility AI).
- **Today's observation is already a district observation**, so the refactor is mostly
  renaming `Faction` state → `District` state and making `Faction` a container.
- Infrastructure (env, masking, delta rewards, BC + critic warm-up, session logging)
  is tier-agnostic and transfers 100%. Policy **weights** do not (22-feature
  single-settlement observation), but retraining is ~3 min.

**Decision:** self-play on the flat sim (old Phase 2) is **deprioritized** — it would
polish a tier whose action space is about to be restructured. Next is the
`Faction → District` refactor (architecture.md §7 Stage 1), using the BC policy as a
regression oracle.

---

## D10 — `Faction` aggregates over districts: write-through at 1, read-only at N

Stage 1 moved every piece of settlement state (`resources`, `buildings`,
`population`, `units`, …) off `Faction` and onto `District`. `Faction` keeps the
same public API, now as an **aggregate over `self.districts`**.

The trap: the old API is used as an **lvalue** all over the codebase and the tests —
`faction.resources['wood'] += 10`, `faction.population -= 1`,
`faction.military_strength = 50`. An aggregate that returns a freshly-summed dict
would accept those writes and **silently discard them**. Every one of those sites
would keep passing its assertions while the simulation quietly stopped working.

**Decision — make the aggregate's writability depend on whether it is ambiguous:**

| districts | `faction.resources` | `faction.population = x` |
|---|---|---|
| exactly 1 | **the district's own dict, by identity** → writes land | writes to that district |
| 2+ | `_AggregateMapping`, a summed read-only view → **raises `TypeError`** | **raises `TypeError`** |

With one district the simulator is byte-for-byte the old one (the regression oracle
holds: 10/10 vs `normal`, `aggressive_medium`, `aggressive_hard`). The moment a
second district exists, every ambiguous faction-level write becomes a **loud crash
pointing at the district to address instead** — never a silent no-op. This converts
"the Stage 3 macro will silently corrupt state" into a compile-time-ish error.

**Corollaries:**
- `can_afford` is `any(district.can_afford(...))`, **not** pooled totals: stock is
  per-district and two half-full warehouses cannot jointly fund one building.
  Logistics (Stage 4) is what will let them.
- `is_defeated` is `not any(d.has_warehouse)` — losing a warehouse is a wound, not
  death. Generalizes the old `warehouse_count == 0` for free.
- The warehouse stays a **normal entry in `buildings`**, not a separate field:
  combat razes a random building, and that is precisely how a district (and, when
  it is the last one, the game) is lost.
- `District.found()` seeds a district with a warehouse and *nothing else*; the bare
  constructor seeds the full starting settlement, which is only correct for the
  district a faction starts the game with.
- `RewardCalculator` needed **no change**: it reads only faction-level aggregates,
  and rewards are genuinely faction-scoped (the objective is winning). Per-district
  reward shaping belongs to Stage 3, when the macro is the thing being trained.

---

## D11 — Size every lever against the episode, or it is unlearnable

Adding age cohorts and policies exposed the same bug three times, in three different
constants. **A rate expressed "per hour" is meaningless until you check how many hours
an episode lasts.** An episode is 1800 s = **0.5 h of game time**.

| Constant | Was | Per episode | Verdict |
|---|---|---|---|
| `BIRTH_CHANCE_PER_TURN` | 0.1 /h | **0.05 births** | one birth every 20 games — fertility, `population_rate`, `block_families` and `BUILD_HOUSE` were all decisions with no consequence |
| `CONSCRIPTION_RATE_PER_HOUR` | 2.0 /h | **1 soldier** | costs −25 happiness for the whole game to buy one soldier: strictly dominated, so no agent could ever want it |
| `ADULT_DURATION_SECONDS` | (600 s, from the game) | whole population retires mid-episode | the game's clock is not the simulator's |

Raised to 12/h, 12/h, and 1800 s. All three are **uncalibrated by necessity** — there
is no finished game to fit them against (§7's calibration protocol is still owed).
But uncalibrated is not the same as arbitrary: a lever the agent cannot feel inside an
episode is not a lever, it is dead code with a happiness cost.

> **Rule.** Before adding an action, compute the effect size it produces in one
> episode. If it rounds to zero, the agent will correctly ignore it and you will
> mistake that for an exploration failure.

### Measure a lever before blaming the agent

BC v2 (obs 29, `Discrete(27)`, policies exposed) never once used a policy. The
tempting diagnosis was bad exploration. Instead we **forced each lever on and measured
it** against the scripted expert (20 episodes, deterministic):

| Lever forced on | vs medium | vs hard | final pop |
|---|---|---|---|
| none (baseline) | 12/20 | 13/20 | 8.4 |
| **L2 `population_rate=INCREASE`** | **16/20** | **15/20** | 11.8 |
| L1 `minimum_food` | 7/20 | 4/20 | 7.0 |
| L4 `forced_conscription` (at 2/h) | 5/20 | 4/20 | 6.4 |

The agent was **right** about two levers and **wrong** about one. `minimum_food` is an
emergency lever: rationing costs happiness and buys food the district does not need
when food is abundant — being net-negative in plenty is the design *working*. After
rescaling, `forced_conscription` became the trade-off it was meant to be: **0/20 vs
medium, 13/20 vs hard** — it razes the economy, so it loses on the timeout score and
wins the fights.

### The fix was to teach the expert, not to tune PPO

RL fine-tuning the policy-blind BC (400k steps) made it **worse** deterministically
(hard 12/20 → 13/20 but medium 11/20 → 9/20, with the stochastic win rate collapsing
60 % → 12 %) and still never touched a lever. D8's logic held, once inverted: the way
to install a skill we can *state* is to put it in the expert and clone it.

`AggressiveBot` now sets `population_rate=INCREASE` on turn one. Since it is also the
opponent, the buff is symmetric — the benchmark got harder, not softer.

### Two bugs the cohorts exposed

Splitting `population` into kids/adults/elders made two long-standing defects visible.

**1. The elder compartment was a one-way sink.** Adults retired, and nobody ever died.
A two-hour game ended with 19 elders and 1 adult, and a labour force permanently below
its starting value. Old age has to close the loop or the demography has no steady
state. Fixed with `ELDER_DURATION_SECONDS`; the population now settles instead of
collapsing.

**2. Soldiers were charged to the workforce twice.** `TrainingManager` removed the
recruit from the population, and then `get_available_workers()` computed
`population - military` and removed him *again*. Pre-existing, and invisible while
every head counted as a worker. With cohorts it became lethal: the BC agent trains ~7
soldiers out of 10 people, so it played the whole game with **zero workers and no
economy**, and still won — which tells you how little the economy was contributing.

```
before fix (workers over an episode):  0.00 → 0.00 → 0.00 → 1.29
after  fix:                            2.95 → 4.64 → 6.04 → 6.39
```

`get_available_workers()` is now `adults + elders × ELDER_PRODUCTIVITY`: enlisting
already removed the soldier from `adults`.

**Result (deterministic, 20 episodes/opponent):**

| | normal | aggressive_medium | aggressive_hard | uses levers |
|---|---|---|---|---|
| BC v2 (policy-blind) | 20/20 | 11/20 | 12/20 | no |
| BC v3 (levers, no old age) | 20/20 | 13/20 | 17/20 | yes |
| **BC v5 (`district_agent_bc_v5.zip`)** | **20/20** | **14/20** | **18/20** | yes |
| BC v2 + RL 400k | 20/20 | 9/20 | 13/20 | no |

> The Stage-1 oracle ("BC policy wins 10/10") is **retired**: cohorts changed the
> dynamics on purpose, and the observation (22→29) and action space (20→27) both grew,
> so `district_agent_bc_v1.zip` no longer loads. The new regression baseline is BC v5's
> row above.
>
> **Still owed:** the agent uses exactly one lever. `minimum_food` is an emergency
> lever it never faces an emergency for, and it never drafts even when losing. Whether
> that is optimal is untested — the honest check is a scripted expert that drafts under
> threat, measured the way the lever table above was measured.

---

## D12 — The episode never reached a conclusion, and the seed never reached the sim

Two findings that invalidate more of D11's numbers than they confirm.

### The seed did not reach the simulator

`RealTimeRTSEnv.reset(seed)` called `super().reset(seed)`, which seeds `self.np_random`
— a generator the simulator never touches. Births, starvation and combat building damage
all draw from the **global** `np.random`. So episodes were never reproducible from
their seed:

```
same policy, deterministic, same seeds:   run 1 = 16/20    run 2 = 19/20
```

It went unnoticed for the whole life of the project because `BIRTH_CHANCE = 0.1/h`
meant ~0.05 births per episode: the simulation was accidentally near-deterministic.
Raising the birth rate (D11) exposed it. Fixed by seeding the global RNG in `reset()`.

> **Every lever measurement in D11 was taken with ±3/20 of noise.** Re-measured with
> seeding, `population_rate=INCREASE` shows **no** win-rate effect over the expert
> baseline (12/20 vs 12/20 at the default horizon). The "+5/20" reported there was
> noise. `forced_conscription`'s decay with horizon (below) is real, and monotone.
>
> BC v5 beating BC v2 (35/40 vs 24/40 over 40 seeded episodes) *does* survive — but it
> is **confounded**: v5 was also trained after the old-age and double-counting fixes,
> in a different simulator. It is not evidence that the lever helps.

### Nothing is ever conquered

With BC v5, deterministic, 20 episodes:

| horizon | wins | by conquest | by timeout |
|---|---|---|---|
| 1800 s (default) | 17/20 | **0** | 20 |
| 3600 s | 18/20 | 2 | 18 |
| 7200 s | 20/20 | 3 | 17 |

**Every game at the default horizon ends on the clock.** The winner is then chosen by
`calculate_victory_score` — an invented development score in which
`military_strength × 100` dwarfs `population × 30` and `buildings × 5`. So the agent
is not learning to win the game. It is learning to maximise a proxy, and the proxy
says: build soldiers, ignore everything else. That is exactly the policy we observe.

### The horizon decides which levers are good

Same expert, same seeds, one lever forced on, vs `aggressive_hard`:

| lever | 1800 s | 3600 s | 7200 s |
|---|---|---|---|
| baseline | 12/20 | 12/20 | 9/20 |
| `population_rate=INCREASE` | 12/20 | 13/20 | 9/20 |
| `forced_conscription` | 12/20 | 10/20 | **5/20** |

Conscription mortgages the future, so its value falls monotonically as the future gets
longer. **A lever's worth is a function of the horizon**, which means the episode
length is not a neutral training knob: a short episode systematically rewards
extractive policies and hides the compounding ones.

An episode is 1800 s and the mean adult life is 1800 s: **exactly one generation per
episode.** A newborn matures at t+300 s and would become a soldier later still. The
agent can barely feel a birth, and cannot feel a grandchild.

### Open, and load-bearing

1. **Time-limit bootstrapping is wrong.** `env.step()` returns `truncated=False`
   always, and sets `done=True` at `max_game_time`. Gymnasium/SB3 therefore treat the
   horizon as a true terminal state and do **not** bootstrap the value function there.
   With `gamma=0.999` and a timeout that fires in 100% of episodes, the critic is being
   taught that the world ends at 1800 s.
2. **Conquest should be the only victory.** Timeout should be `truncated`, not a scored
   win. Until then the agent optimises `calculate_victory_score`, not the game.
3. **Pick the episode/generation ratio deliberately.** It should be ≥ 3 if the
   demographic levers are meant to be learnable at all.

---

## D13 — One anchor: the adult working life

Every duration and rate in `config.py` was an independent number, and they had drifted
into incoherence. They are now **derived from a single anchor**, and the ratios are
stated rather than implied (see the TIME SCALE block at the bottom of `config.py`).

```
ANCHOR   A = ADULT_DURATION_SECONDS = 900 s   (mean adult working life)

kid      0.25 A      episode    10 A = 9000 s   (ten generations)
elder    0.50 A      decisions  1440            (dt = 6.25 s)

standard building  0.05 A   →  20 of them per working life
barracks/warehouse 0.10 A
soldier            0.013 A
```

**Generations and decisions are not independent.** What the agent experiences is
`decisions per generation` (144), and every action's duration is a fraction of one.
Raise generations while holding the decision budget fixed and construction collapses
into a single step — at 10 generations on the old 720-decision budget a soldier would
finish in the very step that ordered him, and an action that resolves instantly is not
a decision, it is a purchase. Both move together, and `config.py` now **asserts** it:

```
standard building ≥ 5 decisions   (7.2)
soldier           ≥ 1.5           (1.8)
childhood         ≥ 20            (36)
```

`gamma` must track the episode too: at 1440 steps `gamma=0.999` discounts the victory
reward to 0.24. Use **0.9995** (effective horizon 2000 steps, victory at 0.49).

Rates are declared **per working life** and converted, so they cannot drift again:
5 births, 8 conscripts, 2× starvation deaths per life. Consumption is *derived* from
the farm's output so that one farm feeds a stated 12 people.

### The bug that made the old numbers unreadable

`farm` was declared in **both** `PRODUCTION_RATES` (90/h) and `RESOURCE_PROCESSING`
(2/h). `ResourceManager` tests `RESOURCE_PROCESSING` first, so
`PRODUCTION_RATES['farm']` was **dead code** and a farm quietly produced **45× less**
than the number every reader (and every previous analysis, including this document's
first draft of D13) took at face value. `config.py` now deletes the shadow and asserts
that no building is both a raw producer and a processor.

### What the retiming bought

| | before (1800 s episode) | after (9000 s = 10 A) |
|---|---|---|
| generations per episode | 1.0 | 10.0 |
| decisions per generation | 720 | 144 |
| one farm feeds | 4 people | 12 people |
| lumberyard payback | ~0.5 A | 0.20 A |
| economy vs opening stock | 30 wood produced vs 200 held | 750 vs 200 |
| **expert games won by conquest** | **0/20** | **20/20 normal, 6/20 hard** |

Doing nothing now kills you: the population plateaus at the farm's carrying capacity,
and food is a real constraint. Famine is possible, which is the precondition for
`minimum_food` ever being worth pulling. The expert's games *end*, in 3.4 generations
against `normal` and ~6 against the aggressive bots, and it now genuinely **loses**
9/20 to `aggressive_hard`.

### More room made the proxy problem worse, not better

BC v7 (`district_agent_bc_v7.zip`), retrained on this scale, wins **20/20, 20/20,
20/20** — and does it **by timeout in 60/60 games, with zero conquests**, while the
expert it cloned conquers and sometimes dies. The clone matches the expert on 98.1% of
actions; the 2% that differ are the attacks, and they decide everything.

It has learned to turtle. `calculate_victory_score` weights `military_strength × 100`
against `buildings × 5`, so holding an army and never spending it out-scores winning —
and a longer episode simply gives it more time to hoard.

> Retiming made conquest *possible* (the expert does it). It made turtling *more
> profitable*. Until the timeout is a `truncated` and the only victory is razing every
> warehouse (D12, open item 2), we are paying the agent to stockpile soldiers, and it
> will keep taking the money. **This is now the single highest-impact fix left.**

---

## D14 — Conquest is the only victory; the timeout is a truncation

Acting on D12/D13. Three coupled changes:

1. **`check_game_over` sets a winner only on conquest** — one faction with a warehouse
   left. `calculate_victory_score` and the whole timeout-scoring block in
   `realtime_simulator.step()` are deleted, along with `calculate_timeout_rewards` and
   the dead `MAX_TURNS` branch. A stalemate has `winner = None`.
2. **The env returns `terminated` vs `truncated` honestly.** Conquest terminates
   (value 0 after). The clock truncates: `RealTimeRTSSimulator.is_out_of_time()`, and
   SB3 receives `TimeLimit.truncated=True` + `terminal_observation`, so the critic
   **bootstraps** across the horizon instead of learning the world ends at 9000 s.
3. **The win-fast bonus is measured against the episode**, not a 200-hour `MAX_TURNS`
   the sim never reaches (which had made it a near-constant added to every victory).

Every rollout loop (`while not done`) now stops on `terminated or truncated`, or it
would spin forever past the horizon. The obsolete tests that pinned the scored victory
(`TestImprovedWinConditions`, `test_winner_determined_at_max_time`) are deleted, not
patched: a test that locks in a deleted objective is worse than no test.

**Result — BC v8 (`district_agent_bc_v8.zip`), deterministic, 30 episodes, and now the
metric is CONQUESTS, because a timeout wins nothing:**

| | conquests | defeats | timeouts | attacks/game |
|---|---|---|---|---|
| BC v7 (scored timeout) | **0** | 0 | 60/60 | ~0 |
| **BC v8 (conquest only)** | **27 / 23 / 22** (normal/med/hard) | 0 / 4 / 5 | 3 / 3 / 3 | ~12 |

The agent went from never attacking to winning by assault in ~80% of games, and it now
genuinely *loses* the ones it misplays — which is the point: a real objective has real
downside. The critic's `explained_var` also jumped (0.81 → 0.93): a coherent terminal
value is easier to fit than a hoarding proxy.

> D12's three open items are closed. The turtle is gone. The next question is no longer
> "why won't it fight" but "are the levers pulled situationally" — which is finally a
> question worth asking, because the agent is now optimising the game and not a proxy.

---

## D15 — A lever's value lives in the composition, not in the lever

With conquest as the only victory (D14) and the RNG seed finally reaching the sim
(D12), the question from the start of this work is finally answerable: **are the policy
levers situationally useful?** Three experiments, each stricter than the last.

**1. Force one lever on the scripted expert, always (40 episodes, conquests):**

| lever | vs medium | vs hard |
|---|---|---|
| baseline | 28 | 28 |
| population_rate=INCREASE | 25 | 25 |
| minimum_food | 13 | 13 |
| forced_conscription | 10 | 10 |

Every lever is neutral-to-harmful. Activating them *situationally* (ration when food
is low, draft when out-strengthed, grow when affluent) does not help either — all
within noise of baseline. First read: the levers are worthless here, because the
expert conquers in **~3 generations of 10** (median 3.2), long before slow demography
diverges.

**2. But that read was wrong, because the expert is the wrong subject.** Ablate the
same lever from the *trained BC policy* (v8) at runtime — mask the policy actions so
`population_rate` stays MAINTAIN — and its win rate collapses:

| BC v8, same weights | normal | medium | hard |
|---|---|---|---|
| levers allowed | 38/40 | 37/40 | 35/40 |
| levers masked off | 31/40 | 15/40 | 15/40 |

**The lever more than doubles the trained policy's conquests.** The contradiction with
experiment 1 is the lesson: a scripted rusher does not re-tune its build and attack
timing around the extra mouths, so the lever is dead weight on it. A learned policy
composes the lever *with* its economy and army timing, and there the value is real.
Forcing a knob on a fixed strategy measures the knob; letting a policy learn around it
measures the strategy. Only the second is the question we care about.

> Consequence: a lever's worth cannot be judged by ablating it on the expert. It must
> be judged on a policy free to adapt. This is why "measure the lever" (D11) was
> necessary but not sufficient — it finds levers that are *unconditionally* good; it
> misses levers that are good only in composition, which is most of them.

**3. Confounds honestly noted.** Masking a lever forces the agent's second-choice
action, so experiment 2 slightly over-states the drop. And BC v9 (retrained with a
lever-free expert) conquered 20/40 hard vs v8's 35 — suggestive but confounded, since
v8 and v9 clone different demonstrations. The clean ablation (experiment 2, same
weights) is the load-bearing evidence, and it is unambiguous in direction.

**Outcome.** The expert keeps setting `population_rate=INCREASE` on turn one; v8 stays
the deliverable. A brief detour removed the lever and trained v9 — reverted once the
ablation showed the lever earns its place. The other three levers remain unused by v8
and, on this evidence, unhelpful in a 1v1 rush; they are infrastructure for the long
game (macro tier, multi-district survival over many generations), where the ~3-gen
horizon that neutralises them no longer applies.

---

## D16 — Grain is not food: the processing chains become necessary

The simulator already had rich processing chains (`grain -> mill -> flour -> bakery ->
bread`, plus cattle, pottery, smelting…), and they were **strictly dominated dead
code**: converting 1 grain to bread lost 52% of its mass, and bread fed a person no
better than raw grain, so no rational player built a mill. The chains existed; the
value did not — the same pattern as the inert policy levers (D10).

The fix, from the game's own `resources_system.md`: **grain and flour are
intermediates, nobody eats them.** The population lives on end products.

- `EDIBLE_FOODS = ['bread', 'meat']` (Phase 1; the Rust doc also has fish, fruit,
  vegetable, which are Phase 2). `get_total_food` sums these, not grain.
- Two routes, a real choice: the **bread chain** (farm→mill→bakery, 3 buildings,
  scalable) vs **hunting** (one shed, one worker, feeds ~8, immediate). A rusher hunts;
  a long game bakes.
- The chains were re-balanced. Loaded processing rates were per-batch tokens (bakery
  3 bread/h against 375/h consumption — 100× too small, because never used). Each stage
  now matches a farm's throughput with a 0.9 per-stage loss; one bakery feeds ~9.7.
- **A varied diet (≥2 foods) grants +10 happiness**, which feeds fertility and
  productivity. This is what gives bread value even where hunting alone would suffice —
  the reward for running two chains instead of the single cheapest one. (`FOOD_VARIETY_
  BONUS` had sat unused in config, like the policy costs.)
- **Starting food stock**, proportioned to the founding population (~2 generations of
  bread+meat), so the settlement eats while its first food building goes up. Grain
  alone would now starve it.

**The scripted bots had to learn to eat, or the BC clones a corpse.** `AggressiveBot`
builds a hunting shed in its minimal economy; `NormalBot` gained a food-security branch.
(NormalBot still starves — but from its pre-existing bug, not this change: it spends all
its wood on houses and quarries and never builds a lumberyard, so it stalls before
reaching any food branch. That is one of the 13 baseline failures, and out of scope.)

**Result — BC v10 (`district_agent_bc_v10.zip`), deterministic, 30 episodes.** The game
is now genuinely harder and deeper: food is a real constraint, and outcomes are
balanced rather than one-sided.

| vs | conquests | defeats | timeouts | pop-min | hunting sheds | diet variety |
|---|---|---|---|---|---|---|
| aggressive_medium | 12 | 11 | 7 | 4.0 | 0.7 | 1.5 |
| aggressive_hard | 10 | 14 | 6 | 4.2 | 0.8 | 1.6 |

The agent survives (pop never hits zero), builds its food source, and fights — winning
some, losing some. A real economy with a real failure mode.

> **Phase 2** adds **vegetables** (garden) and **fruit** (orchard) — both agricultural,
> no map dependency. Each new food is a new action.
>
> **Fish is deferred to Stage 2 (the map).** Fishing presupposes water bodies — rivers,
> sea — which are a feature of the region graph that does not exist yet. A `fishing_dock`
> without water on the map would be a fake abstraction; the fisherman profession + water
> access falls out naturally once the map exists. Tracked in the Stage-2 backlog.

---

## D17 — Opponent names cleaned; curriculum starts at a real threat

- `NormalBot` -> `TestingBot` (`'testing'`): a balanced economy bot kept only as a test
  fixture. It never attacks and (pre-existing bug) starves itself, so it taught the
  agent nothing.
- `aggressive_easy/medium/hard` -> `easy/medium/hard`: every combat bot is aggressive,
  so the prefix was noise. `AggressiveBot` (the class) keeps its name — it *is* the
  aggressive one.
- The **curriculum now opens at `easy`**, a combat bot that attacks late, so the agent
  learns defence from stage one — instead of the old opening against the non-attacking
  economy bot.
- Old names (`normal`, `aggressive`, `aggressive_hard`, …) still resolve, as deprecated
  aliases in the registry, so existing tests and the CLAUDE.md oracle command keep working.

---

## D18 — Food variety (Phase 2): vegetables and fruit, tiered variety bonus

Extends D16's food model with two new edible foods and their sources, both agricultural
(no map dependency; fish stays deferred to Stage 2):

- **vegetable** <- `vegetable_garden`: cheap, fast, low-density (feeds ~4). The
  affordable second food.
- **fruit** <- `orchard`: slow, feeds only ~2 — a **luxury** whose value is variety,
  not calories.

**Variety happiness is now tiered** (was a single 2-food threshold, +10): each distinct
food beyond the first adds `FOOD_VARIETY_BONUS_PER_TYPE` (5), capped at 15. So bread +
meat is +5, a four-food table is +15. This is what makes the orchard worth building — a
distinct food, even at poor calories. The agent gets a `food_variety_ratio` observation
so it can see the state the bonus reads (obs 29→30; actions 27→29).

**Unlike the policy levers (D15), variety pays even on the scripted expert.** Ablation,
40 episodes vs hard:

| expert config | conquests | defeats | timeouts |
|---|---|---|---|
| 1 food (hunting only) | 11 | 13 | 16 |
| 2 foods (+ vegetable garden) | **14** | 10 | 16 |

The reason it works where `population_rate` didn't: a garden gives food **and**
happiness at once, so it helps without the strategy having to re-tune around it. The
expert now builds a garden in its minimal economy; BC v11 clones it (variety ~2.1,
gardens ~0.8). The agent takes the cheap second food (garden) but not the luxury
(orchard) — correct for a rush; the orchard is for a long game that rewards happiness
over many generations.

**Deliverable: `district_agent_bc_v11.zip`.** Only v1 (historic Stage-0) and v11 kept.

---

## D19 — Agent-based population, step 1: the sim exists

Reading 2 of the "manage the sims" direction: the *simulator* models individuals; the
sims will self-manage (scripted); the RL agent governs from above. This is the first
structural step — replace the cohort ints with a list of `Sim`.

**Feasibility spike first (the gate).** Before rewriting anything, measured the cost of
per-sim updates vs the ~183 us cohort step:

| population model | per-step | RL 400k |
|---|---|---|
| cohort ints (baseline) | 183 us | 73 s |
| 50 sims/faction, vectorised (numpy) | +1.1x | — |
| 50 sims/faction, plain Python objects | +1.3x | — |
| 100 sims/faction, Python objects | +1.6x | — |

Even the naive OOP path (a list of `Sim` objects) is cheap to ~100-200 sims/faction.
The bottleneck is *not* the sims (§9 confirmed backwards: the cohort step already costs
183 us for its own reasons; 13 real sims added **4 us**). Rule that falls out: keep the
per-sim update light/vectorisable; reserve Python loops for rare events (births,
profession changes), which are O(few) per step, not O(N).

**Step 1 shipped.** `District.sims: List[Sim]` (a `Sim` is just an `age` for now).
Cohort membership — `kids`/`adults`/`elders` — is *derived* from age thresholds
(`KID_END`/`ADULT_END`/`LIFESPAN`), so everything downstream that read the cohorts keeps
working. The three cohort transitions (mature/retire/die), which were constant-rate
accumulators, collapse into one `age_sims()` pass: membership is now a function of age,
not a flow. Starting adults are spread deterministically across the adult age band so
they don't age out in a synchronised wave.

**Results.** Suite still 13 baseline. Real step cost **187 us (+2%)**, RL 400k ≈ 75 s.
Demography is healthy over 10 generations (oscillates 5-11, more lifelike than the
smooth cohort flow). BC v11, trained on the cohort model, does **not** degrade — it
improves (18/18 conquests vs 9/8), because the age-based dynamics free up adults
differently. No retrain needed for the structural step; profession/observation changes
in step 2 (professions) will need one.

> Next: professions + skills on the `Sim`, building slots that require a profession,
> and sims that self-train and self-assign (the unit tier, scripted) — then RL levers
> over that world (schools, incentives).

### Step 2: professions and skills (structure + acquisition)

The `Sim` is now grown as **components**, mirroring the Rust `Person { core, stats,
needs, skills, family }` — so gender, kinship, needs, stats are added as fields/
sub-dataclasses later without disturbing the rest. This answers the "structure must
scale with per-sim information" requirement: AoS-with-components, not a flat blob and
not full ECS (over-engineering for a Python sim). The spike already showed AoS is cheap
enough.

`simulator/professions.py` mirrors the game's skill system, restricted to professions
tied to buildings that exist: L1 (farmer, hunter, woodcutter, miner, miller…), L2
(mason, carpenter, potter — need 50% of an L1 prerequisite), L3 (baker needs miller,
blacksmith needs miner). `BUILDING_PROFESSION` maps each building to the profession its
work needs; `can_learn` enforces the prerequisite chain.

`District.update_professions()` matches working sims (adults + elders) to building slots
by the food>resource>processing priority, preferring sims who already have or can learn
the slot's profession; assigned sims adopt it and grow that skill on the job. Sims
specialise coherently (woodcutters where there are lumberyards, miners at quarries…).

**Step 2a changed only the sims, not the economy** — productivity kept keying off
head-count, so the oracle stayed green (BC v11 at 18/18) while professions developed.

### Step 2b: skill drives productivity

`assign_work()` now sets each building's productivity from WHO staffs it: `(slots
filled / slots) x (mean effective skill)`, where a novice contributes `SKILL_FLOOR`
(0.6) and a master 1.0, elders scaled by `ELDER_PRODUCTIVITY`. **Mentorship**: the most
skilled worker present lifts everyone's learning rate (`MENTORSHIP_MULTIPLIER`), so
skill accumulates across generations instead of every cohort restarting from zero —
verified (mean skill climbs 0.67 → 0.89 and stays high). Skill mastery is anchored to
the adult life (~0.3 of it, working, to master a skill).

**A demographic trap this exposed.** Skill-reduced output feeds fewer people, which cuts
fertility, which fails to replace the founders when they age out — a death spiral,
visible when food is thin. Two fixes: a higher novice floor (0.6, so a fresh settlement
of novices can still feed itself) and a **founding age pyramid** (`seed_founding_
population` spreads ages over the whole lifespan, so births and deaths stagger from turn
one instead of the founders dying in one wave). With adequate food the population is
healthy (oscillates 5-11); with thin food it still collapses — correct, and the agent's
job to prevent.

**Result.** Retrained **BC v12** recovers to **18/18** conquests on both aggressive bots
(BC v11, trained pre-skill, had dropped to 8/8 — the model got harder, and the agent
re-learned it). Suite 13 baseline; step cost ~205 us. Population runs lean (pop-min ~3):
the agent conquers before demography bites, and no game times out.

> Open: pop-min ~3 is thin — the age model is more volatile at small populations than
> the smooth cohort flow was. And the RL levers over this world (schools to speed
> training, incentives) are not built yet: the agent lives with the sims' self-training,
> it cannot yet invest in it. That is the next step. Deliverable: `district_agent_bc_v12.zip`.

---

## D21 — The first RL lever over the sim world: schools

The agent could observe the sims but not act on them. The **school** is the first lever
that does: an edificio that invests an adult as a **teacher** (who then produces nothing)
to raise the whole district's skill faster, and reach the L2/L3 professions sooner.

For this to matter, base learning had to be slowed (a sim working *alone* can't master a
skill within a life, `_SKILL_MASTERY_FRACTION` 0.3→1.5) — otherwise skill saturates on
its own and a school buys nothing. Now expertise must be *transmitted*: mentorship and
schools are what make it accumulate. A staffed school multiplies learning by
`SCHOOL_TRAINING_BONUS`.

The agent gets a `skill_ratio` observation (mean worker skill) so it can see its human
capital and judge whether a school is worth building. Action space 29→30, observation
30→31.

**Measured to pay**, like variety (D18) and unlike the policy levers (D15): forcing one
school on the scripted expert lifts conquests **13→16 / 40** and cuts timeouts (16→10) —
the teacher lost to production is repaid by everyone else being more skilled. The expert
now builds one school; **BC v13 clones it** (builds ~0.8 schools, holds mean skill ~0.70,
conquers 13/13). Suite 13 baseline. Deliverable: `district_agent_bc_v13.zip`.

> This closes the agent-based arc for the district tier: sims exist (D19), have
> professions and skills that drive output (D20), and the agent can now **invest** in
> them (schools). Still open: finer incentives (directing training toward a specific
> profession), the rest of the `Sim` components (needs, gender, kinship — the structure
> is ready), and pop-min stays thin (~3) — the age model is volatile at small scale.

## D22 — A soldier is a man who stopped being a worker

`Sim` grew `gender` (M/F), a per-sim `life_scale` (age-threshold jitter, so cohorts stop
dying in synchronised waves), and `pregnant_remaining` — only adult women conceive, they
gestate ~0.1 of an adult life and work at `PREGNANCY_PRODUCTIVITY` meanwhile.

The reason gender had to exist was **pop-min**, and the cause was not demography. Training
a soldier consumes an adult (`remove_adult`), and the rusher was converting its *entire*
civilian population into army: civilians hit 0 by generation 1, nobody worked, nobody was
born, and the district died with an army standing in it. The fix is that **only men
enlist**. Women stay civilian and fertile, so the population can regenerate while the army
is raised: **pop-min 1.6 → 5.0**.

This makes MEN the binding constraint on any army, which is why the arrival forecast (D25)
caps itself on `adult_men` and not on `adults`, and why `ATTACK_ARMY_STRENGTH` had to be
*lowered* (6→4) rather than raised — men-only recruitment makes armies smaller, so the old
threshold blocked attacks outright.

> The same confusion bit conscription: `conscript_district` looped on `adults > 0` while
> `remove_adult()` only takes men, so once the men were gone it kept adding soldiers it
> could not take anyone for — **320 soldiers out of 7 adults**, conjured from the women.
> The draft is now gated on `adult_men`.

## D23 — An attack is a march, not a click

`ATTACK` no longer resolves where it is ordered. It schedules a `MarchingArmy` that fights
on **arrival**, `ATTACK_TRAVEL_TIME_SECONDS` (~90 s, 0.1 of an adult life) later. An army
already on the road cannot be re-sent. This is the placeholder for real map movement: there
is no spatial layer yet, so the distance is one fixed march rather than a path.

It lives on `GameState`, not on the simulator, because a march is **state**: an opponent
could see the column coming, and — as D25 found — a bot must be able to ask "is my army
already out?" before ordering another attack.

## D24 — Conquest was a lottery; make it something you can INTEND

`WAREHOUSE_DESTRUCTION_THRESHOLD = 2.0` ("need 2x strength to destroy a warehouse") had sat
in `config.py` since the beginning, **read by nothing**. So the only way to take a district
was the incidental damage roll: each won battle deleted a *random* building at 30%, and you
won by grinding through every garden and shed until the warehouse happened to come up.

That is a lottery whose odds get *worse* as the game gets richer — and D18/D21 had been
busily adding buildings (garden, orchard, school). A defender with 13 buildings gives ~1/13
per damage event: **~43 won battles to expect one warehouse hit**, while the march (D23)
caps you at one attack per 90 s. Conquest quietly decayed into something you could not do
on purpose, and the expert's win rate decayed with it.

It also inverted the strategy: since only the NUMBER of won battles mattered and never the
SIZE of the army that won them, raiding forever with a token force strictly dominated
committing a real one.

Now an attacker that arrives with `>= 2x` the defender's bonus-adjusted strength **takes the
district** (razes the warehouse); a narrower win only damages a building, and can never hit
the warehouse. An undefended district simply falls — which is what makes leaving home
uncovered a real risk.

Measured on the scripted expert vs a fixed opponent, 40 seeds:

| | conquests | defeats | timeouts | pop-min |
|---|---|---|---|---|
| lottery (as it was) | 12/40 | 16 | 12 | 4.00 |
| overwhelming force takes the district | **28/40** | 12 | **0** | **5.35** |

Every game now reaches a result, and conquest is back in the 65-80% band it used to be in.

## D25 — Commit on the state you will MEET, and clone the 1% of actions that decide games

Three findings, one negative — the negative one is the point.

**1. The bot was shouting at an empty road.** The attack branch fired on "I have N strength",
so while the army marched (D23) the tree short-circuited on it every single step: the order
was a no-op, but it still consumed the decision. The bot built nothing and trained nobody for
the whole journey — **12% of all its decisions** — while the defender used those same seconds
freely. The tree now falls through when the army is away (`army_is_home`), so the army grows
while it walks; the env masks `ATTACK` in transit for the same reason.

**2. The fixed strength floor is the wrong criterion; the arrival forecast is the right one.**
"Attack once I have 4" was tuned for instantaneous combat and knows nothing about the enemy:
it marched into defences that outnumbered it, and sat at home waiting for a number it did not
need when the enemy was empty. It is replaced by a forecast of **both** armies at the moment
of arrival (current army + the training already paid for that lands inside the window, capped
by the men who can actually enlist — D22). Attacks/episode 9.2 → 12.8, defeats 16 → 12 per 40.
The floor survives only as a *difficulty knob*: a high floor makes a bot easy, because it
hoards an army instead of spending it.

**3. NEGATIVE RESULT: modelling the defender's reinforcement during the march is worthless
here.** This was the obvious thing to build — the defender gets 90 s to react, so credit it
with the soldiers it could train — and it was built, ablated, and removed. It changed the
forecast by **0.02 strength** and not a single decision. The reason is not a bug: mid-game the
defender is chronically broke — **~5 wood (a soldier costs 10) and ~1.7 adult men**. It
*cannot* reinforce, so crediting it with reinforcements only made us attack less. Demanding
the full capture margin (2.0x) at arrival was worse still: 29 → 23 conquests, because waiting
to be overwhelming means not attacking, and a rusher does not grant the time.

> Half the levers we have added turned out neutral or harmful (D15, and now this). The
> discipline that catches them is ablation against **conquests**, never against a proxy.

**4. The rare action is the decisive one, and BC was silently dropping it.** Fixing (1) had an
unlovely side effect: the old bot's ATTACK spam was, by accident, most of the ATTACK signal in
the demonstrations. With the spam gone, `ATTACK` is **0.93%** of demo steps — and an unweighted
clone reaches **99% action-match by never attacking at all**: it copies the economy perfectly,
hoards an army of 19, and conquers **0/40**. Accuracy was measuring everything except the thing
that wins.

Two changes, both required:
- the agent must be able to SEE the march (`army_marching` in the observation, 31→32). Without
  it the expert's trigger is not a function of the observation, and no clone can learn it:
  ATTACK recall stayed at **0.0%** even when the loss was re-weighted.
- the BC loss is weighted by inverse class frequency (`--class-weight-power`, default 1.0).

Ablated on conquests (vs easy/medium/hard, 40 seeds each), not on accuracy:

| weighting | ATTACK recall | action-match | conquests |
|---|---|---|---|
| none (p=0) | 27% | 99.3% | 14 / 14 / 10 |
| sqrt-inverse (p=0.5) | 59% | 98.4% | 17 / 22 / 16 |
| **inverse (p=1.0)** | **92%** | 85.9% | **25 / 24 / 18** |

Full inverse frequency spends aggregate accuracy on the only decisions that end games.

**Deliverable: `district_agent_bc_v14.zip`** — 25/26/18 conquests of 40 vs easy/medium/hard,
**zero timeouts**, pop-min 5.67-5.88. It matches the scripted expert (25/27/20, pop-min
5.47-5.72), which is exactly what a clone should do. Note vs `hard` the ceiling is ~50%: the
expert IS the hard bot, so that matchup is a mirror.

> **A bigger network is not the lever.** The 256×256 MLP already fits the expert to 99%
> action-match when unweighted — capacity was never the constraint. The BC bottleneck was the
> loss and a missing observation, and the remaining headroom is in the EXPERT and the game, not
> in the net. Self-play, not parameters.

## D26 — The army that leaves is not home to defend

D23 made an attack take time, but the army never actually *went* anywhere: its strength
stayed in the district, defending, while it simultaneously fought on the other side of the
map. The army was in two places at once, and soldiers trained *behind* a marching column
teleported into its battle on arrival. Every forecast built on that (D25) was forecasting a
battle that could not happen.

A `MarchingArmy` now **carries its units**. They come out of the district when the order is
given and they are gone until they walk back (one march there, one march home, so ~180 s
away in total). Three consequences, and they are the whole point:

- **Leaving is a risk.** Attack, and your district is empty. Since an undefended district
  falls (D24), a raid can lose you the game outright.
- **Raiding an army that is out is a real move** — the counter-punch exists.
- **Reinforcements stay home**, which is exactly where they are needed; they no longer
  materialise at the front.

The bots' commit criterion was re-ablated from scratch under the new rules, because the old
verdict was reached in the incoherent world. It **survives, and hardens**: committing on a
predicted win (1.0x) is right, and demanding more is progressively suicidal — you never
leave, so you never take anything, and you die at home.

| commit margin (expert vs fixed `hard`, 40 seeds) | conquests | defeats |
|---|---|---|
| **1.0x (predicted win)** | **18** | 22 |
| 1.5x | 13 | 26 |
| 2.0x | 5 | 29 |
| 2.6x (the capture threshold) | 3 | 29 |

Attacks per episode fell from 3.2 to **0.5**: committing the army stopped being a chore and
became a decision you make once, and might not survive.

**Deliverable: `district_agent_bc_v15.zip`** — 25/19/16 conquests of 40 vs easy/medium/hard,
zero timeouts, pop-min 5.45-5.97, ATTACK recall 84%. It matches the scripted expert
(22/20/18), and the difficulty ladder is finally monotone (easy > medium > hard), which it
was not before.

> Open, and now visible: the ATTACK action commits the WHOLE garrison. There is no way to
> leave a home guard, because the action space has no "attack with N" — so "how much do I
> send" is a decision the agent is not yet allowed to make. That, not the forecast, is the
> next real lever. Note also the critic warm-up now explains only ~0.50 of the return
> variance (was 0.92): outcomes genuinely got more volatile, which matters before any RL
> fine-tune.

## D27 — The world: one map, shared by every tier

Until now nothing had a position. Buildings existed as counts, the enemy was a number, and
the march (D23) was a constant standing in for a distance nobody had measured. `simulator/
map.py` gives the simulator the ground it was missing — **one grid, built once, for both
brains**: the civil district places its buildings on it, and the military camp tier (not
built yet) will move on the same tiles. An Area is centred on a warehouse or on a camp, and
the map does not care which — which is precisely why it is built once, here.

It mirrors the game: `TileType` and the elevation model come from `src/components/map/tile.rs`
and `game_data/map.json`. The training map is 64×64 where the game's is 300×300, and that is
not an approximation to apologise for — nothing depends on the tile COUNT. What must be
faithful are the RATIOS (the same discipline as the time scale, D13):

    DISTRICT_RADIUS : SETTLEMENT_SEPARATION      how much world a district owns, vs how far
                                                 away the enemy lives
    SEPARATION × MARCH_SECONDS_PER_TILE          = the march, in seconds — DERIVED, not typed

**Terrain is the everyday economy; deposits are the strategy.** Wood comes from woods, stone
from the mountainside, crops from grass — all abundant, so a district on decent land runs a
normal economy and geography constrains it without taxing it. Iron and gold are veins: they
are somewhere specific or they are nowhere. A mine without a vein and a well without water
are *impossible*, not merely expensive, and they fail before the costs are charged. (The
game's `spawners/resources/veins.rs` is entirely commented out — the sim is defining deposits
first, and the game can follow.)

**The map is rotationally symmetric, and this is the load-bearing decision.** The first
version let each faction pick its own site, and the one that settled first took the better
ground: over 200 seeds, faction 0 had water in 84% of its plots against faction 1's 58%. That
is a permanent, invisible head-start, and it would have quietly poisoned every measurement
made on this simulator — we have already lost a session to reading an artefact as a result
(D25). Competitive RTS maps are mirrored for exactly this reason. Each half is congruent to
the other under a 180° rotation and the two settlements sit at opposite points, so the plots
are the SAME ground: measured yield difference **0.0000**, identical iron and water access. A
mirror match is now a coin flip *by construction* rather than by hope.

**Neutrality on introduction.** A new subsystem must not silently re-balance the game the day
it lands, or nothing measured before it can be compared with anything measured after. The
march comes out at 92.8 s against the 90 s constant it replaces, and the auto-placer finds
sites yielding 0.95 of the old flat rate. The expert scores 18/18/16 conquests of 40 against
22/20/18 before the map — within noise, with a slight consistent dip that matches the ~5%
yield tax. Zero timeouts, pop-min 5.5-6.1.

**A plot must be VIABLE, and that is a requirement, not a preference.** A settlement is
founded once and lives with its ground for the whole game, so a plot that cannot run an
economy is a game lost before the player has made a single decision. Before
`plot_is_viable` existed, plots ranged from **37 to 168 buildable tiles**, and some had no
forest, no mountain or no clay at all — meaning no timber, no stone or no bricks, and
therefore no bakery, no dormitory and no forge, *ever*. Variety in what a plot is GOOD at
(iron, rich clay, deep forest) is the whole point of a map; variety in whether it is VIABLE
is just an unearned loss.

**Scoring a plot: what it can feed and build, not how many tiles it counts.** The first
`_plot_score` weighted plot SIZE alongside everything else, and size promptly swamped the
rest: settlements were founded on big empty stretches of sand. Terrain variety is now the
dominant term (25 points per must-have terrain present), with deposits and buildable area as
minor tie-breakers.

> **What the map is NOT yet: an advantage.** `BUILD` still names only a type, so the WORLD
> chooses the tile — and it chooses the best one available. Geography can therefore forbid
> (no vein, no mine) but cannot yet be exploited *badly*. That is deliberate: it keeps the
> economy neutral so that when the pointer action `BUILD(type, tile)` arrives, its value can
> be **measured against this baseline** instead of being confounded with the map's mere
> existence. Step 2 is the factored action (one head over types, one over tiles, both masked)
> plus a local-patch observation — and that is where the retained `TransformerExtractor`
> finally becomes the right tool over a set of tiles, rather than the wrong one over 32 flat
> scalars.

**Deliverable: `district_agent_bc_v16.zip`** — 19/22/12 conquests of 40 vs easy/medium/hard,
zero timeouts, pop-min 5.6-6.2. The agent is still BLIND to the map (its observation is
unchanged), which is exactly why its score did not move: it cannot yet use what it cannot see.

## D28 — The agent says WHERE. It works. It is worth nothing (yet).

Step 2 of the map: the district agent stops naming a building and letting the world find it
a site (D27), and starts choosing the ground itself.

**The machinery** (`env.spatial=True`, kept beside the flat agent so the baseline survives):
- **Observation**: the district summary (32 scalars) *plus* its plot as a 16×13×13 stack of
  feature maps — terrain, deposit, richness, occupancy, distance. A grid is what a convolution
  is for, so `MapExtractor` is a small convnet over the plot next to an MLP over the summary.
  (The `TransformerExtractor` is still the wrong tool here: it is for a variable-size SET of
  entities, not a fixed lattice.)
- **Action**: `MultiDiscrete([type, tile])` — one head for WHAT, one for WHERE, each masked.
  Not flattened into `Discrete(30 × 169)`: that shares no structure, so every (type, tile)
  pair would have to be learned separately and nothing learned about a tile would transfer.
- The tile mask is generic (on the map, and free). It cannot say "a mine needs iron", because
  the heads are masked independently — so naming impossible ground is a legal move with a bad
  outcome: the build is refused, uncharged, and the decision is spent. The agent learns the
  geology instead of being shielded from it.
- BC learns both heads, with the tile term **switched off on the ~95% of steps that place
  nothing** — training it there would be fitting noise.

**It works.** After rebalancing the extractor (the first convnet flattened to 10,816 features
against the summary's 128 and simply drowned the economy out — action-match 50%), the clone
reproduces its teacher: action-match 86%, ATTACK recall 95%, **tile-match 97%**, and it places
at **0.95** of the auto-placer's yield (0.96). `district_agent_bc_v17_spatial.zip`.

**And it buys nothing.** Conquests 18/15/14 of 40 vs the flat baseline's 19/22/12 — no gain,
and `medium` is worse. That much was predictable (a clone cannot beat the teacher it clones;
the teacher IS the auto-placer). So the real question was asked directly, by ablation:

> **Does WHERE matter at all?** Hold the strategy, the opponent and the seeds fixed, and move
> only the ground: every building on its BEST legal tile, versus every building on its WORST.

| placement | yield | conquests (easy / medium / hard) |
|---|---|---|
| best (what the world picks today) | 0.96 | 18 / 18 / 16 |
| **worst legal tile, every time** | **0.77** | **19 / 17 / 17** |

**Nothing.** A 20% cut in raw production changes the outcome by noise. The spatial action space
is, today, a cost with no prize — and it is worth saying plainly rather than shipping it as
progress.

**Why**, and this is the useful part:
1. **The strategic deposits lead nowhere.** The mine produces `iron_ore`, smelting turns it
   into `iron_ingots`, and **nothing in the game consumes them**. Soldiers cost wood and grain.
   So the one thing on the map that is genuinely scarce and worth fighting over — iron, gold —
   cannot be converted into anything that wins a game.
2. **Production is not the binding constraint.** Armies are capped by adult MEN (D22), not by
   wood. Placement scales resource yield, and resource yield is not what runs out.

> **The lever to pull next is not a better spatial policy — it is giving the ground something
> to be worth.** Connect the veins to what actually decides games: iron → weapons/armour →
> soldier strength. Then "does my plot have iron" becomes a real strategic difference, the
> symmetric map (D27) guarantees both players face the same question, and `BUILD(type, tile)`
> finally has a prize worth choosing. Only then is RL fine-tuning from this warm start worth
> the compute — the machinery is built and waiting.

## D29 — The war economy: an army must be EQUIPPED (and the iron tier does not fit)

Stage A of [the roadmap](roadmap_war_economy_macro.md): primary resources are refined into
derivatives, and a soldier cannot exist without his kit.

```
forest    -> lumberyard -> wood      -> carpentry -> wooden_weapon  -> SOLDIER      (1.0)
iron vein -> mine       -> iron_ore  -> blacksmith -> iron_weapon
                                                   -> armour        -> MAN-AT-ARMS  (1.8)
```

`TRAIN_SOLDIER` now costs `{grain, wooden_weapon}` — the wood that used to be in the soldier's
price tag simply moved one step upstream, to the carpenter, so the district pays the same wood
per spearman. The carpentry stopped being a workshop and became **the armoury**: no spears, no
army.

Two more dead wirings turned up on the way, exactly like `WAREHOUSE_DESTRUCTION_THRESHOLD` (D24):
the recipes for smelting and weapon-smithing were already in `game_data`, but keyed by RECIPE
name rather than owned by any building, so they had never once run; and the blacksmith's price
tag demanded 20 `clay_bricks` — a clay pit and a pottery, two buildings and two workers whose
only purpose was to make the forge legal — which made the forge unbuildable in practice.

### What it revealed: the army was the demographic valve

With a weapon in front of every soldier, recruiting slowed — and the rusher promptly **bred
itself into a famine**: population 16, food to zero, everybody dead by mid-game. It had been
surviving on a single hunting shed only because its army was eating its population (every
soldier trained consumes an adult man — D22). Remove the sink and the Malthusian truth shows.
The expert now builds a second food building, and pop-min *rose* from 5.5-6.1 to **6.8-7.6**.

### NEGATIVE: the iron tier does not fit in a district this small

**Zero men-at-arms, in every episode measured.** The mine is dug and the forge is built, and
the forge never works a single hour — because it is never *staffed*. The arithmetic is brutal
and it is the whole finding:

> A district has **5-8 adults**. Its buildings ask for roughly **15 worker slots** (farm 2,
> hunting 2, garden 1, lumberyards 4, quarry 2, carpentry 1, school 1, mine 1, forge 1). The
> forge is last in the queue and there is nobody left to send to it.

Worse, before the staffing priorities were fixed the mine *outranked the carpentry*, so a
district that found iron lost its armoury to it: peak army **0**, against 4 on a plot with no
iron at all. Iron was not a prize, it was a trap.

The three-stage chain (mine → smelter → blacksmith) was cut to two, because a three-worker
refining chain costs three men to upgrade the two that remain — a losing trade at this scale.
That helped and did not save it. The ingot stage stays in the data, deferred.

### Where that leaves the map

**D28's negative stands: the ground still has no prize.** Not because the design is wrong —
iron doubling the strength of each man is exactly the right shape, since men are what a
district runs out of — but because **the district cannot staff the industry that would use it**.

> The blocker is not the map and not the economy. It is **district scale**. Making iron
> reachable means a settlement that can afford a specialist: more food, more people, more
> hands. That is roadmap **Stage C** (bigger districts and a bigger map), and this measurement
> says it is a *prerequisite* of Stage A's payoff rather than a sequel to it. The equipment
> chain is built, correct, and waiting for a district big enough to run it.

Expert after D29: 23/19/19 conquests of 40 (from 18/18/16), pop-min 6.8-7.6 — but peak army
fell from ~4-5 to ~1.5-2, and timeouts returned. Wars are now fought by a handful of equipped
men, which is truer, and thinner.

## D30 — Closing the production circles (and growing the district to staff them)

An audit of every resource — who makes it, who consumes it — found most of the game's lines
OPEN: things produced that nothing used (`stone_bricks`, `leather`, `wool`, `wood_logs`),
things required that nothing made (`gold_ore`, `herbs`), and recipes that no building owned.
A resource nobody consumes is not an economy, it is scenery; a job nobody needs is not a job.
That is why the district could never fill its higher trades.

Every circle is now closed (verified: zero dead ends, zero orphans):
clay → bricks → the buildings that need them; stone → dressed stone; wood → timber and spears;
cattle → meat, leather, wool → the tailor → **clothes, which the PEOPLE wear** (the one
derivative made for the district itself rather than for an army).

**GOLD IS GONE**, from the map as well. Nothing consumed it and nothing produced its ore: the
golden veins invited the district to spend real miners digging for something it could not
spend. It returns the day there is something to buy with it.

The district itself was grown (population 12 → 20, food per building up, births up), because
D29's blocker was arithmetic: ~15-20 worker slots against 5-8 adults meant the forge was never
manned. The MAP grew with it (64² → 96²) and its deposits now come in **clusters** — ore in
seams, woods in stands. Sprinkling them uniformly made every plot statistically identical, and
then WHERE could not matter, whatever the placement rules said. Terrain is cut by QUANTILE so
the world keeps its proportions at any size (fixed thresholds had erased every grain of sand
the day the map grew).

## D31 — Careers, not job titles: experience on the job is the main road

`skills_system.md` says it in one line, and we had not implemented it:

> "Level X can be accessed when getting enough experience on a specific Level X-1 job, **or
> pairing with another worker** with that experience."

Two roads: the job itself, and the man beside you. **The school is not one of them** — in the
game it is only "a boost for future learning speed". It had drifted into being the unlock,
which was papering over a modelling error.

Three divergences from the game's table, all fixed: **clay digging** is its own L1 career
(a potter ripens out of a *digger*, not a generic miner); **iron mining is L2** (50% stone
mining); **blacksmith and smelter are L3**, above iron mining. So the forge sits three careers
deep — stone → iron → anvil — which is the real reason it was never staffed in D29. Not a
shortage of hands. A shortage of CAREER.

**And the allocation rule was backwards.** Workers were assigned in order of BUILDING PRIORITY,
so the experienced men were handed to the low-tier jobs first (they matter more day to day) and
the specialists arrived at an empty pool: the pottery, the tailor and the forge sat at 0.00
productivity for entire games while the qualified men were out cutting wood. Now the SCARCE
QUALIFICATION is allocated first — only a man who has dug clay for years may work the pottery,
and *anyone* can dig clay. The promotion-and-backfill the game describes falls out of that
ordering instead of out of luck.

> Worth recording: I claimed the promotion mechanism did not exist. The test refuted me — it
> did, and it worked. The bug was the ordering, not the absence.

## D32 — The civil district makes PEOPLE and GOODS. It does not fight.

War is the military camp's job, and the camp is a different model (`agents/camp/`). This tier
had been trained against a rusher with conquest as the only victory (D14) — and it learned
exactly what it was paid to learn: barracks first, development never. Measured at the end of
D31: six or seven adults sat unemployed in a district that had never built a clay pit, because
the trades ranked below the army. You cannot ask one brain to win a war and raise a
civilisation and then be surprised that it does the first.

`env(objective='economy')`:
- **one faction.** An opponent it cannot fight is not an opponent, it is noise in the reward.
- **no military actions in the space at all** — not discouraged from war, incapable of it.
- **the score is a weighted sum of the final state**: `1 × people + 5 × trades practised
  (weighted by tier) + 8 × distinct goods`. Beating the bot means out-scoring it on the same
  ground.
- **the step reward is the DELTA of exactly that score**, so shaping and objective can never
  disagree — and an idle district is paid nothing for what it already had (D2).
- the expert is the new **`SettlerBot`**, which builds a settlement and never fights.

Two real bugs surfaced: a lone district was **declared the winner on step zero** (the conquest
check saw one surviving faction and crowned it), and the well's water requirement deadlocked
the build plan (191 identical orders in a row). The well now yields the same wherever it is
dug — the water table is under every tile alike.

> **The teacher is not ready, and it is marked as such.** The SettlerBot still starves some
> seeds and practises too few trades; `tests/unit/test_settler.py` pins what it must do, with
> the two unmet requirements as STRICT xfails — the day it is fixed those tests fail as
> unexpectedly passing, and force us to notice. A clone can only reproduce its teacher (D28
> measured this twice), so training BC on a mediocre teacher does not give a mediocre agent to
> improve — it gives an agent that has learned the wrong habits well.

## D30 — Closing the production circles, and growing a district that can staff them

D29 ended on a number: the district had **5-8 adults** and its buildings wanted **~15 worker
slots**, so the forge was never manned and the iron tier could not exist. This is the answer
to that, and it began with an audit of every production line — which is where the real
problem turned out to be.

### The audit: most of the economy was not an economy

| line | state before |
|---|---|
| clay → pottery → **clay_bricks** → bakery | the bakery already COST bricks, but nobody built a pottery, so **bread — the only food that scales — was unreachable** |
| stone → stone_cutter → **stone_bricks** | consumed only by `defensive_wall_stone`, which **had no build action**. A dead end. |
| cattle → **leather**, **wool** | consumed by recipes that **no building owned**. Dead ends. |
| carpentry → **wood_logs** | consumed by nothing. |
| mine → **gold_ore** | **nothing produced it.** The golden veins on the map were scenery. |
| `armor`, `weapons`, `clothes`, `potions`, `iron_ingots` | produced by recipes with no owning building, or consumed by nothing |

A resource nobody consumes is not an economy, it is scenery — and **a job nobody needs is not
a job**. That is why the district could never fill its higher trades: they led nowhere.

### What was closed

Everything now follows `documentation/gameplay/{resources,building}_system.md`:

```
clay   -> pottery      -> clay_bricks   -> bakery, dormitory
stone  -> stone_cutter -> stone_bricks  -> blacksmith, watchtower, stone wall
wood   -> carpentry    -> wood_logs     -> barracks, wooden wall
                        + wooden_weapon -> soldier
cattle -> leather + wool -> TAILOR      -> clothes -> WORN BY THE PEOPLE (happiness)
iron   -> blacksmith   -> iron_weapon + armour -> man-at-arms
```

The tailor's line is the one that ends **in the population** rather than in the army: clothes
wear out, so a district stays clothed only while somebody keeps making them. Without a sink
like that, wool and leather were scenery.

**Gold is deleted** — from the recipes and from the map. A dead resource is worse than a
missing one: it invites the agent to spend real miners digging for something it cannot spend.

### Two circular deadlocks, both self-inflicted, both measured

1. **The school cost bricks.** School needs bricks → bricks need a potter → the potter needs
   a miner's SKILL → skill is what the school teaches. The potter sat at 0.00 productivity all
   game while the clay piled up. The school is the **bootstrap**: it cannot cost what it unlocks.
2. **The trades were gated on skill, not on hands.** The higher professions have prerequisites
   (potter needs miner, baker needs miller, smith needs miner, tailor needs breeder). A district
   with twenty idle adults and no school still cannot man a pottery. The school moved into the
   CORE build order, ahead of the trades that depend on it.

### The district that can staff itself

Anchored on a ratio, as always: `TARGET_ADULT_WORKERS = 20` — enough hands for the whole
workshop chain — and everything else derived to reach it. Starting population 12 → 20, food per
farm 24 → 40 (via the bread chain, which is why the clay circle had to be closed first), births
per life 5 → 7, housing per building raised, the dormitory made buildable.

| | before D30 | after |
|---|---|---|
| peak population | ~16 | **22.3** |
| peak adults | **5-8** | **14.3** |
| pop-min (expert, 40 eps) | 6.8-7.6 | **11.6** |
| conquests (easy/med/hard) | 23/19/19 | 21/19/20 |

### The map, grown to hold it

96×96 (from 64×64), and two generation bugs fixed on the way:
- **Deposits now come in CLUSTERS** — seams and stands, not salt-and-pepper. Uniformly
  sprinkled deposits make every plot statistically identical, and then WHERE cannot matter
  whatever the placement rules say. (The first attempt was a percolation process above its
  threshold: one "forest" seam claimed **3370 of 4114** forest tiles. A seam has a *size*.)
- **Terrain is cut by QUANTILE, not by absolute noise thresholds.** Smoothed noise does not
  keep its distribution when the grid grows, so the fixed thresholds silently rewrote the world
  the day the map did: at 96×96 one seed came out with **22% water and no mountains at all**.
  And the noise is now folded BEFORE anything is decided from it, or the quantiles are computed
  over a world that is then half thrown away.
- A district's **territory is capped** at `DISTRICT_RADIUS` (enforced by `tiles_within`). This is the
  hard limit that will force a faction to FOUND ANOTHER DISTRICT rather than grow one settlement
  without end — the seam the macro tier is built on.

### Where it actually got to — and where it stops

The chain now *runs*, and the skill gate works: **school 40/40, pottery 40/40** (staffed 37/40),
mill 19/40. Conquests rose to **23/40**.

But it stops there. **Bakery 2/40, stone cutter 2/40, cattle shed 2/40, tailor 0/40, forge 0/40 —
and men-at-arms 0/40, still.** The district is now big enough to STAFF the deep trades and is not
rich enough to BUILD them inside one episode: ten generations is not long enough to raise a
settlement of twenty, arm it, fight a war, and put up eight more workshops.

> **The remaining lever is throughput or time, and it is a real choice, not a bug.** Either the
> economy produces more per hour, or the episode is longer (more generations), or the expert
> must choose between developing and fighting rather than doing both. Whichever is picked, it
> must be picked deliberately and measured — and until one is, the iron tier and the D28
> placement ablation stay out of reach.

- **MLP instead of Transformer** for the feature extractor: the observation is a
  flat 22-vector with no sequential/entity structure, so attention is wasted
  compute. MLP is smaller and faster. (Transformer is kept for the future spatial
  brain, where attention-over-entities is the right tool.)
- **Single source of truth for normalization** (`env_config.json`). Two files
  previously used different constants (÷1000 vs ÷500), badly scaling observations.
- **Reward magnitudes bounded** (roughly ±100). An earlier config had victory
  =10000 vs continuous rewards ~0.1, creating a sparse-reward problem; everything
  is now on a comparable scale.
- **Combat left abstract on purpose.** See [architecture.md](architecture.md) §3 —
  it's a swappable oracle for a future tactical brain, not something to polish now.

---

## Housekeeping done alongside these fixes

- **Removed the diversity/monotony machinery entirely** (code in
  `RewardCalculator`, the `record_action` call in the simulator, and the
  `diversity_rewards` config block). It was disabled-but-present — a footgun that
  had already caused D5. Verified the BC policy's episode reward is unchanged (1063).
- **`test_trained_model.py` re-implemented the observation** with stale constants
  (`/1000`, `/100`, `/500`) while the env uses config values (`/500`, `/50`, `/50`),
  and it called `predict()` with **no action mask** and `delta_time=60.0`. It now
  drives `RealTimeRTSEnv` directly, so observations/masks always match training.
  Its `test_against_opponent` was also renamed to `evaluate_against_opponent` (it is
  a utility, not a pytest test — the `test_` prefix caused a collection error).
- **`train_ppo.py` / `train_curriculum.py` used plain `PPO`** (which silently ignores
  action masks, see D1) and `train_curriculum.py` imported a class that no longer
  exists (`LargerTransformerExtractor`). Both now use `MaskablePPO` + `MLPExtractor`.
- **Curriculum sequence** extended to `normal → aggressive_easy → aggressive_medium
  → aggressive_hard`.

## D33 — The district gets ONE anchor, and food, housing and labour are derived from it

The yields and the housing were never sized against a target: they were tuned one at a
time, and the result did not add up. A full settlement asks for ~26 worker slots, and a
district of 40 people supplies about 18 adults — a **45% labour deficit**. No build order
can solve that, which is why the settler bot could only ever choose *which* failure to
have: starve, or never develop.

So the district gets one anchor and everything follows from it (the same discipline as the
time scale, D13, and the map, D27 — fix the ratios, derive the numbers, never hand-tune
the numbers):

- `TARGET_DISTRICT_POPULATION = 60` — **the anchor**: what a mature district *is*.
- `ADULT_SHARE = 0.45` — kids and elders are most of the other half.
- `FOOD_LABOUR_SHARE = 0.20` — feeding itself must not cost the district more than a fifth
  of its hands. Above that is a settlement that exists to eat, with nothing left over to
  become anything.
- `PEOPLE_FED_PER_FOOD_WORKER` then *follows*: ~18 mouths per worker on food. It is
  computed at **novice** skill on purpose — a fresh settlement is staffed by novices, and a
  food supply that only works once everybody is a master is one that starves the district
  through its first generation.

Everything else (what a person eats, what each food building yields, beds per house) is
written as a multiple of that, so the whole table moves together and cannot drift apart
again. The invariant `_SETTLEMENT_SLOTS <= TARGET_ADULT_WORKERS` is asserted at import: if
it ever fails again, the settlement is arithmetically impossible and no policy, bot or
agent can save it.

## D34 — Who gets assigned, and in what order: two queues, not one

A LEVEL-2+ building can only be worked by someone qualified for it; a LEVEL-1 job can be
worked by anyone. They are therefore **not competing for the same people**, and they must not
be filled from the same queue. It took two wrong answers to see it.

**First wrong answer: sort by building priority.** The low-tier jobs matter more day to day,
so they came first — and they took the experienced men. The specialists then arrived at an
empty pool, and the pottery and the forge sat at **0.00 productivity for whole games**.

**Second wrong answer: sort by trade, highest first.** This broke it the other way, and more
subtly: it starved the L1 jobs that *feed* the ladder. Nobody was left to dig the clay, so
nobody ever became a digger, so nobody could ever become a potter. The pottery stood finished
and empty in **6 settlements out of 6**, with not one grain of clay-digging experience
anywhere in the district.

**The rule:** the clay pit is the potter's school, and a career ladder needs its bottom rung
manned. So:

- **pass 1** — the qualified go to the workshops only they can work, highest trade first;
- **pass 2** — everyone else fills the ordinary jobs, food before the rest.

Combined with D35, each pass runs twice: once to give every building its *minimum* crew, and
once to fatten the crews with whoever is left.

## D35 — A crew has a MINIMUM and a MAXIMUM

`WORKERS_NEEDED` is the **maximum** crew — the full complement, at which a building runs at
100%. `WORKERS_MIN` is the smallest crew that can run it at all. Between the two, output is
proportional: half a crew is half the output. That much was already true.

What was *not* true is that the district filled buildings **to the maximum in priority
order**, emptying the worker pool before it reached the bottom of the list. Two quarries
took four men while the lumberyards stood empty — so a settlement in the middle of a forest
ran out of *wood* — and the clay pit and the pottery, further down the list, were never
manned at all in a whole game.

A district does not staff itself that way. It mans every building, thinly if it must, and
then fattens the crews it cares about with whatever is left over.

## D36 — Births must scale with the women who have them

D33 gave the district a target of 60 people and sized its whole economy on reaching it. It
never did: measured settler-vs-settler over 24 seeds, mean population *fell* from a founding 20
to ~16, and 2/24 districts went extinct — **with ~220 food in the warehouse and happiness at
85-90**. Not famine, not unhappiness: all three fertility gates stayed at 1.0 the whole game.

The cause was one line in `PopulationManager`:

```python
per_woman = self.fertility(district) / len(fertile)   # the bug
```

`fertility` returns `BIRTH_CHANCE_PER_TURN = BIRTHS_PER_LIFE(7) · _PER_LIFE_TO_PER_HOUR(4) = 28`,
which is *7 births over one woman's adult life* — a **per-woman** rate. Dividing it by the number
of fertile women cancelled that count, so the district's total birth rate was a **constant** ~28/h
(42/h under the settler's pro-natalist policy), independent of population. Deaths, meanwhile,
scale with population: a 1575 s lifespan means ~2.29·pop deaths per hour. A constant birth rate
against a proportional death rate has exactly **one equilibrium** — 42/2.29 ≈ **18** — and the
population was pinned there, unable to approach 60. The short lifespan makes turnover violent, so
that small equilibrium random-walked into the absorbing zero state.

**The fix is to drop the division**: every fertile woman draws against the per-woman rate, so the
district's births scale with its fertile women. `BIRTHS_PER_LIFE` was left at 7 — it was never
inflated to compensate, the division was the sole bug. Measured, same 24 seeds:

| | mean final pop | extinct | adults at end |
|---|---|---|---|
| before | 15.8 | 2/24 | ~9 |
| after | **51.1** | **0/24** | **29.5** |

The population now grows until food per capita falls to ~2.0 (the growth threshold), where it
self-limits — food becomes the binding equilibrium, which is the *correct* dynamic. Adults at end
(29.5) finally clear `TARGET_ADULT_WORKERS = 27`: the workforce D33 sized for exists at last.

This is a **balance change**, not a neutral refactor: it invalidates the civil district's trained
checkpoints, which learned against the pinned population. The relevant teacher is the SettlerBot
(D32) — the AggressiveBot is not used to train the civil tier — so retraining is the civil BC
run, not a military one.

Footnote for whenever the military camp tier is revived: the AggressiveBot's demographic valve
assumed the *old* dynamics (conscription eating adults faster than a suppressed birth rate
replaced them). That assumption no longer holds and its build/attack timing will need re-checking
against the corrected regime — but it is out of scope while the focus is the civil district.

## D37 — Population must track the jobs, not the food ceiling

D36 unpinned the population, and it promptly overshot: with the skill fix feeding the district
well, food per capita never fell to its growth threshold, so **nothing checked growth**. The
district ran to ~115 people against ~55 jobs — half of them idle, mouths that eat and do not
produce, dragging food per capita down and blocking the very development they were meant to
staff. The old regulator (food) had stopped biting, and the population grew against the wrong
constraint. It needs to grow against the **demand for labour**.

The mechanism is a feedback loop through happiness, plus two brakes on runaway fertility:

- **Unemployment lowers happiness.** `assign_work` now marks each sim `employed`; the share of
  working-age sims with no job costs happiness (`UNEMPLOYMENT_HAPPINESS_PENALTY`). Grow past the
  jobs and the district feels it.
- **Happiness is the MAIN gate on births.** The fertility floor dropped 0.5 → 0.15, so an
  unhappy district really does breed slowly. It stays 1.0 at BASE_HAPPINESS (the un-governed
  district is unchanged) and only starvation/prohibition reach exactly 0.
- **A post-birth cooldown** (`FERTILITY_COOLDOWN_SECONDS`): a woman spaces her children instead
  of conceiving back-to-back.
- **High parity decays fertility** (`HIGH_PARITY_FERTILITY_DECAY`): the first two children come
  at full rate, the third and beyond progressively rarer — growth is spread across many women,
  not carried by a few. (No kinship model yet, so the woman stands in for the couple.)

The malus must **dominate the happiness bonuses** (varied diet +15, clothes +8), or a
well-provisioned district stays content while half of it sits idle. Tuned on measurement (16
seeds): penalty 50 was too weak (~120 pop, 60% idle, violent boom-bust); **150** lands the
mature district near full employment (~16% idle) and halves the oscillation. Population now
settles where the district can employ it — which the dynamic build plan (D38) then raises by
building more jobs.

Balance change: invalidates trained checkpoints; retrain the civil BC run after D38. The
per-sim state (`children_born`, `fertility_cooldown_remaining`, `employed`) is pinned by
`tests/unit/test_demographics.py`.

## Open items

1. **Opponent quality ceiling** — the agent is only as good as its opponent, and the
   scripted expert is already optimal here (D8). Next: **self-play** to remove the
   ceiling and create headroom for RL.
2. ~~**Faction-0 turn-order advantage**~~ — **checked and closed (D25).** It does not
   exist. Reversing the order in which the two factions' actions are executed changes
   nothing (11/23/6 either way, 40 seeds), the founding population is fair over 400 seeds
   (adult men 3.47 vs 3.39, well inside the CI), and env and raw simulator agree exactly.
   A mirror match is a coin flip, so **any expert measured against its own bot is capped at
   ~50%** — worth remembering before reading a conquest rate as a skill level.
3. **NormalBot never builds military** — its behavior-tree military branch is
   unreachable (`has_good_economy` requires a lumberyard it often never builds).
   `tests/integration/test_normal_bot.py` correctly fails on this; the bot is unfixed.
4. ~~**`STARTING_FOOD_STOCK` is computed from constants that no longer exist.**~~ —
   **fixed and measured.** Found while de-sedimenting `config.py`: the file assigned
   `POPULATION_FOOD_CONSUMPTION`, `STARTING_POPULATION` and `PEOPLE_FED_PER_FARM` two or
   three times each, and the starting-stock lines sat *in the middle*, so they captured the
   **stale** values — a consumption of 18.75 (from `PEOPLE_FED_PER_FARM = 24`, later 74) and
   a population of 12 (later 20). The district opened with `bread: 56.25, meat: 56.25`:
   roughly **3x the food its own formula intends**, and a runway of 3333 s where the formula
   says 1800 s.

   Now re-derived, and sized in LIVES of eating so it moves with the time scale:
   `_FOUNDING_FOOD_LIVES = 2.0` gives exactly two adult working lives of food per founding
   head (1800 s of runway, against a hunting shed that takes 45 s to build).

   **Measured, 24 seeds, settler vs settler: the correction changes nothing that matters.**
   Districts starved 2/24 either way, population minimum, final population and trades
   practised are *identical* seed for seed; only the leftover stock at the end differs, by
   exactly the 51.75 removed. The founding stock was never the binding constraint — the
   district carried the surplus to the end untouched. The old value was wrong, not
   load-bearing.

   One thing it *was* holding up: `test_policies.py`'s `nominal()` fixture zeroed the meat to
   get a single-food diet, which also halved the calories, and only stayed above
   `MIN_FOOD_PER_CAPITA_FOR_GROWTH` because the stock was inflated. It now moves the meat into
   the bread instead — same calories, one food.

5. ~~**A civil district does not grow; it ages out.**~~ — **fixed and measured (D36).** The
   birth rate was divided by the number of fertile women, making it a district constant that
   pinned population at an equilibrium of ~18 against a target of 60, with 2/24 extinctions.
   Dropping the division lets births scale with the women who have them: mean population 15.8
   → 51.1, extinctions 2/24 → 0, adults at end ~9 → 29.5 (clearing `TARGET_ADULT_WORKERS`).

6. **D33's food economy has never been stress-tested at full population, and it shows.** With
   D36 the district finally reaches ~51 people, and two D33 assumptions crack under it:
   - **Trades practised fell** (4.5 → 3.4 over the same seeds). Food per capita now sits right
     on the growth threshold (2.02), so the district must keep most of its adults on food and
     has little slack for the L2+ workshops, while the many recent births dilute mean skill.
     The `PEOPLE_FED_*` anchors were set when population never grew past ~18, so they were
     never actually tested at 60. They need re-deriving against a district that reaches its
     target.
   - **`ADULT_SHARE = 0.45` is wrong.** The age pyramid (adult band 900 s of a 1575 s life)
     makes adults **~58%** of the population, measured, not 45%. D33's "~27 hands" was derived
     from the wrong share; the real figure is higher, which only widens the gap in the point
     above.
