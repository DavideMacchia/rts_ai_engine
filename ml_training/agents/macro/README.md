# Macro tier — not built yet

The empire-level brain. It decides **where to place warehouses** (each warehouse
founds a district), assigns sims to districts, and later allocates forces and sets
empire policy.

It cannot exist until the simulator has ≥2 districts to coordinate — see
[`docs/architecture.md`](../../docs/architecture.md) §2–4 and the roadmap (§7).

**Design already settled:**
- Map = attributed **region graph** (node + edge attributes; rivers on edges; fog of
  war per node).
- Actions are **node-scoped pointers**: `FOUND_CIVILIAN_DISTRICT(region)`,
  `FOUND_MILITARY_CAMP(region)`, `ASSIGN_POPULATION(district, n)`, `NO_OP` — masked by
  affordability/occupancy/reachability.
- Network = **attention over regions** (variable node count) — this is where
  `agents/common/extractors.py::TransformerExtractor` becomes the right tool.
- Trained with the district brains **frozen** (Stage 3).
