"""
Test Trained Model Against Scripted Opponents
Benchmark AI performance and watch games
"""

import os
import sys
import argparse
import numpy as np
from sb3_contrib import MaskablePPO

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_training_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, ml_training_dir)

from simulator.opponents import get_opponent, OPPONENT_POOL
from config.config_loader import load_env_config
from agents.district.env import RealTimeRTSEnv

# Import custom feature extractor (required for loading trained models)
from agents.common import extractors as _extractors  # noqa: F401  (registers MLPExtractor for unpickling)

DEFAULT_DECISION_INTERVAL = 2.5


def evaluate_against_opponent(model, opponent_difficulty, num_games=100, verbose=True,
                          env_config=None, decision_interval=DEFAULT_DECISION_INTERVAL):
    """
    Test model against a specific opponent.

    Drives RealTimeRTSEnv directly so observations, normalization and action masks
    always match training. (This used to re-implement the observation with stale
    constants, which silently fed the model wrong inputs.)

    Args:
        model: Trained MaskablePPO model
        opponent_difficulty: e.g. 'normal', 'aggressive_hard'
        num_games: Number of games to play
        verbose: Print detailed output
        env_config: Unused, kept for backwards compatibility
        decision_interval: Seconds of game time per agent step (must match training)

    Returns:
        Dict with results (win_rate, avg_game_length, etc.)
    """
    wins = 0
    losses = 0
    draws = 0
    game_lengths = []

    env = RealTimeRTSEnv(opponent_type=opponent_difficulty,
                         decision_interval=decision_interval)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing vs {opponent_difficulty}")
        print(f"{'='*70}")

    for game_num in range(num_games):
        obs, _ = env.reset(seed=game_num)
        done = False
        steps = 0

        while not done:
            action_masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, _reward, done, _truncated, info = env.step(int(action))
            steps += 1

        game_lengths.append(steps)
        winner = info.get('winner')

        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1

        # Print progress every 10 games
        if verbose and (game_num + 1) % 10 == 0:
            current_win_rate = wins / (game_num + 1)
            print(f"  Games {game_num + 1}/{num_games}: Win rate {current_win_rate:.1%}")

    win_rate = wins / num_games
    avg_length = np.mean(game_lengths)

    results = {
        'opponent': opponent_difficulty,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'total_games': num_games,
        'win_rate': win_rate,
        'avg_game_length': avg_length,
        'min_game_length': min(game_lengths),
        'max_game_length': max(game_lengths)
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"Results vs {opponent_difficulty}:")
        print(f"{'='*70}")
        print(f"  Wins:      {wins}/{num_games} ({win_rate:.1%})")
        print(f"  Losses:    {losses}/{num_games} ({losses/num_games:.1%})")
        print(f"  Draws:     {draws}/{num_games} ({draws/num_games:.1%})")
        print(f"  Avg Game Length: {avg_length:.1f} steps")
        print(f"  Min/Max: {min(game_lengths)}/{max(game_lengths)} steps")

    return results


def benchmark_all_opponents(model, num_games=100):
    """
    Benchmark model against NormalBot opponent.

    Args:
        model: Trained PPO model
        num_games: Number of games to play

    Returns:
        Dict with results
    """
    print("\n" + "="*70)
    print("BENCHMARK TEST")
    print("="*70)
    print(f"Testing model against NormalBot")
    print(f"Games: {num_games}")
    print("="*70)

    env_config = load_env_config()

    # Test against normal difficulty
    results = evaluate_against_opponent(model, 'normal', num_games, verbose=True, env_config=env_config)

    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    win_rate = results['win_rate']
    avg_length = results['avg_game_length']

    if win_rate >= 0.90:
        rating = "⭐⭐⭐⭐⭐ Mastered"
    elif win_rate >= 0.70:
        rating = "⭐⭐⭐⭐ Strong"
    elif win_rate >= 0.50:
        rating = "⭐⭐⭐ Competent"
    elif win_rate >= 0.30:
        rating = "⭐⭐ Learning"
    else:
        rating = "⭐ Weak"

    print(f"Win Rate:     {win_rate:>6.1%}")
    print(f"Avg Length:   {avg_length:>6.1f} steps")
    print(f"Rating:       {rating}")
    print("="*70)

    if win_rate >= 0.80:
        print("\n🏆 EXCELLENT - AI is highly competent!")
    elif win_rate >= 0.60:
        print("\n✅ GOOD - AI has learned solid strategy")
    elif win_rate >= 0.40:
        print("\n⚠️  FAIR - AI needs more training")
    else:
        print("\n❌ POOR - AI needs significant more training")

    return results


def watch_game(model, opponent_difficulty='normal', verbose=True):
    """
    Watch a single game with detailed output.

    Args:
        model: Trained PPO model
        opponent_difficulty: Opponent to play against
        verbose: Print detailed game log
    """
    print("\n" + "="*70)
    print(f"WATCHING GAME VS {opponent_difficulty.upper()}")
    print("="*70)

    env = RealTimeRTSEnv(opponent_type=opponent_difficulty,
                         decision_interval=DEFAULT_DECISION_INTERVAL)
    obs, _ = env.reset(seed=0)
    sim = env.sim

    done = False
    steps = 0

    print(f"\n{'Step':<6} {'AI Action':<25} {'AI Pop':<8} {'Opp Pop':<8} {'AI Army':<8} {'Opp Army'}")
    print("-" * 90)

    while not done:
        action_masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, _reward, done, _truncated, info = env.step(int(action))

        ai_faction = sim.state.get_faction(0)
        opp_faction = sim.state.get_faction(1)

        if verbose and steps % 20 == 0:
            print(f"{steps:<6} {info['action']:<25} "
                  f"{ai_faction.population:<8} {opp_faction.population:<8} "
                  f"{ai_faction.military_strength:<8.1f} {opp_faction.military_strength:<8.1f}")

        steps += 1

    print("\n" + "="*70)
    print("GAME RESULT")
    print("="*70)

    winner = info.get('winner')
    ai_faction = sim.state.get_faction(0)
    opp_faction = sim.state.get_faction(1)

    if winner == 0:
        print("🏆 AI WON!")
    elif winner == 1:
        print("💀 AI LOST")
    else:
        print("🤝 DRAW")

    print(f"\nGame Length: {steps} steps ({sim.game_time/60:.1f} minutes)")
    print(f"\nFinal State:")
    print(f"  AI:  Pop={ai_faction.population}, Military={ai_faction.military_strength:.1f}, "
          f"Buildings={sum(ai_faction.buildings.values())}")
    print(f"  Opp: Pop={opp_faction.population}, Military={opp_faction.military_strength:.1f}, "
          f"Buildings={sum(opp_faction.buildings.values())}")


def compare_checkpoints(checkpoint_paths, opponent_difficulty='normal', num_games=50):
    """
    Compare multiple model checkpoints.

    Args:
        checkpoint_paths: List of paths to model checkpoints
        opponent_difficulty: Opponent to test against
        num_games: Games per checkpoint
    """
    print("\n" + "="*70)
    print("CHECKPOINT COMPARISON")
    print("="*70)
    print(f"Testing {len(checkpoint_paths)} checkpoints vs {opponent_difficulty}")
    print(f"Games per checkpoint: {num_games}")
    print("="*70)

    env_config = load_env_config()

    results = []

    for path in checkpoint_paths:
        print(f"\nLoading: {path}")
        model = MaskablePPO.load(path)

        checkpoint_name = os.path.basename(path)
        result = evaluate_against_opponent(model, opponent_difficulty, num_games, verbose=False, env_config=env_config)
        result['checkpoint'] = checkpoint_name
        results.append(result)

    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Checkpoint':<30} {'Win Rate':<12} {'Avg Length':<12}")
    print("-"*70)

    for result in sorted(results, key=lambda x: x['win_rate'], reverse=True):
        print(f"{result['checkpoint']:<30} {result['win_rate']:>6.1%}      {result['avg_game_length']:>6.1f}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Test trained RTS AI model against NormalBot')
    parser.add_argument('model_path', type=str, nargs='?', help='Path to trained model (.zip file)')
    parser.add_argument('--test', type=str, choices=['benchmark', 'watch', 'quick'],
                        default='benchmark', help='Test type to run')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of games for benchmark (default: 100)')
    parser.add_argument('--compare', nargs='+', type=str,
                        help='Compare multiple checkpoints (provide multiple paths)')

    args = parser.parse_args()

    if not args.compare:
        if not args.model_path:
            print("Error: model_path is required (or use --compare)")
            parser.print_help()
            return

        print(f"\n📂 Loading model from: {args.model_path}")
        try:
            model = MaskablePPO.load(args.model_path)
            print(f"✅ Model loaded successfully")
            print(f"   Device: {model.device}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return

    # Run test (always against normal difficulty)
    if args.compare:
        compare_checkpoints(args.compare, 'normal', num_games=50)
    elif args.test == 'benchmark':
        benchmark_all_opponents(model, num_games=args.games)
    elif args.test == 'watch':
        watch_game(model, 'normal', verbose=True)
    elif args.test == 'quick':
        evaluate_against_opponent(model, 'normal', num_games=10, verbose=True)


if __name__ == '__main__':
    main()
