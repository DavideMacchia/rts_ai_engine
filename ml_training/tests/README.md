# ML Training Tests

Organized test suite for the RTS ML training system.

## Test Organization

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for combined systems
├── system/           # Full system tests (population, production, etc.)
├── model/            # Trained model testing and benchmarking
└── debug/            # Debug-specific tests
```

---

## 📁 Folder Structure

### `unit/` - Unit Tests
Tests for individual components and features.

**Files:**
- `test_achievements.py` - Achievement system tests
- `test_scripted_opponents.py` - Scripted opponent behavior tests
- `test_quick_diagnostic.py` - Quick diagnostic checks

**Run:**
```bash
python tests/unit/test_achievements.py
python tests/unit/test_scripted_opponents.py
```

---

### `integration/` - Integration Tests
Tests for combined systems and Phase implementations.

**Files:**
- `test_ai_improvements.py` - AI improvement features
- `test_phase1_complete.py` - Phase 1 verification (if exists)

**Run:**
```bash
python tests/integration/test_ai_improvements.py
```

---

### `system/` - System Tests
Tests for complete game systems.

**Files:**
- `test_population.py` - Population mechanics
- `test_production_system.py` - Resource production
- `test_worker_distribution.py` - Worker allocation
- `test_military_attacks.py` - Combat system

**Run:**
```bash
python tests/system/test_population.py
python tests/system/test_production_system.py
```

---

### `model/` - Model Tests
**⭐ Most Important for evaluating trained AI**

**Files:**
- `test_trained_model.py` - Comprehensive model testing and benchmarking

**Run:**
```bash
# Full benchmark (all opponents)
python tests/model/test_trained_model.py curriculum_logs/model_final --test benchmark

# Watch a game
python tests/model/test_trained_model.py curriculum_logs/model_final --test watch --opponent medium

# Quick test (10 games)
python tests/model/test_trained_model.py curriculum_logs/model_final --test quick --opponent hard

# Compare checkpoints
python tests/model/test_trained_model.py \
  --compare curriculum_logs/model_easy curriculum_logs/model_final \
  --opponent medium
```

---

### `debug/` - Debug Tests
Tests for debugging specific issues.

**Files:**
- `test_attack_debug.py`
- `test_population_debug.py`
- `test_starvation_debug.py`

**Run:** Use these when debugging specific issues.

---

## Quick Test Commands

### Run All Unit Tests
```bash
cd src/ml_training
python -m pytest tests/unit/ -v
# OR run individually:
python tests/unit/test_achievements.py
python tests/unit/test_scripted_opponents.py
```

### Run Integration Tests
```bash
python tests/integration/test_ai_improvements.py
```

### Run System Tests
```bash
python tests/system/test_population.py
python tests/system/test_production_system.py
```

### Test Trained Model (After Training)
```bash
# Benchmark against all opponents
python tests/model/test_trained_model.py \
  curriculum_logs/model_final \
  --test benchmark \
  --games 100
```

---

## Test Categories by Purpose

### Before Training
Run these to verify system is ready:
- ✅ `tests/unit/test_achievements.py` - Achievement system works
- ✅ `tests/unit/test_scripted_opponents.py` - Opponents work
- ✅ `tests/integration/test_ai_improvements.py` - Phase 1 features work

### During Training
Monitor training progress:
- Check tensorboard logs
- Observe win rates in curriculum stages

### After Training
Evaluate model performance:
- ⭐ `tests/model/test_trained_model.py` - **Primary evaluation tool**
  - Benchmark vs all opponents
  - Watch games
  - Compare checkpoints

---

## Expected Test Results

### Unit Tests
All should pass ✅
- Achievements: 62 loaded, tests passing
- Scripted opponents: All 5 difficulty levels working
- Quick diagnostic: System functional

### Model Tests (After Training)
Expected win rates:
- Tutorial: 98-100% ⭐⭐⭐⭐⭐
- Easy: 90-95% ⭐⭐⭐⭐⭐
- Medium: 70-80% ⭐⭐⭐⭐
- Hard: 55-65% ⭐⭐⭐
- Very Hard: 50-60% ⭐⭐⭐

Overall: 70-80% average = Good AI ✅

---

## Adding New Tests

### Unit Test Template
```python
"""Test description"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.your_module import YourClass

def test_your_feature():
    # Setup
    obj = YourClass()

    # Test
    result = obj.your_method()

    # Assert
    assert result == expected
    print("✓ Test passed")

if __name__ == '__main__':
    test_your_feature()
```

### Model Test Template
```python
"""Model test description"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.scripted_opponents import get_opponent

# Your test code here
```

---

## Troubleshooting

### Import Errors
All tests are organized in subfolders and need to go up 3 levels to reach ml_training:
```python
# For tests in unit/, integration/, system/, model/, or debug/ folders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
```
This allows importing from `simulator/`, `training/`, and other ml_training modules.

### Model Test Failing
1. Check model path is correct
2. Ensure model was trained with Phase 1 features
3. Verify enhanced state is being used (34 features)
4. Check action masking is working

### Achievement Tests Failing
1. Check achievements.json exists at: `src/assets/constants/achievements.json`
2. Verify path calculation in `achievements_rewards.py`

---

## CI/CD Integration

### Pre-commit Tests
```bash
# Run from src/ml_training directory
cd src/ml_training
python tests/unit/test_achievements.py
python tests/unit/test_scripted_opponents.py
```

### Pre-training Tests
```bash
# Verify system ready for training
python tests/integration/test_ai_improvements.py
```

### Post-training Tests
```bash
# Benchmark trained model
python tests/model/test_trained_model.py curriculum_logs/model_final --test benchmark --games 50
```

---

## Summary

| Folder | Purpose | When to Run | Key Files |
|--------|---------|-------------|-----------|
| `unit/` | Component tests | Before training | test_achievements.py |
| `integration/` | System integration | Before training | test_ai_improvements.py |
| `system/` | Game mechanics | When debugging | test_population.py |
| `model/` | **AI evaluation** | **After training** | **test_trained_model.py** |
| `debug/` | Issue investigation | When debugging | test_*_debug.py |

**Most important:** `tests/model/test_trained_model.py` - Use this to evaluate your AI! 🚀
