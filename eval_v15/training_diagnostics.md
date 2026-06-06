# SNCP-PPO Training Diagnostics

CSV: `logs\training_20260605_140811.csv`
Rows: 1059
Evaluated holdout rows: 1040
Observed replay ratio: not logged

## Best Holdout

Best step: 942080
Best min success: 36.0%
Best reason: best updated (priority: min_success, tie-break: avg_reward, then lower collision_rate)

## Final Holdout

Final step: 2168832
Final min success: 0.0%
Collapse delta: -36.0%
Collapse detected: yes

## PPO Stability

Final std linear: not logged
Final std angular: not logged
Max std linear: not logged
Max std angular: not logged
Std linear delta: not logged
Std angular delta: not logged

## Per-Scenario Success

| Scenario | Best | Final | Delta |
|---|---:|---:|---:|
| easy | 66.0% | 0.0% | -66.0% |
| hard | 60.0% | 0.0% | -60.0% |
| circle | 36.0% | 0.0% | -36.0% |
