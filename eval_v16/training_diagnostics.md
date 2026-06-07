# SNCP-PPO Training Diagnostics

CSV: `logs\training_20260607_131329.csv`
Rows: 1221
Evaluated holdout rows: 1202
Observed replay ratio: 17.8%

## Best Holdout

Best step: 1515520
Best min success: 56.0%
Best reason: best updated (priority: min_success, tie-break: avg_reward, then lower collision_rate)

## Final Holdout

Final step: 2500608
Final min success: 46.0%
Collapse delta: -10.0%
Collapse detected: no

## PPO Stability

Final std linear: 0.153
Final std angular: 0.243
Max std linear: 0.153
Max std angular: 0.243
Std linear delta: 0.017
Std angular delta: 0.020

## Per-Scenario Success

| Scenario | Best | Final | Delta |
|---|---:|---:|---:|
| easy | 70.0% | 76.0% | +6.0% |
| hard | 62.0% | 78.0% | +16.0% |
| circle | 56.0% | 46.0% | -10.0% |
