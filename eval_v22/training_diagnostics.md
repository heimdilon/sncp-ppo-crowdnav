# SNCP-PPO Training Diagnostics

CSV: `logs/training_20260612_073945.csv`
Rows: 1221
Evaluated holdout rows: 1202
Observed replay ratio: 17.8%

## Best Holdout

Best step: 696320
Best min success: 52.0%
Best reason: best updated (priority: min_success, tie-break: avg_reward, then lower collision_rate)

## Final Holdout

Final step: 2500608
Final min success: 50.0%
Collapse delta: -2.0%
Collapse detected: no

## PPO Stability

Final std linear: 0.164
Final std angular: 0.239
Max std linear: 0.164
Max std angular: 0.239
Std linear delta: 0.029
Std angular delta: 0.015

## Per-Scenario Success

| Scenario | Best | Final | Delta |
|---|---:|---:|---:|
| easy | 92.0% | 86.0% | -6.0% |
| hard | 64.0% | 58.0% | -6.0% |
| circle | 52.0% | 50.0% | -2.0% |

## Per-Scenario Failure Profile

| Scenario | Final success | Final collision | Final timeout | Final avg steps | Final avg I_sp | Final min d_min |
|---|---:|---:|---:|---:|---:|---:|
| easy | 86.0% | 4.0% | 10.0% | 49.3 | 0.002 | 0.593 |
| hard | 58.0% | 22.0% | 20.0% | 48.1 | 0.008 | 0.276 |
| circle | 50.0% | 46.0% | 4.0% | 45.2 | 0.009 | 0.380 |
