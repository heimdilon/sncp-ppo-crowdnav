"""Analyze the preregistered v36-vs-v30 multi-seed decision rule.

Uses Wilson intervals plus two-proportion z-tests with Bonferroni alpha=0.0125.
The v30 baseline is embedded so this script only needs v36_multiseed_result.json.

Run from the repo root:
    python scratch/_analyze_v36.py
"""

import json
import math
from pathlib import Path

from scipy.stats import norm


Z_95 = 1.959963984540054
ALPHA = 0.0125
DENSITIES = (5, 10, 15, 20)
HIGH_DENSITIES = (15, 20)
LOW_DENSITIES = (5, 10)
BASELINE_N = 250
V30_SUCCESS = {5: 0.972, 10: 0.896, 15: 0.856, 20: 0.792}
V30_COLLISION = {5: 0.028, 10: 0.104, 15: 0.144, 20: 0.208}
V32_SUCCESS = {5: 0.976, 10: 0.940, 15: 0.876, 20: 0.788}
V32_COLLISION = {5: 0.028, 10: 0.060, 15: 0.132, 20: 0.212}


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    denominator = 1 + Z_95**2 / n
    center = (p + Z_95**2 / (2 * n)) / denominator
    half_width = (
        Z_95 * math.sqrt(p * (1 - p) / n + Z_95**2 / (4 * n**2)) / denominator
    )
    return center - half_width, center + half_width


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z_score = (p1 - p2) / se if se else 0.0
    return z_score, 2 * norm.sf(abs(z_score))


def count(rate: float, n: int) -> int:
    return round(rate * n)


def comparison(v36: dict, metric: str, baseline: dict[int, float]) -> dict[int, dict]:
    rows = {}
    for density in DENSITIES:
        result = v36[str(density)]
        n36 = int(result["n"])
        rate36 = float(result[metric])
        rate30 = baseline[density]
        z_score, p_value = two_proportion_z(
            count(rate36, n36), n36, count(rate30, BASELINE_N), BASELINE_N
        )
        rows[density] = {
            "v36": rate36,
            "v30": rate30,
            "delta": rate36 - rate30,
            "z": z_score,
            "p": p_value,
            "significant": p_value < ALPHA,
        }
    return rows


def print_comparison(title: str, rows: dict[int, dict]) -> None:
    print(f"\n{title} (two-proportion z; Bonferroni alpha={ALPHA})")
    print(" N | v30 % | v36 % | delta pp |      z |       p | significant")
    print("-" * 70)
    for density in DENSITIES:
        row = rows[density]
        print(
            f"{density:>2} | {row['v30'] * 100:>5.1f} | {row['v36'] * 100:>5.1f} | "
            f"{row['delta'] * 100:>+8.1f} | {row['z']:>+6.2f} | "
            f"{row['p']:>7.4f} | {str(row['significant']):>11}"
        )


def main() -> None:
    path = Path("v36_multiseed_result.json")
    v36 = json.loads(path.read_text(encoding="utf-8"))
    missing = [str(density) for density in DENSITIES if str(density) not in v36]
    if missing:
        raise SystemExit(f"Incomplete sweep; missing densities: {', '.join(missing)}")

    print("v36 honest sweep: 5 seeds x 50 episodes, paper_challenging")
    print(" N | success % (95% Wilson) | collision % | timeout % | v32 success/collision %")
    print("-" * 92)
    for density in DENSITIES:
        result = v36[str(density)]
        n = int(result["n"])
        success = float(result["pooled_success"])
        low, high = wilson(count(success, n), n)
        print(
            f"{density:>2} | {success * 100:>5.1f} [{low * 100:>5.1f}, {high * 100:>5.1f}] | "
            f"{result['pooled_collision'] * 100:>11.1f} | "
            f"{result['pooled_timeout'] * 100:>9.1f} | "
            f"{V32_SUCCESS[density] * 100:>5.1f}/{V32_COLLISION[density] * 100:>5.1f}"
        )

    success_rows = comparison(v36, "pooled_success", V30_SUCCESS)
    collision_rows = comparison(v36, "pooled_collision", V30_COLLISION)
    print_comparison("SUCCESS", success_rows)
    print_comparison("COLLISION", collision_rows)

    high_n_success_up = any(
        success_rows[n]["significant"] and success_rows[n]["delta"] > 0
        for n in HIGH_DENSITIES
    )
    high_n_collision_down = any(
        collision_rows[n]["significant"] and collision_rows[n]["delta"] < 0
        for n in HIGH_DENSITIES
    )
    low_n_regression = any(
        (success_rows[n]["significant"] and success_rows[n]["delta"] < 0)
        or (collision_rows[n]["significant"] and collision_rows[n]["delta"] > 0)
        for n in LOW_DENSITIES
    )
    timeout_zero = all(float(v36[str(n)]["pooled_timeout"]) == 0.0 for n in DENSITIES)
    positive = (
        (high_n_success_up or high_n_collision_down)
        and not low_n_regression
        and timeout_zero
    )

    print("\nPREREGISTERED DECISION")
    print(f"  high-N success increased significantly: {high_n_success_up}")
    print(f"  high-N collision decreased significantly: {high_n_collision_down}")
    print(f"  no significant N=5/10 regression: {not low_n_regression}")
    print(f"  timeout is zero at every density: {timeout_zero}")
    print(f"  VERDICT: {'POSITIVE' if positive else 'NEGATIVE/FLAT'}")


if __name__ == "__main__":
    main()
