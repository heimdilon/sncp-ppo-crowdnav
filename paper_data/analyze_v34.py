"""v34 (Beta action distribution) vs v30 champion + cumulative v28->...->v34.

v34's pre-registered decision rule (spec): Beta helps iff high-N SUCCESS rises and/or
COLLISION drops (esp N=15/20), with no regression at N=5/10 and timeout 0, vs the v30 baseline
(the champion). Reports BOTH success and collision z-tests vs v30. Caveats carried into the
verdict: Beta touches the PPO loop (gaussian path preserved); the entropy scale implicitly
changes under Beta (fixed c2); single training seed; 2.5M matched to v30.

Run:  C:/ProgramData/miniconda3/python.exe scratch/_analyze_v34.py
"""
import json
import math

from scipy.stats import norm

Z = 1.959963984540054
DENS = [5, 10, 15, 20]
ALPHA = 0.0125  # Bonferroni, 4 densities


def load(p):
    return json.loads(open(p).read())


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    return z, 2 * norm.sf(abs(z))


def cohens_h(p1, p2):
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def diff_ci(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return d, d - Z * se, d + Z * se


def main():
    v28 = load("v28_multiseed_result.json")
    v29 = load("v29_multiseed_result.json")
    v30 = load("v30_multiseed_result.json")
    v31 = load("v31_multiseed_result.json")
    v32 = load("v32_multiseed_result.json")
    v33 = load("v33_multiseed_result.json")
    v34 = load("v34_multiseed_result.json")

    print("=" * 100)
    print("CUMULATIVE honest sweep (5 seeds x 50 ep = 250/density, paper_challenging, robot 1.0)")
    print("=" * 100)
    print(f"{'N':>3} | {'v30 %':>6} | {'v33 %':>6} | {'v34 %':>6} | {'v34 95% Wilson CI':>20} | {'v34 coll%':>9} | {'v34 to%':>7}")
    print("-" * 100)
    for N in DENS:
        s = str(N)
        n = v34[s]["n"]
        k34 = round(v34[s]["pooled_success"] * n)
        lo, hi = wilson(k34, n)
        print(f"{N:>3} | {v30[s]['pooled_success']*100:>5.1f} | {v33[s]['pooled_success']*100:>5.1f} | "
              f"{v34[s]['pooled_success']*100:>5.1f} | "
              f"[{lo*100:>6.1f}, {hi*100:>6.1f}]      | {v34[s]['pooled_collision']*100:>8.1f} | {v34[s]['pooled_timeout']*100:>6.1f}")

    print()
    print("=" * 100)
    print("v34 vs v30 SUCCESS (two-proportion z, Cohen's h, diff 95% CI); Bonferroni sig iff p<0.0125")
    print("=" * 100)
    print(f"{'N':>3} | {'dPP':>6} | {'z':>6} | {'p':>8} | {'sig?':>5} | {'Cohen h':>7} | {'diff 95% CI (pp)':>20} | clears")
    print("-" * 100)
    succ = {}
    for N in DENS:
        s = str(N)
        n30, n34 = v30[s]["n"], v34[s]["n"]
        k30 = round(v30[s]["pooled_success"] * n30)
        k34 = round(v34[s]["pooled_success"] * n34)
        z, p = two_prop_z(k34, n34, k30, n30)
        h = cohens_h(k34 / n34, k30 / n30)
        d, dlo, dhi = diff_ci(k34, n34, k30, n30)
        v34lo, v34hi = wilson(k34, n34)
        v30lo, v30hi = wilson(k30, n30)
        clears_up = v34lo > v30hi
        regress_down = v34hi < v30lo
        tag = "UP" if clears_up else ("DOWN" if regress_down else "overlap")
        sig = "YES" if p < ALPHA else "no"
        print(f"{N:>3} | {d*100:>+5.1f} | {z:>+5.2f} | {p:>8.4f} | {sig:>5} | {h:>+6.3f} | [{dlo*100:>+5.1f}, {dhi*100:>+5.1f}]    | {tag}")
        succ[N] = dict(d=d, p=p, sig=p < ALPHA, clears_up=clears_up, regress_down=regress_down,
                       timeout=v34[s]["pooled_timeout"])

    print()
    print("=" * 100)
    print("v34 vs v30 COLLISION (v34 LOWER = good). z on collision rate; sig iff p<0.0125")
    print("=" * 100)
    print(f"{'N':>3} | {'v30 coll%':>9} | {'v34 coll%':>9} | {'dPP':>6} | {'z':>6} | {'p':>8} | {'sig?':>5} | drop?")
    print("-" * 100)
    coll = {}
    for N in DENS:
        s = str(N)
        n30, n34 = v30[s]["n"], v34[s]["n"]
        c30 = round(v30[s]["pooled_collision"] * n30)
        c34 = round(v34[s]["pooled_collision"] * n34)
        z, p = two_prop_z(c34, n34, c30, n30)
        d = v34[s]["pooled_collision"] - v30[s]["pooled_collision"]
        sig = "YES" if p < ALPHA else "no"
        drop = "DROP" if (d < 0 and p < ALPHA) else ("up" if d > 0 else "ns")
        print(f"{N:>3} | {v30[s]['pooled_collision']*100:>8.1f} | {v34[s]['pooled_collision']*100:>8.1f} | "
              f"{d*100:>+5.1f} | {z:>+5.2f} | {p:>8.4f} | {sig:>5} | {drop}")
        coll[N] = dict(d=d, p=p, sig=p < ALPHA, drop=(d < 0 and p < ALPHA), rise=(d > 0 and p < ALPHA))

    print()
    print("=" * 100)
    print("DECISION RULE (v34 spec): Beta action dist helps iff high-N SUCCESS rises and/or")
    print("COLLISION drops (esp N=15/20), no regression at N=5/10, timeout stays 0. (vs v30).")
    print("=" * 100)
    highN_coll_drop = any(coll[N]["drop"] for N in (15, 20))
    highN_succ_up = any(succ[N]["sig"] and succ[N]["d"] > 0 for N in (15, 20)) or any(succ[N]["clears_up"] for N in (15, 20))
    lowN_ok = not (succ[5]["regress_down"] or succ[10]["regress_down"]) and not (succ[5]["sig"] and succ[5]["d"] < 0) and not (succ[10]["sig"] and succ[10]["d"] < 0)
    highN_coll_rise = any(coll[N]["rise"] for N in (15, 20))
    to_ok = all(succ[N]["timeout"] == 0 for N in DENS)
    print(f"  high-N collision DROP (N=15 or 20, significant): {highN_coll_drop}")
    print(f"  high-N success UP (N=15 or 20, significant/CI-clears): {highN_succ_up}")
    print(f"  high-N collision ROSE significantly (regression flag): {highN_coll_rise}")
    print(f"  no success regression at N=5/10: {lowN_ok}")
    print(f"  timeout stays 0 everywhere: {to_ok}")
    helps = (highN_coll_drop or highN_succ_up) and lowN_ok and to_ok and not highN_coll_rise
    print()
    if helps:
        print("  >>> VERDICT: POSITIVE - Beta action dist improves high-N (success up / collision down). v34 = new best.")
    else:
        print("  >>> VERDICT: NEGATIVE/FLAT - did not clear the bar vs v30. v30 stays best.")
    print()
    dom = all(v34[str(N)]["pooled_success"] >= v30[str(N)]["pooled_success"] - 1e-9 and
              v34[str(N)]["pooled_collision"] <= v30[str(N)]["pooled_collision"] + 1e-9 for N in DENS)
    print(f"  Pareto check: v34 >= v30 on BOTH success and collision at every density: {dom}")
    print()
    print("  Cumulative success | collision (v28->v29->v30->v31->v32->v33->v34):")
    for N in DENS:
        s = str(N)
        print(f"    N={N:>2}: succ {v28[s]['pooled_success']*100:.1f}->{v29[s]['pooled_success']*100:.1f}->{v30[s]['pooled_success']*100:.1f}->{v31[s]['pooled_success']*100:.1f}->{v32[s]['pooled_success']*100:.1f}->{v33[s]['pooled_success']*100:.1f}->{v34[s]['pooled_success']*100:.1f}"
              f"  | coll {v28[s]['pooled_collision']*100:.1f}->{v29[s]['pooled_collision']*100:.1f}->{v30[s]['pooled_collision']*100:.1f}->{v31[s]['pooled_collision']*100:.1f}->{v32[s]['pooled_collision']*100:.1f}->{v33[s]['pooled_collision']*100:.1f}->{v34[s]['pooled_collision']*100:.1f}")


if __name__ == "__main__":
    main()
