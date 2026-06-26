"""v38 (locked v34 Beta + training-free action shield) vs v34 (no shield) and vs v30.

The shield effect is ISOLATED by v38-vs-v34: identical checkpoint, seeds, and episodes,
only the runtime shield differs -> any gap is attributable to the shield alone. Same honest
protocol/accounting as the rest (5 seeds x 50 ep = 250/density). Wilson 95% CI, pooled
two-proportion z, Cohen h, Bonferroni alpha = 0.05/4 = 0.0125 (success & collision are
separate pre-registered families).

Run:  C:/ProgramData/miniconda3/python.exe scratch/_analyze_v38.py
"""
import json
import math

from scipy.stats import norm

Z = 1.959963984540054
DENS = [5, 10, 15, 20]
ALPHA = 0.0125


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


def head(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


def compare(name, cur, base):
    head(f"v38 vs {name} -- SUCCESS (two-prop z, Cohen h, diff 95% CI); Bonferroni sig iff p<0.0125")
    print(f"{'N':>3} | {name+' %':>7} | {'v38 %':>6} | {'dPP':>6} | {'z':>6} | {'p':>8} | {'sig':>4} | {'h':>6} | diff95%CI")
    print("-" * 100)
    for N in DENS:
        s = str(N); n = cur[s]["n"]
        kc = round(cur[s]["pooled_success"] * n); kb = round(base[s]["pooled_success"] * n)
        z, p = two_prop_z(kc, n, kb, n); h = cohens_h(kc / n, kb / n); d, lo, hi = diff_ci(kc, n, kb, n)
        print(f"{N:>3} | {base[s]['pooled_success']*100:>6.1f} | {cur[s]['pooled_success']*100:>5.1f} | "
              f"{d*100:>+5.1f} | {z:>+5.2f} | {p:>8.4f} | {'YES' if p<ALPHA else 'no':>4} | {h:>+5.2f} | [{lo*100:>+5.1f},{hi*100:>+5.1f}]")
    head(f"v38 vs {name} -- COLLISION (v38 LOWER = good); sig iff p<0.0125")
    print(f"{'N':>3} | {name+' c%':>7} | {'v38 c%':>6} | {'dPP':>6} | {'z':>6} | {'p':>8} | {'sig':>4} | drop?")
    print("-" * 100)
    for N in DENS:
        s = str(N); n = cur[s]["n"]
        cc = round(cur[s]["pooled_collision"] * n); cb = round(base[s]["pooled_collision"] * n)
        z, p = two_prop_z(cc, n, cb, n); d = cur[s]["pooled_collision"] - base[s]["pooled_collision"]
        drop = "DROP" if (d < 0 and p < ALPHA) else ("up" if d > 0 else "ns")
        print(f"{N:>3} | {base[s]['pooled_collision']*100:>6.1f} | {cur[s]['pooled_collision']*100:>5.1f} | "
              f"{d*100:>+5.1f} | {z:>+5.2f} | {p:>8.4f} | {'YES' if p<ALPHA else 'no':>4} | {drop}")


def main():
    v30 = load("v30_multiseed_result.json")
    v34 = load("v34_multiseed_result.json")
    v38 = load("v38_multiseed_result.json")

    head("v38 = locked v34 Beta + TRAINING-FREE ACTION SHIELD -- honest 5-seed pool (250/density)")
    print(f"{'N':>3} | {'v38 succ':>8} | {'Wilson 95% CI':>16} | {'coll%':>6} | {'to%':>5} | {'navstep':>7}")
    print("-" * 72)
    for N in DENS:
        s = str(N); n = v38[s]["n"]; k = round(v38[s]["pooled_success"] * n); lo, hi = wilson(k, n)
        print(f"{N:>3} | {v38[s]['pooled_success']*100:>7.1f} | [{lo*100:>5.1f}, {hi*100:>5.1f}] | "
              f"{v38[s]['pooled_collision']*100:>5.1f} | {v38[s]['pooled_timeout']*100:>4.1f} | "
              f"{v38[s].get('avg_success_steps', float('nan')):>7.1f}")

    compare("v34", v38, v34)   # isolates the shield (same policy)
    compare("v30", v38, v30)   # vs prior champion

    head("DECISION (v38 shield vs v34 raw): high-N (15/20) collision DROP &/or success UP (Bonferroni),"
         " no low-N (5/10) regression, timeout not worse")
    highN_coll_drop = highN_succ_up = False
    lowN_ok = True
    for N in DENS:
        s = str(N); n = v38[s]["n"]
        cc = round(v38[s]["pooled_collision"] * n); cb = round(v34[s]["pooled_collision"] * n)
        _, pc = two_prop_z(cc, n, cb, n)
        ks = round(v38[s]["pooled_success"] * n); kb = round(v34[s]["pooled_success"] * n)
        _, ps = two_prop_z(ks, n, kb, n)
        dcoll = v38[s]["pooled_collision"] - v34[s]["pooled_collision"]
        dsucc = v38[s]["pooled_success"] - v34[s]["pooled_success"]
        if N in (15, 20):
            if dcoll < 0 and pc < ALPHA:
                highN_coll_drop = True
            if dsucc > 0 and ps < ALPHA:
                highN_succ_up = True
        if N in (5, 10):
            if (dsucc < -1e-9 and ps < ALPHA) or (dcoll > 1e-9 and pc < ALPHA):
                lowN_ok = False
    helps = (highN_coll_drop or highN_succ_up) and lowN_ok
    print(f"  high-N collision DROP (Bonferroni): {highN_coll_drop}")
    print(f"  high-N success UP   (Bonferroni): {highN_succ_up}")
    print(f"  no low-N regression: {lowN_ok}")
    print(f"  timeout v34->v38: " + ", ".join(f"N={N}:{v34[str(N)]['pooled_timeout']*100:.1f}->{v38[str(N)]['pooled_timeout']*100:.1f}" for N in DENS))
    print("\n  >>> VERDICT:",
          "POSITIVE -- shield delivers a Bonferroni-significant high-N improvement over raw v34."
          if helps else "NOT Bonferroni-clearing on the pre-registered rule; see deltas above.")


if __name__ == "__main__":
    main()
