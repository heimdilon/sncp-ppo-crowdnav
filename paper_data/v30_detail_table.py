"""Compute v30 champion detailed-metric LaTeX rows from the honest sweep JSON.
success% [Wilson 95% CI] / collision% / timeout% / nav-time(s)=steps*0.25 / I_sp(x1e-3).
Run from repo root: C:/ProgramData/miniconda3/python.exe scratch/_v30_detail_table.py
"""
import json
import math

Z = 1.959963984540054


def wilson(p, n):
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100


v30 = json.load(open("v30_multiseed_result.json"))
print("N & Başarı [\\%95 GA] & Çarpışma & Zaman aşımı & Nav-süre (s) & $\\isp$ ($\\times10^{-3}$) \\\\")
for N in (5, 10, 15, 20):
    s = v30[str(N)]
    p, n = s["pooled_success"], s["n"]
    lo, hi = wilson(p, n)
    navt = s["avg_success_steps"] * 0.25
    print(f"{N} & {p*100:.1f} [{lo:.1f}, {hi:.1f}] & {s['pooled_collision']*100:.1f} & "
          f"{s['pooled_timeout']*100:.1f} & {navt:.2f} & {s['avg_i_sp']*1000:.1f} \\\\")
