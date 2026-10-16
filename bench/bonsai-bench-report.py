#!/usr/bin/env python3
"""Summarize bonsai-benchmark.sh runs.csv: per configuration and test mean, stddev, 95% CI, stability flags.

usage: bonsai-bench-report.py runs.csv REFERENCE_CONFIG NCPU
A run is flagged 'unstable' if, compared with the coolest/calmest state seen in the session, the machine was
>= 8 C hotter, its CPU frequency cap was >= 5% lower, the 1-min load average exceeded 25% of the CPUs above
the minimum, or the power source differed from the first run.
"""
import csv
import math
import statistics
import sys
from collections import defaultdict

T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
       12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04}


def t95(df):
    if df <= 0:
        return float("nan")
    keys = sorted(T95)
    for k in keys:
        if df <= k:
            return T95[k]
    return 1.96


def stats(xs):
    n = len(xs)
    if n == 0:
        return None
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    ci = t95(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return n, m, sd, ci


def main():
    path, ref, ncpu = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("\n(no successful runs)\n")
        return
    f = lambda r, k: float(r[k]) if r[k] not in ("", None) else 0.0
    min_temp = min(f(r, "temp_c") for r in rows)
    max_cap = max(f(r, "freq_cap_mhz") for r in rows)
    min_load = min(f(r, "load1") for r in rows)
    power0 = rows[0]["power"].split("(")[0]
    unstable = 0
    for r in rows:
        why = []
        if f(r, "temp_c") >= min_temp + 8:
            why.append("hot")
        if max_cap and f(r, "freq_cap_mhz") < 0.95 * max_cap:
            why.append("throttled")
        if f(r, "load1") > min_load + 0.25 * ncpu:
            why.append("busy")
        if r["power"].split("(")[0] != power0:
            why.append("power-changed")
        r["flags"] = ",".join(why)
        unstable += bool(why)

    data = defaultdict(lambda: defaultdict(lambda: {"all": [], "stable": []}))
    for r in rows:
        d = data[r["config"]][r["test"]]
        d["all"].append(float(r["tps"]))
        if not r["flags"]:
            d["stable"].append(float(r["tps"]))

    print("## Results (tokens/s, mean +- 95% CI over rounds)\n")
    print(f"{len(rows)} measurements, {unstable} flagged unstable (hot / throttled / busy / power source changed).")
    print(f"Session minimum temperature {min_temp:.0f} C, max frequency cap {max_cap:.0f} MHz, minimum load {min_load:.2f}.\n")
    for test in ("tg", "pp"):
        refm = None
        if ref in data and data[ref][test]["all"]:
            refm = statistics.fmean(data[ref][test]["all"])
        print(f"### {'generation (tg)' if test == 'tg' else 'prompt processing (pp)'}\n")
        print("| configuration | n | mean | sd | 95% CI | vs " + ref + " | stable-only mean (n) |")
        print("|---|---|---|---|---|---|---|")
        order = sorted(data, key=lambda c: -statistics.fmean(data[c][test]["all"]) if data[c][test]["all"] else 0)
        for cfg in order:
            s = stats(data[cfg][test]["all"])
            if not s:
                continue
            n, m, sd, ci = s
            st = stats(data[cfg][test]["stable"])
            rel = f"{m / refm:.2f}x" if refm else "-"
            stab = f"{st[1]:.2f} ({st[0]})" if st else "-"
            print(f"| {cfg} | {n} | {m:.2f} | {sd:.2f} | +-{ci:.2f} | {rel} | {stab} |")
        print()
    print("Differences smaller than the confidence intervals are not significant. Only compare configurations\n"
          "measured in the same session; the per-run conditions are in runs.csv.\n")


if __name__ == "__main__":
    main()
