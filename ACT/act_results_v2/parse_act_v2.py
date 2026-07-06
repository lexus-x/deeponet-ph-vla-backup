"""Parse the ACT V2 (15K finetune) eval log into a clean per-suite / per-variant
summary: in-dist avg (mean over 10 tasks x 3 seeds) + Plus robustness avg."""
import re, csv, sys

LOG = "/media/user/C2FE578FFE577A9D/ACT_v2/run_ft15k.out"
VARIANTS = ["act", "act_deeponet", "act_deeponet_ph"]
lines = open(LOG, encoding="utf-8", errors="ignore").read().splitlines()

# in-dist: track current suite via "EVAL indist start <Suite>"; sum task% per (suite,variant)
indist = {}   # (suite,variant) -> [task pcts]
plus = {}     # (suite,variant) -> pct
cur_suite = None
plus_suite = None
task_re = re.compile(r"\[LIBERO-[A-Z0-9]+\]\s+(\w+)\s+task\d+:\s+([\d.]+)%")
for ln in lines:
    m = re.search(r"EVAL indist start (\w+)", ln)
    if m: cur_suite = m.group(1); continue
    m = task_re.search(ln)
    if m and cur_suite:
        var, pct = m.group(1), float(m.group(2))
        if var in VARIANTS:
            indist.setdefault((cur_suite, var), []).append(pct)
        continue
    m = re.search(r"\[plus\]\s+(\w+)\s+robustness avg =\s+([\d.]+)%", ln)
    if m and m.group(1) in VARIANTS:
        # plus blocks come in variant order before "RESULT READY: <suite>"
        # find the next RESULT READY suite by scanning forward is complex; instead
        # attribute via order: track a pending suite set at RESULT READY below.
        plus.setdefault(m.group(1), []).append(float(m.group(2)))
        continue
    m = re.search(r"RESULT READY:\s+(\w+)", ln)
    if m:
        s = m.group(1)
        for v in VARIANTS:
            if plus.get(v):
                plus[(s, v)] = plus[v].pop(0)

SUITES = ["Spatial", "Object", "Long", "Goal"]
rows = []
print(f"{'suite':8s} {'variant':16s} {'indist%':>8s} {'plus%':>7s}")
for s in SUITES:
    for v in VARIANTS:
        ind = indist.get((s, v), [])
        ia = round(sum(ind) / len(ind), 1) if ind else None
        pa = plus.get((s, v))
        rows.append([s, v, ia, pa])
        print(f"{s:8s} {v:16s} {str(ia):>8s} {str(pa):>7s}")

with open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/act_v2_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["suite", "variant", "indist_avg", "plus_avg"]); w.writerows(rows)
print("WROTE", sys.argv[1] if len(sys.argv) > 1 else "/tmp/act_v2_summary.csv")
