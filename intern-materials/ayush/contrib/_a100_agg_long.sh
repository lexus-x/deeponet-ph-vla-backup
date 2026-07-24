#!/bin/bash
set -u
OUT=$HOME/deeponet_campaign/long_confirm_results
python3 - <<'PY'
import json, os
out = os.path.expanduser("~/deeponet_campaign/long_confirm_results")
summary = {}
for name, mid in [("m3_long","m3"),("flow_long","flow")]:
    p = os.path.join(out, name, "success_rates.json")
    d = json.load(open(p))
    pt = d["LIBERO-10"][mid]["per_task"]
    rates = {k: float(v["success_rate"])*100 for k,v in pt.items()}
    summary[name] = {"mean_sr": sum(rates.values())/len(rates), "n": len(rates), "per_task": rates}
json.dump(summary, open(os.path.join(out,"SUMMARY_long_confirm.json"),"w"), indent=2)
print(json.dumps(summary, indent=2))
PY
