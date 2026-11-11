"""Build the before/after table for the README from results/*_gsm8k.jsonl (same 12 GSM8K test problems, 6000-token cap)."""
import json, os, subprocess, sys
rows=[]
for label,name in [("base","Qwen3.5-0.8B (unedited)"),("mine","this repo's edit (v1)"),("astra","Astra's published edit")]:
    p=f"results/{label}_gsm8k.jsonl"
    if not os.path.exists(p): continue
    s=json.loads(subprocess.check_output([sys.executable,"scripts/analyze.py",p]))
    rows.append((name,s))
print("| model | n | GSM8K acc | hit 6000 cap | verbatim loop rate | kept going after answer | mean tokens |")
print("|---|---|---|---|---|---|---|")
for name,s in rows:
    print(f"| {name} | {s['n']} | {s['accuracy']*100:.0f}% | {s['hit_cap']*100:.0f}% | {s['loop_rate']*100:.0f}% | {s['kept_going']*100:.0f}% | {s['mean_tokens']} |")
