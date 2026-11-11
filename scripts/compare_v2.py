"""v2 comparison on the same 12 GSM8K test problems (6,000 cap).
completed_correct = correct AND the model closed its thinking and answered (finish == stop).
A boxed number found only inside unfinished thinking at the cap is not counted as completed.
usage: python scripts/compare_v2.py label=results/x_gsm8k.jsonl ..."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
def tail_dup(text, n=8, tail=400):
    w=text.split()
    if len(w)<n+tail//4: return 0.0
    seen=set(); flags=[]
    for i in range(len(w)-n+1):
        g=tuple(w[i:i+n]); flags.append(g in seen); seen.add(g)
    t=flags[-tail:]; return sum(t)/len(t)
print("| model | n | correct (any boxed) | completed correct | hit cap | verbatim loops | mean tokens | mean tokens, completed correct | completed-correct idx |")
print("|---|---|---|---|---|---|---|---|---|")
for arg in sys.argv[1:]:
    lab,f=arg.split("=",1); rs=[json.loads(l) for l in open(f)]; n=len(rs)
    cc=[r for r in rs if r["correct"] and r["finish"]=="stop"]
    cap=sum(r["finish"]=="length" for r in rs)
    loops=sum(r["finish"]=="length" and tail_dup((r.get("reasoning") or "")+(r.get("answer_text") or ""))>=0.5 for r in rs)
    mt=round(sum(r["tokens"] for r in rs)/n); mcc=round(sum(r["tokens"] for r in cc)/len(cc)) if cc else "-"
    print(f"| {lab} | {n} | {sum(r['correct'] for r in rs)} | {len(cc)} | {cap}/{n} | {loops}/{n} | {mt} | {mcc} | {', '.join(str(r['idx']) for r in sorted(cc,key=lambda r:r['idx']))} |")
