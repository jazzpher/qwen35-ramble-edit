"""Summarise eval files.
hit_cap      = answer hit max_tokens (never closed its thinking / never answered)
kept_going   = hit the cap even though the correct number already appeared in its reasoning
               (it had the answer and kept talking: the behaviour we want to remove)
loop_rate    = hit the cap AND >=50% of the word 8-grams in the last 400 words appeared earlier (verbatim loop)"""
import json, re, sys
def tail_dup(text, n=8, tail=400):
    w=text.split()
    if len(w)<n+tail//4: return 0.0
    seen=set(); flags=[]
    for i in range(len(w)-n+1):
        g=tuple(w[i:i+n]); flags.append(g in seen); seen.add(g)
    t=flags[-tail:]; return sum(t)/len(t)
def has_gold(text, gold):
    g=float(gold); nums=re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return any(abs(float(x.replace(",",""))-g)<1e-6 for x in nums if x.replace(",","").replace("-","").replace(".","",1).isdigit())
for f in sys.argv[1:]:
    rs=[json.loads(l) for l in open(f)]; n=len(rs)
    if not n: continue
    txt=lambda r:(r.get("reasoning") or "")+(r.get("answer_text") or "")
    cap=[r["finish"]=="length" for r in rs]
    print(json.dumps({"file":f,"n":n,"accuracy":round(sum(r["correct"] for r in rs)/n,3),
      "hit_cap":round(sum(cap)/n,3),
      "kept_going":round(sum(c and has_gold(r.get("reasoning") or "",r["gold"]) for c,r in zip(cap,rs))/n,3),
      "loop_rate":round(sum(c and tail_dup(txt(r))>=0.5 for c,r in zip(cap,rs))/n,3),
      "closed_thinking":round(sum(bool(r.get("answer_text")) for r in rs)/n,3),
      "mean_tokens":round(sum(r["tokens"] for r in rs)/n)}))
