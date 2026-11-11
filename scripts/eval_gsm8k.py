"""GSM8K sample eval against a llama.cpp server (OpenAI API), thinking on, Astra/post sampling.
usage: python eval_gsm8k.py <label> [max_tokens] [port] [n]
"ramble" = answer hit max_tokens (finish_reason == length)."""
import json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
label=sys.argv[1]; MAXT=int(sys.argv[2]) if len(sys.argv)>2 else 3000
PORT=sys.argv[3] if len(sys.argv)>3 else "8080"; N=int(sys.argv[4]) if len(sys.argv)>4 else 40
SUFFIX="\nPlease reason step by step, and put your final answer within \\boxed{}."
import os
probs=[json.loads(l) for l in open(os.environ.get("SAMPLE","results/gsm8k_sample.jsonl"))][:N]
def last_boxed(t):
    out=None; i=0
    while True:
        j=t.find("\\boxed{",i)
        if j<0: return out
        k=j+7; d=1
        while k<len(t) and d:
            d+= {"{":1,"}":-1}.get(t[k],0); k+=1
        out=t[j+7:k-1]; i=k
def num(s):
    if s is None: return None
    s=s.replace(",","").replace("\\$","").replace("$","").replace("\\%","").replace("%","")
    s=re.sub(r"\\text\{[^}]*\}","",s); s=re.sub(r"\\(dfrac|frac)\{(-?\d+)\}\{(\d+)\}",lambda m:str(float(m.group(2))/float(m.group(3))),s)
    m=re.findall(r"-?\d+(?:\.\d+)?",s)
    return float(m[-1]) if m else None
def run(p):
    body={"messages":[{"role":"user","content":p["question"]+SUFFIX}],"max_tokens":MAXT,"temperature":0.6,
          "top_p":0.95,"top_k":20,"presence_penalty":1.5,"seed":0,"chat_template_kwargs":{"enable_thinking":True}}
    req=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"})
    t=time.time(); r=json.load(urllib.request.urlopen(req,timeout=3600))
    m=r["choices"][0]["message"]; reas=m.get("reasoning_content") or ""; ans=m.get("content") or ""
    b=last_boxed(ans) or last_boxed(reas); v=num(b)
    ok= v is not None and abs(v-float(p["gold"]))<1e-6
    return {**p,"pred":b,"correct":ok,"finish":r["choices"][0]["finish_reason"],"tokens":r["usage"]["completion_tokens"],
            "sec":round(time.time()-t,1),"answer_text":ans,"reasoning":reas}
out=f"results/{label}_gsm8k.jsonl"; res=[json.loads(l) for l in open(out)] if os.path.exists(out) else []
done={r["idx"] for r in res}; probs=[p for p in probs if p["idx"] not in done]
t0=time.time()
with ThreadPoolExecutor(4) as ex, open(out,"a") as f:
    for r in ex.map(run,probs):
        res.append(r); f.write(json.dumps(r)+"\n"); f.flush()
        print(len(res),r["idx"],r["finish"],r["tokens"],"OK" if r["correct"] else "x",flush=True)
n=len(res); s={"label":label,"n":n,"max_tokens":MAXT,"accuracy":sum(r["correct"] for r in res)/n,
   "ramble_rate":sum(r["finish"]=="length" for r in res)/n,"mean_tokens":sum(r["tokens"] for r in res)/n,"minutes":round((time.time()-t0)/60,1)}
json.dump(s,open(f"results/{label}_summary.json","w"),indent=1); print(s)
