"""Find the 'keep talking' (loop) direction and project it out of MLP down_proj columns.
usage: python find_direction.py <generations.jsonl ...> --alpha 6 --k 64 --out /home/user/edited_x
Teacher-forced read of the model's own reasoning texts. Token positions whose word-level 8-gram already
appeared earlier in the same text, or that come after the correct answer already appeared (--contrast both) = 'keep talking'; the rest of the text (before the loop starts) = 'productive'.
u_l = unit(mean resid_after_layer_l[rep] - mean resid_after_layer_l[prod]) for l in LAYERS.
Writer neurons: per down_proj column j, contribution c_j = act_j * (u . W[:,j]); take the top-k by
mean_rep(c_j) - mean_prod(c_j). Edit (same form as Astra's): W' = W - alpha * outer(u, mask * (u @ W)).
No training, no gradients."""
import argparse, gc, json, os, sys, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
ap=argparse.ArgumentParser(); ap.add_argument("files",nargs="+"); ap.add_argument("--alpha",type=float,default=6)
ap.add_argument("--k",type=int,default=64); ap.add_argument("--layers",default="15,19"); ap.add_argument("--out",required=True)
ap.add_argument("--maxtok",type=int,default=2500); ap.add_argument("--contrast",default="both",choices=["rep","gold","both"]); ap.add_argument("--base",default="/home/user/qwen08")
a=ap.parse_args(); LAYERS=[int(x) for x in a.layers.split(",")]
torch.set_num_threads(2)
tok=AutoTokenizer.from_pretrained(a.base)
m=AutoModelForCausalLM.from_pretrained(a.base,dtype=torch.bfloat16); m.eval()
cur={}
for l in LAYERS:
    m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod,inp,l=l: cur.__setitem__(l,inp[0][0].float()))
S={l:{k:torch.zeros(1024) for k in ("hr","hp")} for l in LAYERS}
C={l:{k:torch.zeros(3584) for k in ("cr","cp")} for l in LAYERS}
W={l:m.model.layers[l].mlp.down_proj.weight.float() for l in LAYERS}
texts=[json.loads(x) for f in a.files for x in open(f)]
store=[]
for r in texts:
    reas=r.get("reasoning") or ""
    if len(reas)<200: continue
    prompt=tok.apply_chat_template([{"role":"user","content":r["question"]+"\nPlease reason step by step, and put your final answer within \\boxed{}."}],
        tokenize=False,add_generation_prompt=True,enable_thinking=True)
    p_ids=tok(prompt,add_special_tokens=False).input_ids; g=tok(reas,add_special_tokens=False,return_offsets_mapping=True)
    g_ids=g.input_ids[:a.maxtok]; offs=g.offset_mapping[:a.maxtok]
    # word 8-gram repetition per char position -> per token
    words=[]; pos=0
    for w in reas.split():
        s=reas.find(w,pos); words.append((s,w)); pos=s+len(w)
    seen=set(); rep_char=np.zeros(len(reas)+1,bool)
    for i in range(7,len(words)):
        gr=tuple(w for _,w in words[i-7:i+1])
        if gr in seen: rep_char[words[i][0]]=True
        seen.add(gr)
    rep_tok=np.array([rep_char[s:e+1].any() for s,e in offs])
    # 'kept going': every token after the correct answer first shows up in the reasoning
    import re
    gm=re.search(r"(?<![\d.])"+re.escape(str(r["gold"]).replace(",",""))+r"(?![\d])",reas)
    if len(str(r["gold"]).replace(",",""))<2: gm=None   # a 1-digit answer shows up by chance
    gold_tok=np.array([gm is not None and s>=gm.end()+40 for s,e in offs])
    if a.contrast=="gold": rep_tok=gold_tok
    elif a.contrast=="both": rep_tok=rep_tok|gold_tok
    if rep_tok.any(): first=int(np.argmax(rep_tok)); prod_tok=np.zeros(len(g_ids),bool); prod_tok[:first]=True
    else: prod_tok=np.ones(len(g_ids),bool)
    if rep_tok.sum()<20 and prod_tok.sum()<20: continue
    ids=torch.tensor([p_ids+g_ids])
    with torch.no_grad(): out=m.model(input_ids=ids,output_hidden_states=True,use_cache=False)
    off=len(p_ids)
    for l in LAYERS:
        h=out.hidden_states[l+1][0,off:].float(); act=cur[l][off:]; uW=None
        rt=torch.from_numpy(rep_tok); pt=torch.from_numpy(prod_tok)
        store.append(None)
        S[l]["hr"]+=h[rt].sum(0); S[l]["hp"]+=h[pt].sum(0)
        C[l].setdefault("acts_r",torch.zeros(3584)); C[l].setdefault("acts_p",torch.zeros(3584))
        C[l]["acts_r"]+=act[rt].sum(0); C[l]["acts_p"]+=act[pt].sum(0)
        C[l].setdefault("nr",0); C[l].setdefault("np",0); C[l]["nr"]+=int(rt.sum()); C[l]["np"]+=int(pt.sum())
    print(r["idx"],r["finish"],"tokens",len(g_ids),"rep",int(rep_tok.sum()),"prod",int(prod_tok.sum()),flush=True)
    del out; gc.collect()
edit={}; info={"alpha":a.alpha,"k":a.k,"layers":LAYERS,"texts":len(texts)}
for l in LAYERS:
    nr,np_=C[l]["nr"],C[l]["np"]
    d=S[l]["hr"]/nr-S[l]["hp"]/np_; u=d/d.norm()
    uw=u@W[l]                                   # (3584,) how much each neuron's column writes along u
    score=(C[l]["acts_r"]/nr-C[l]["acts_p"]/np_)*uw   # mean contribution difference rep - prod
    idx=torch.topk(score,a.k).indices.sort().values
    mask=torch.zeros(3584); mask[idx]=1
    Wn=W[l]-a.alpha*torch.outer(u,mask*uw)
    edit[l]=Wn.to(torch.bfloat16)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","edit",f"u_layer{l}.npy"),u.numpy()) if os.path.isdir("edit") else None
    info[str(l)]={"n_rep_tokens":nr,"n_prod_tokens":np_,"diff_norm":float(d.norm()),"neurons":idx.tolist(),
        "relative_change":float((Wn-W[l]).norm()/W[l].norm()),"changed_columns":a.k,"u":u.tolist()}
    print("layer",l,"rep",nr,"prod",np_,"rel change",round(info[str(l)]["relative_change"],4),flush=True)
del m,W; gc.collect()
from safetensors.torch import load_file, save_file
os.makedirs(a.out,exist_ok=True)
for fn in os.listdir(a.base):
    if not fn.endswith(".safetensors") and not fn.startswith("."):
        dst=os.path.join(a.out,fn)
        if not os.path.exists(dst): os.symlink(os.path.join(a.base,fn),dst)
st=[f for f in os.listdir(a.base) if f.endswith(".safetensors")][0]
T=load_file(os.path.join(a.base,st))
for l in LAYERS:
    n=f"model.language_model.layers.{l}.mlp.down_proj.weight"; assert T[n].shape==edit[l].shape; T[n]=edit[l]
save_file(T,os.path.join(a.out,st),{"format":"pt"})
json.dump(info,open(os.path.join(a.out,"edit_info.json"),"w"))
print("saved",a.out)
