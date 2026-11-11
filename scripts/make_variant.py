"""Make a v2 sweep variant from v1's direction and neuron list (results/mine_edit_info.json), without loading the model.
W' = W - alpha * outer(u, mask * (u @ W)) on down_proj of the chosen layers; mask = v1's neurons
(k<64 keeps the k with the largest |u.W[:,j]|).
usage: python make_variant.py <base_dir> <out_dir> --alpha 3 --layers 15,19 --k 64"""
import argparse, json, os, struct, shutil, numpy as np
ap=argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("out")
ap.add_argument("--alpha",type=float,required=True); ap.add_argument("--layers",default="15,19"); ap.add_argument("--k",type=int,default=64)
ap.add_argument("--info",default=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results","mine_edit_info.json"))
a=ap.parse_args(); L=[int(x) for x in a.layers.split(",")]; info=json.load(open(a.info))
os.makedirs(a.out,exist_ok=True)
st=[f for f in os.listdir(a.base) if f.endswith(".safetensors")][0]
for fn in os.listdir(a.base):
    if fn!=st and not fn.startswith("."):
        d=os.path.join(a.out,fn)
        if not os.path.exists(d): os.symlink(os.path.join(a.base,fn),d)
src=os.path.join(a.base,st); dst=os.path.join(a.out,st); shutil.copyfile(src,dst)
meta={"alpha":a.alpha,"layers":L,"k":a.k}
with open(dst,"r+b") as f:
    n=struct.unpack("<Q",f.read(8))[0]; hdr=json.loads(f.read(n)); start=8+n
    for l in L:
        name=f"model.language_model.layers.{l}.mlp.down_proj.weight"; m=hdr[name]; assert m["dtype"]=="BF16"
        s,e=m["data_offsets"]; f.seek(start+s); bits=np.frombuffer(f.read(e-s),np.uint16).reshape(m["shape"])
        W=(bits.astype(np.uint32)<<16).view(np.float32)
        u=np.array(info[str(l)]["u"],np.float32); u/=np.linalg.norm(u)
        uw=u@W; neur=np.array(info[str(l)]["neurons"])
        if a.k<len(neur): neur=neur[np.argsort(-np.abs(uw[neur]))[:a.k]]
        mask=np.zeros(W.shape[1],np.float32); mask[neur]=1
        Wn=W-a.alpha*np.outer(u,mask*uw)
        uu=Wn.view(np.uint32).astype(np.uint64); nb=((uu+0x7FFF+((uu>>16)&1))>>16).astype(np.uint16)
        f.seek(start+s); f.write(nb.tobytes())
        meta[str(l)]={"neurons":sorted(neur.tolist()),"relative_change":float(np.linalg.norm(Wn-W)/np.linalg.norm(W))}
        print("layer",l,"alpha",a.alpha,"k",len(neur),"rel",round(meta[str(l)]["relative_change"],4),flush=True)
json.dump(meta,open(os.path.join(a.out,"edit_info.json"),"w"))
