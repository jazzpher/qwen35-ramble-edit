"""Rebuild Astra's published p50c edit (RikiyaT/lm_understand-logs exp8_math) as uint16 bf16 bits, hash-checked."""
import sys, os, json, hashlib, numpy as np
sys.path.insert(0, os.path.expanduser("~/ref_logs/exp8_math/code"))
import st_numpy as S, apply_edit as A
base, edit_dir, out = sys.argv[1:4]
man = json.load(open(os.path.join(edit_dir, "manifest.json"))); z = np.load(os.path.join(edit_dir, "edit.npz"))
T = S.load(base, [t["name"] for t in man["tensors"]]); os.makedirs(out, exist_ok=True)
for i, t in enumerate(man["tensors"]):
    dt, b = T[t["name"]]; assert hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest() == t["base_sha256"]
    e = A.rebuild(b, t, z, i); assert hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest() == t["edited_sha256"]
    np.save(os.path.join(out, t["name"] + ".npy"), np.ascontiguousarray(e).view(np.uint16)); print(t["name"], "hash OK")
