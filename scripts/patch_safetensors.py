"""Write edited tensors into a copy of a .safetensors file in place (same dtype/shape -> same bytes),
without loading the whole model (fits a 4-8 GB RAM machine).
usage: python patch_safetensors.py <base.safetensors> <out.safetensors> <name=file.npy (bf16 bits as uint16 or float32)> ..."""
import json, shutil, struct, sys, hashlib, numpy as np
base, out, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
shutil.copyfile(base, out)
with open(out, "r+b") as f:
    n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); start = 8 + n
    for p in pairs:
        name, fn = p.split("=", 1); meta = hdr[name]; assert meta["dtype"] == "BF16"
        a = np.load(fn)
        if a.dtype == np.float32:  # round-to-nearest-even to bf16 bits
            u = a.view(np.uint32).astype(np.uint64); a = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        a = np.ascontiguousarray(a.astype(np.uint16).reshape(meta["shape"]))
        s, e = meta["data_offsets"]; assert e - s == a.nbytes
        f.seek(start + s); f.write(a.tobytes())
        print(name, "patched sha256", hashlib.sha256(a.tobytes()).hexdigest())
