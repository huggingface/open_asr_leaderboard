"""Export the Phonon-2 container to an exact dense NeMo checkpoint (.nemo) of nvidia/parakeet-tdt-0.6b-v3's architecture.

The container's low-bit encoder, int6 and fp16 records are expanded exactly to fp32 (fermion_container.read_container), renamed from the
HF ParakeetForTDT names back to NeMo's with the INVERSE of transformers' convert_nemo_to_hf.py mapping (pure renames; no tensor
transform except the kernel-1 pointwise convs, stored [O, I] and restored to [O, I, 1] as both frameworks expect), loaded strict
into the base model restored from the Hub, and saved with `save_to`. Only the preprocessor's featurizer buffers (window, fb) come
from the base .nemo -- they are not weights. Usage: python nemo_export.py model.fermion out.nemo
"""
import json, re, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fermion_container import read_container

NEMO_TO_HF = {  # transformers/src/transformers/models/parakeet/convert_nemo_to_hf.py (NEMO_TO_HF_WEIGHT_MAPPING + NEMO_TDT_WEIGHT_MAPPING)
    r"encoder\.pre_encode\.conv\.": r"encoder.subsampling.layers.", r"encoder\.pre_encode\.out\.": r"encoder.subsampling.linear.",
    r"encoder\.pos_enc\.": r"encoder.encode_positions.", r"encoder\.layers\.(\d+)\.conv\.batch_norm\.": r"encoder.layers.\1.conv.norm.",
    r"decoder\.decoder_layers\.0\.(weight|bias)": r"ctc_head.\1", r"linear_([kv])": r"\1_proj", r"linear_out": r"o_proj", r"linear_q": r"q_proj",
    r"pos_bias_([uv])": r"bias_\1", r"linear_pos": r"relative_k_proj",
    r"decoder\.prediction\.embed\.": r"decoder.embedding.", r"decoder\.prediction\.dec_rnn\.lstm\.": r"decoder.lstm.",
    r"joint\.enc\.": r"encoder_projector.", r"joint\.pred\.": r"decoder.decoder_projector.", r"joint\.joint_net\.2\.": r"joint.head.",
}
def to_hf(key):
    for a, b in NEMO_TO_HF.items():
        key = re.sub(a, b, key)
    return key

def container_state_dict(container):
    tensors, index = read_container(container, with_raw=False)
    sd = {}
    for k, v in tensors.items():
        if k.endswith("num_batches_tracked"):
            v32 = float(np.asarray(v, dtype=np.float32).reshape(-1)[0]); sd[k] = torch.tensor(int(v32) if np.isfinite(v32) else 0, dtype=torch.int64)
        else:
            arr = np.ascontiguousarray(v.astype(np.float32))
            if arr.ndim == 2 and re.search(r"\.conv\.pointwise_conv[12]\.weight$", k): arr = arr[:, :, None]
            sd[k] = torch.from_numpy(arr)
    return sd, index

def main():
    container, out = sys.argv[1], sys.argv[2]; base = sys.argv[3] if len(sys.argv) > 3 else "nvidia/parakeet-tdt-0.6b-v3"
    from nemo.collections.asr.models import ASRModel
    t0 = time.time(); m = ASRModel.from_pretrained(base, map_location="cpu"); m.eval()
    hf, index = container_state_dict(container); used = set(); new = {}; report = {"base": base, "container": container, "records": len(index), "shape_fixes": [], "kept_from_base": []}
    for k, v in m.state_dict().items():
        if k.endswith(("featurizer.window", "featurizer.fb")) or k.startswith("preprocessor."):
            new[k] = v; report["kept_from_base"].append(k); continue
        hk = to_hf(k)
        if hk not in hf: raise SystemExit(f"no container tensor for NeMo key {k} (looked for {hk})")
        t = hf[hk]; used.add(hk)
        if t.shape != v.shape:
            if t.numel() == v.numel(): report["shape_fixes"].append([k, list(t.shape), list(v.shape)]); t = t.reshape(v.shape)
            else: raise SystemExit(f"shape mismatch {k}: container {tuple(t.shape)} vs nemo {tuple(v.shape)}")
        new[k] = t.to(v.dtype)
    report["unused_container_tensors"] = sorted(set(hf) - used)
    if report["unused_container_tensors"]: raise SystemExit(f"container tensors not consumed: {report['unused_container_tensors'][:8]}")
    missing, unexpected = m.load_state_dict(new, strict=True), None
    m.save_to(out); report.update({"out": out, "out_bytes": Path(out).stat().st_size, "params": sum(p.numel() for p in m.parameters()), "seconds": round(time.time() - t0, 1),
                                   "strict_load": "ok", "n_weights_replaced": len(new) - len(report["kept_from_base"])})
    json.dump(report, open(out + ".export.json", "w"), indent=1); print(json.dumps({k: v for k, v in report.items() if k not in ("kept_from_base",)}, default=str)[:1500])

if __name__ == "__main__":
    main()
