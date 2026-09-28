"""Single-stream (batch 1) RTF through NeMo greedy TDT (label-looping + CUDA graphs = NeMo's greedy_batch defaults), 10 LS test-other utts."""
import json, sys, time, torch, soundfile as sf
from datasets import load_dataset
from nemo.collections.asr.models import ASRModel
mid, tag = sys.argv[1], sys.argv[2]
m = ASRModel.restore_from(mid, map_location="cuda") if mid.endswith(".nemo") else ASRModel.from_pretrained(mid, map_location="cuda")
m.to(torch.bfloat16).eval()
if m.cfg.decoding.strategy != "beam": m.cfg.decoding.strategy = "greedy_batch"; m.change_decoding_strategy(m.cfg.decoding)
ds = load_dataset("hf-audio/open-asr-leaderboard", "librispeech", split="test.other", streaming=True, token=True).take(10)
files = []
for i, x in enumerate(ds):
    p = f"/tmp/b1_{i}.wav"; sf.write(p, x["audio"]["array"], x["audio"]["sampling_rate"]); files.append((p, len(x["audio"]["array"]) / x["audio"]["sampling_rate"]))
with torch.inference_mode():
    for p, _ in files[:2]: m.transcribe([p], batch_size=1, verbose=False, num_workers=0)
    rows = []
    for p, dur in files:
        torch.cuda.synchronize(); t = time.perf_counter(); m.transcribe([p], batch_size=1, verbose=False, num_workers=0); torch.cuda.synchronize(); w = time.perf_counter() - t
        rows.append({"audio_s": round(dur, 2), "wall_s": round(w, 4), "rtf": round(w / dur, 4)})
ta = sum(r["audio_s"] for r in rows); tw = sum(r["wall_s"] for r in rows)
out = {"model": mid, "tag": tag, "gpu": torch.cuda.get_device_name(0), "path": "NeMo greedy_batch TDT (label-looping, CUDA graphs), bf16, batch 1, file input", "n": len(rows), "rtf_pooled": round(tw / ta, 4), "x_realtime_pooled": round(ta / tw, 1), "ms_per_audio_second": round(1000 * tw / ta, 1), "rows": rows}
json.dump(out, open(f"results/batch1_nemo_{tag}.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
