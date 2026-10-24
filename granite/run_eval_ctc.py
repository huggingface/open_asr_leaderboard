"""run_eval variant that runs the packaged CTC Conformer (hf_package/export)
through the standard HuggingFace AutoModel / AutoProcessor API.

The whole pipeline -- log-mel front-end + frame-stacking, the patched
GraniteSpeechCTCEncoder + `out` head, CTC greedy decode, and SentencePiece
detokenization -- now lives inside the packaged model/processor, so this script
only loads them, feeds audio, and scores. Build the export dir first with:

  python hf_package/build_package.py --ckpt param/0.0005/29.safetensors \
      --out hf_package/export

Then:
  python run_eval_ctc.py --model_id hf_package/export \
      --dataset_path hf-audio/esb-datasets-test-only-sorted \
      --dataset voxpopuli --split test --device 0

Inference runs batched with bucketed sequence lengths, bf16 weights, and
torch.compile(mode="reduce-overhead") on the encoder.

Timing follows NVIDIA's nemo_asr/run_eval.py: all audio is gathered up front
(decode / I/O outside the timer), then the full transcription loop is timed
ONCE with a single sync at the end -- no per-batch synchronize() -- so CPU data
prep overlaps GPU compute and RTFx reflects sustained throughput. RTFx =
total_audio_seconds / total_wall_time.

The processor returns RAW (un-normalized) text, and raw references/predictions
are what get written to the manifest -- matching the other run_eval scripts, so
that scoring-time normalization stays revisable. data_utils.normalizer is only
applied to the WER printed at the end of this script.
"""

import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm
import evaluate

from datasets import IterableDataset
from normalizer import data_utils
from transformers import AutoModel, AutoProcessor

wer_metric = evaluate.load("wer")
# TF32 for the fp32 mel front-end (torchaudio's MelScale is a matmul); the
# encoder itself is bf16, so this does not affect it.
torch.set_float32_matmul_precision("high")


def load_model(model_id, device, revision=None, compile_scope="layers"):
    """Load the packaged CTC Conformer, cast to bf16, encoder compiled."""
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    model = model.eval().to(device).to(torch.bfloat16)
    # Static shapes, one graph per frame bucket; dynamic=True re-specializes on
    # every distinct T' and is far slower here. dynamic=False must be explicit:
    # the default auto-dynamic marks T' symbolic on the first recompile, and
    # inductor then crashes in sympy on the encoder's padding/subsampling math.
    # Per-layer compile keys every (shape, block-variant) pair on ONE shared code
    # object -- subsampling blocks and the trailing block see different T' -- so
    # the limit must cover ~3x the bucket count.
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 2048
    compile_kw = dict(mode="reduce-overhead", dynamic=False)
    layers = getattr(model.encoder, "layers", None)
    if compile_scope == "layers" and layers is not None:
        # Regional compilation: the conformer blocks are the same class, so with
        # inline_inbuilt_nn_modules (params as graph inputs) they share one
        # compiled graph per input shape. Each bucket compiles ~one block's worth
        # of code instead of the whole 16-layer encoder. No CUDA graphs here:
        # every block replays the same graph, so a block's outputs (e.g. the mask
        # it hands to the next block) live in a CUDA-graph buffer that the next
        # block's replay overwrites ("accessing tensor output of CUDAGraphs that
        # has been overwritten").
        for layer in layers:
            layer.compile(mode="default", dynamic=False)
    elif compile_scope != "none":
        model.encoder = torch.compile(model.encoder, **compile_kw)
    return model


def _to_bf16(inputs):
    """Cast input_features to the bf16 weight dtype; the processor emits fp32."""
    feats = inputs.get("input_features")
    if feats is not None and feats.dtype != torch.bfloat16:
        inputs = {**inputs, "input_features": feats.to(torch.bfloat16)}
    return inputs


def bucketed_frames(processor, num_samples, frame_bucket):
    """Encoder frame count T' for a batch whose longest clip has num_samples,
    rounded up to a multiple of frame_bucket."""
    mel = num_samples // processor.hop_length
    t_prime = -(-mel // processor.stack_factor)
    return frame_bucket * -(-t_prime // frame_bucket)


def featurize(processor, audios, device, frame_bucket):
    """processor(audios) with T' padded up to a multiple of frame_bucket.

    The packaged processor emits each batch's exact T' (it ignores any
    frame_bucket attribute), so every batch would be a fresh compiled shape.
    Right-pad the waveform batch to exactly T'_bucketed * stack * hop samples;
    the mask is built from the per-clip TRUE lengths, so the extra frames are
    masked as padding like any other right-padding in the batch.
    """
    x, lengths = processor.prepare_audio(audios, device=device, return_lengths=True)
    target = bucketed_frames(processor, x.shape[1], frame_bucket) * processor.stack_factor * processor.hop_length
    if x.shape[1] < target:
        x = torch.nn.functional.pad(x, (0, target - x.shape[1]))
    # A padded tensor through __call__ (not _frontend directly): __call__ moves the
    # mel/STFT window to x's device. Given a tensor it emits no mask (no lengths),
    # so build it here from the true lengths.
    feats = processor(x)["input_features"]
    mask = processor._build_mask(lengths, feats.shape[1], feats.device)
    return {"input_features": feats, "attention_mask": mask}


def prefetch_inputs(processor, audios, batch_size, device, frame_bucket, pad_last=False):
    """Yield (processor(chunk), n_real) one batch ahead, on a background thread.

    The next batch's prep -- numpy build + host->device copy + mel front-end --
    runs on a worker thread while the main thread runs the current batch's
    transcribe (which blocks on its internal .tolist() sync). This overlaps the
    CPU/copy-bound prep with GPU compute so it no longer serializes in front of
    the encoder. Profiling showed prep (~18k stage-RTFx) and transcribe (~9.3k)
    each fast but serial; overlapping them lifts end-to-end toward the
    transcribe-bound ceiling. Single worker = at most one batch in flight, so
    GPU memory stays bounded.

    pad_last tops the final short chunk up to batch_size with 1-sample dummy
    clips so it reuses the full-size compiled graph instead of triggering its own
    compile; `n_real` lets the caller drop the dummy predictions.
    """
    import queue
    import threading

    chunks = [audios[i:i + batch_size] for i in range(0, len(audios), batch_size)]
    counts = [len(c) for c in chunks]
    if pad_last and chunks and len(chunks[-1]) < batch_size:
        short = batch_size - len(chunks[-1])
        chunks[-1] = chunks[-1] + [np.zeros(1, dtype=np.float32)] * short
    q = queue.Queue(maxsize=1)

    # inference_mode is THREAD-LOCAL and torch.compile guards on the resulting
    # dispatch key set, so tensors built here must mirror the caller's mode or
    # every warmed graph is invalidated and recompiles inside the timed loop.
    inference = torch.is_inference_mode_enabled()

    def worker():
        with torch.inference_mode(inference):
            for chunk, n_real in zip(chunks, counts):
                q.put((featurize(processor, chunk, device, frame_bucket), n_real))
        q.put(None)  # sentinel

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item
    t.join()


def main(args):
    device = torch.device(f"cuda:{args.device}" if (torch.cuda.is_available() and args.device >= 0) else "cpu")
    model = load_model(args.model_id, device, revision=args.revision,
                       compile_scope=args.compile_scope)
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    processor = AutoProcessor.from_pretrained(
        args.model_id, trust_remote_code=True, revision=args.revision
    )
    is_cuda = device.type == "cuda"

    # --- Gather the whole corpus up front (decode / I/O OUTSIDE the timer) ---
    # Following NVIDIA's nemo_asr/run_eval.py: prepare all audio first, then time
    # the full transcription loop ONCE (no per-batch synchronize), so CPU-side
    # data prep overlaps GPU compute and RTFx reflects sustained throughput
    # rather than per-batch barriers.
    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        # NOTE (ebezzam) chunked datasets are always map-style, regardless of --streaming
        if isinstance(dataset, IterableDataset):
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset)

    # Chunked datasets give every chunk of a session the *session* transcript as
    # its reference, so per-chunk scoring is meaningless: the chunk ids are kept
    # alongside each row and the predictions are merged per session before WER.
    is_chunked = data_utils.is_chunked_dataset(args.dataset_path)

    audios, durations, references = [], [], []
    chunk_metadata = {key: [] for key in data_utils.CHUNK_METADATA_KEYS} if is_chunked else {}
    for sample in tqdm(iter(dataset), desc="Loading samples..."):
        arr = sample["audio"]["array"]
        audios.append(arr)
        durations.append(len(arr) / 16000.0)
        references.append(sample["original_text"])  # raw; normalization applied at scoring time
        for key in chunk_metadata:
            chunk_metadata[key].append(sample[key])

    # Sort by duration (desc) so each batch is length-homogeneous -> less padding
    # waste (mirrors the nemo harness). Chunk ids ride along so each row keeps
    # its own, and the merge re-orders by chunk_index anyway.
    order = sorted(range(len(durations)), key=lambda k: durations[k], reverse=True)
    audios = [audios[i] for i in order]
    durations = [durations[i] for i in order]
    references = [references[i] for i in order]
    chunk_metadata = {key: [values[i] for i in order] for key, values in chunk_metadata.items()}

    def run_all():
        # Prefetched: next batch's prep overlaps the current batch's GPU work.
        preds = []
        for inputs, n_real in prefetch_inputs(processor, audios, args.batch_size,
                                              device, args.frame_bucket, pad_last=True):
            output = model.transcribe(**_to_bf16(inputs))
            preds.extend(processor.batch_decode(output.preds[:n_real]))
        return preds

    # --- Warmup (untimed): one batch per distinct compiled shape ---
    # Compile is per-shape, so warming a fixed number of batches would leave later
    # buckets to compile inside the timed loop.
    seen = set()
    warm = []
    for i in range(0, len(audios), args.batch_size):
        chunk = audios[i:i + args.batch_size]
        # Same T' featurize() will produce for this batch in the timed loop.
        key = bucketed_frames(processor, max(len(a) for a in chunk), args.frame_bucket)
        if key not in seen:
            seen.add(key)
            if len(chunk) < args.batch_size:
                chunk = chunk + [np.zeros(1, dtype=np.float32)] * (args.batch_size - len(chunk))
            warm.append(chunk)
    print(f"Compile warmup: {len(warm)} distinct shape(s) (this is slow, once each)...")
    t0 = time.time()
    # Must warm under inference_mode and through prefetch_inputs, matching the
    # timed loop: dynamo guards on the input tensors' dispatch key set, and a
    # mismatch moves every recompile inside the timer.
    #
    # Two passes per shape: with mode="reduce-overhead" the first invocation only
    # warms a shape and capture happens on the second, so a single pass leaves the
    # graphs to be recorded inside the timed loop.
    with torch.inference_mode():
        for _ in range(2):
            for chunk in warm:
                for inputs, _ in prefetch_inputs(processor, chunk, len(chunk), device, args.frame_bucket):
                    model.transcribe(**_to_bf16(inputs))
        if is_cuda:
            torch.cuda.synchronize()
    print(f"Compile warmup done in {time.time() - t0:.1f}s")
    if is_cuda:
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")

    # --- Timed loop: whole corpus, single sync at the end ---
    start_time = time.time()
    with torch.inference_mode():
        predictions = run_all()  # raw; normalization applied at scoring time
    if is_cuda:
        torch.cuda.synchronize()
    total_time = time.time() - start_time

    avg_time = total_time / len(audios)

    manifest_path = data_utils.write_manifest(
        references, predictions, args.model_id,
        args.dataset_path, args.dataset, args.split,
        audio_length=durations,
        transcription_time=[avg_time] * len(audios),
        extra_fields=chunk_metadata if is_chunked else None,
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    if is_chunked:
        # Concatenate each session's chunk predictions (in chunk order) and score
        # against the session transcript.
        sessions = data_utils.merge_chunked_manifest(data_utils.read_manifest(manifest_path))
        references = [session["text"] for session in sessions]
        predictions = [session["pred_text"] for session in sessions]

    norm_refs = [data_utils.normalizer(r) for r in references]
    norm_preds = [data_utils.normalizer(p) for p in predictions]
    wer = wer_metric.compute(references=norm_refs, predictions=norm_preds)
    wer = round(100 * wer, 2)
    rtfx = round(sum(durations) / total_time, 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="hf_package/export",
                        help="Packaged model dir or Hub id (built by hf_package/build_package.py).")
    parser.add_argument("--revision", type=str, default=None,
                        help="Model repo revision (branch, tag or commit sha). Defaults to the main branch.")
    parser.add_argument("--dataset_path", type=str, default="esb/datasets")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Sweep-optimal default: 128.")
    parser.add_argument("--frame_bucket", type=int, default=32,
                        help="Pad stacked frame lengths up to a multiple of this so the "
                             "compiled encoder sees few distinct shapes. Sweep-optimal "
                             "default: 32.")
    parser.add_argument("--compile_scope", choices=["layers", "encoder", "none"], default="layers",
                        help="What to torch.compile: each conformer block (fast compile, "
                             "one graph shared by all blocks), the whole encoder (slowest "
                             "compile), or nothing.")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    # data_utils.load_data reads args.streaming; set it without a CLI flag, since
    # this harness always reads the dataset locally.
    parser.set_defaults(streaming=False)

    args = parser.parse_args()
    main(args)
