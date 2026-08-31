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
      --dataset voxpopuli --split test --batch_size 128 --device 0

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

import torch
from tqdm import tqdm
import evaluate

from datasets import IterableDataset
from normalizer import data_utils
from transformers import AutoModel, AutoProcessor

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision("high")


def load_model(model_id, device, revision=None):
    """Load the packaged CTC Conformer via AutoModel (trust_remote_code)."""
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    return model.eval().to(device)


def transcribe_batch(model, processor, audios, device):
    """One batch of waveforms -> list of raw transcription strings."""
    inputs = processor(audios, device=device)
    output = model.transcribe(**inputs)
    return processor.batch_decode(output.preds)


def prefetch_inputs(processor, audios, batch_size, device):
    """Yield processor(chunk) one batch ahead, on a background thread.

    The next batch's prep -- numpy build + host->device copy + mel front-end --
    runs on a worker thread while the main thread runs the current batch's
    transcribe (which blocks on its internal .tolist() sync). This overlaps the
    CPU/copy-bound prep with GPU compute so it no longer serializes in front of
    the encoder. Profiling showed prep (~18k stage-RTFx) and transcribe (~9.3k)
    each fast but serial; overlapping them lifts end-to-end toward the
    transcribe-bound ceiling. Single worker = at most one batch in flight, so
    GPU memory stays bounded.
    """
    import queue
    import threading

    chunks = [audios[i:i + batch_size] for i in range(0, len(audios), batch_size)]
    q = queue.Queue(maxsize=1)

    def worker():
        for chunk in chunks:
            q.put(processor(chunk, device=device))
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
    model = load_model(args.model_id, device, revision=args.revision)
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
        for inputs in prefetch_inputs(processor, audios, args.batch_size, device):
            output = model.transcribe(**inputs)
            preds.extend(processor.batch_decode(output.preds))
        return preds

    # --- Warmup (untimed): a few batches to trigger lazy CUDA init / autotune ---
    if args.warmup_steps:
        n = min(args.warmup_steps * args.batch_size, len(audios))
        for i in range(0, n, args.batch_size):
            transcribe_batch(model, processor, audios[i:i + args.batch_size], device)
        if is_cuda:
            torch.cuda.synchronize()

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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=2)

    args = parser.parse_args()
    parser.set_defaults(streaming=False)
    main(args)
