"""Multilingual ASR evaluation for BharatGen Shrutam-2.

Shrutam-2 (https://huggingface.co/bharatgenai/Shrutam-2) is a Conformer encoder
bridged to a frozen Llama decoder through a SMEAR Mixture-of-Experts projector.
It ships as loose Python modules on the Hub rather than a Transformers model
class, so the code is downloaded at runtime and imported from the snapshot dir.

The reference `inference_script.py` on the Hub transcribes one file at a time.
This script adds batching; see `encode_batch` and `_patch_moe_masking` for the
places where the upstream code assumes an unpadded batch of one. Batched output
is verified to match one-at-a-time inference to float32 round-off.
"""

import argparse
import json
import os
import re
import sys
import time

import evaluate
import torch
import torch.nn.functional as F
from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

from normalizer import data_utils
from normalizer.eval_utils import normalize_compound_pairs

wer_metric = evaluate.load("wer")

SAMPLING_RATE = 16_000

# The 12 languages Shrutam-2 supports. The language name is interpolated into
# the prompt (e.g. "Transcribe speech to Hindi text."), which is how the model
# is steered towards a target script.
LANGUAGE_NAMES = {
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "or": "Odia",
    "bn": "Bengali",
    "ur": "Urdu",
    "as": "Assamese",
    "gu": "Gujarati",
    "pa": "Punjabi",
}


def _patch_moe_masking(model):
    """Let the SMEAR router see which encoder frames are padding.

    `ASRModel.forward` calls `MoELayer_routing(ds_encoder_outs)` with no mask, so
    the utterance-level expert gating averages the router probabilities over
    padded frames too — a sample's chosen expert mixture, and hence its
    transcription, would depend on what else landed in its batch. The router
    already accepts a mask; this makes it pick one up from `moe.frame_mask`.

    At batch size 1 the mask is all ones, so results match the reference
    `inference_script.py` exactly.
    """
    moe = model.MoELayer_routing
    unmasked_forward = moe.forward

    def masked_moe_forward(x, mask=None):
        if mask is None:
            mask = getattr(moe, "frame_mask", None)
        return unmasked_forward(x, mask=mask)

    moe.forward = masked_moe_forward


def encode_batch(encoder, wavs):
    """Batched replacement for `SpeechEncoder.encode`, invariant to batching.

    `SpeechEncoder.encode` zero-pads the *waveforms* and pushes the whole padded
    batch through the front end. Everything from the STFT down to the conformer
    subsampling then mixes a clip with its neighbours' padding:

    * `MelSpectrogramPreprocessor` normalizes per feature over the full padded
      time axis, and zero padding is a large constant in log-mel space, so the
      statistics of every real frame get skewed by the batch's longest clip.
    * `torch.stft(center=True)` reflect-pads the signal, so the final frames of
      a padded clip see zeros where an unpadded clip would see its own tail.
    * `ConvSubsampling`'s convolutions have biases, so zero padding does not
      stay zero: by the third conv the padded region carries a non-zero
      constant that bleeds into the last real frame.

    Running the front end per clip on its own unpadded waveform avoids all
    three. It is cheap next to the 24-layer conformer body, which is batched
    below and already masks padding (key-masked attention, and the convolution
    module zeroes padded positions before its depthwise conv).

    Returns (features [B, T', D], encoder_lengths [B]) like the original, with
    padded frames zeroed so the downstream `EncoderDownsamplerCov1d` — whose
    first conv has kernel 3 — sees the same zeros it would see at the end of an
    unpadded sequence.
    """
    conformer = encoder.model
    device = encoder.device

    subsampled, lengths = [], []
    for wav in wavs:
        wav = wav.to(device)
        mel, mel_length = encoder.preprocessor(
            input_signal=wav, length=torch.tensor([wav.shape[-1]], device=device)
        )
        x, out_length = conformer.subsampling(mel, mel_length)
        subsampled.append(x[0])
        lengths.append(int(out_length[0]))

    max_frames = max(frames.shape[0] for frames in subsampled)
    x = torch.stack(
        [F.pad(frames, (0, 0, 0, max_frames - frames.shape[0])) for frames in subsampled]
    )
    encoder_lengths = torch.tensor(lengths, device=device)

    # Remainder of `ConformerEncoder.forward`, which is padding-safe.
    x = x * conformer.scale
    pos_emb = conformer.pos_enc(max_frames)
    mask = torch.arange(max_frames, device=device)[None, :] < encoder_lengths[:, None]
    for layer in conformer.layers:
        x = layer(x, pos_emb, mask)

    return x * mask[:, :, None], encoder_lengths


def load_model(model_id, device, torch_dtype, use_cache):
    """Download the Shrutam-2 code + weights and assemble the ASR model."""
    # The Hub repo's modules import each other by bare name (`from smear import
    # ...`), so the snapshot directory has to be importable.
    code_dir = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.py", "*.yaml", "*.json", "*.txt"],
    )
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    ckpt_path = hf_hub_download(repo_id=model_id, filename="model.pt")
    encoder_path = hf_hub_download(repo_id=model_id, filename="encoder.pt")
    llm_path = os.path.join(snapshot_download(repo_id=model_id, allow_patterns="llm/*"), "llm")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from asr_model import ASRModel, EncoderDownsamplerCov1d, EncoderProjectorLinear
    from inference_config import inference_config, model_config
    from speech_encoder import SpeechEncoder

    # `SpeechEncoder` picks its own device (cuda:0 if visible); honour --device.
    encoder = SpeechEncoder(encoder_path)
    encoder.device = device
    encoder.preprocessor.to(device)
    encoder.model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    llm = AutoModelForCausalLM.from_pretrained(llm_path, dtype=torch_dtype, trust_remote_code=True)
    for param in llm.parameters():
        param.requires_grad = False
    llm.eval()

    # The checkpoint ships with `use_cache: false`, which makes autoregressive
    # decoding recompute the whole prefix at every step. Re-enabling the KV
    # cache does not change the generated tokens, only the RTFx we report.
    llm.config.use_cache = use_cache
    llm.generation_config.use_cache = use_cache

    model = ASRModel(
        encoder=encoder,
        llm=llm,
        encoder_projector=[
            EncoderProjectorLinear(model_config) for _ in range(model_config.num_experts)
        ],
        down_sampler=EncoderDownsamplerCov1d(model_config),
        tokenizer=tokenizer,
        train_config=inference_config,
        model_config=model_config,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    # `encoder` is a plain object rather than a submodule, so it keeps its own
    # float32 weights and device placement; only the projector/LLM move here.
    model.to(device=device, dtype=torch_dtype)
    model.eval()

    _patch_moe_masking(model)

    return model, tokenizer, encoder, model_config


def main(args):
    CONFIG_NAME = args.config_name  # None for single-config repos (e.g. VoiceArena/Monsoon_hi_test)
    SPLIT_NAME = args.split

    # Determine language for normalization: use --language if provided, otherwise
    # extract from config_name (e.g. "fleurs_hi") or, for single-config repos,
    # from the dataset name (e.g. "Monsoon_hi_test").
    if args.language:
        LANGUAGE = args.language
    else:
        source = CONFIG_NAME if CONFIG_NAME else os.path.basename(args.dataset)
        lang_match = re.search(r"_([a-z]{2})(?:_test)?$", source)
        LANGUAGE = lang_match.group(1) if lang_match else "hi"

    if LANGUAGE not in LANGUAGE_NAMES:
        raise ValueError(
            f"Shrutam-2 does not support language '{LANGUAGE}'. Supported: {sorted(LANGUAGE_NAMES)}"
        )

    device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    torch_dtype = getattr(torch, args.dtype)

    model, tokenizer, encoder, model_config = load_model(
        args.model_id, device, torch_dtype, use_cache=not args.disable_kv_cache
    )
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    prompt = args.prompt or f"Transcribe speech to {LANGUAGE_NAMES[LANGUAGE]} text."
    print(f"Prompt: {prompt!r}")
    prompt_ids = torch.tensor(
        tokenizer.encode(model_config.prompt_template.format(prompt)), dtype=torch.int64
    )
    prompt_len = prompt_ids.shape[0]
    pad_token_id = tokenizer.pad_token_id
    # `EncoderDownsamplerCov1d`'s second conv has kernel == stride == this rate,
    # so it maps `n` encoder frames to `n // ds_rate` projector frames.
    ds_rate = model_config.encoder_projector_ds_rate

    @torch.no_grad()
    def transcribe(audios):
        """Batched transcription. `audios` is a list of 1-D float arrays @ 16 kHz."""
        # Right-padded encoder features (valid frames first) plus the true
        # subsampled length of each sample.
        wavs = [torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1) for audio in audios]
        features, encoder_lengths = encode_batch(encoder, wavs)

        batch_size = len(wavs)
        max_frames = features.shape[1] // ds_rate
        frame_lengths = torch.div(encoder_lengths, ds_rate, rounding_mode="floor")
        frame_lengths = frame_lengths.clamp(min=1, max=max_frames).cpu()

        # Left-pad, since `generate` continues from the right-hand edge of the
        # sequence. Layout per sample:
        #   [pad] * (max_frames - n) + [audio placeholder] * n + [prompt]
        seq_len = max_frames + prompt_len
        input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.int64)
        attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        modality_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for i, n_frames in enumerate(frame_lengths.tolist()):
            start = max_frames - n_frames
            input_ids[i, start:max_frames] = -1  # placeholder ids for the audio span
            input_ids[i, max_frames:] = prompt_ids
            attention_mask[i, start:] = True
            modality_mask[i, start:max_frames] = True

        # The encoder output is right-padded (valid frames first), so the mask
        # handed to the SMEAR router is left-aligned, unlike `modality_mask`.
        model.MoELayer_routing.frame_mask = (
            torch.arange(max_frames)[None, :] < frame_lengths[:, None]
        ).to(device)

        output_ids = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            modality_mask=modality_mask.to(device),
            audio_rep=features.to(device=device, dtype=torch_dtype),
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )

        # Generating from `inputs_embeds` returns only the newly decoded tokens.
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def load_eval_dataset():
        dataset = load_dataset(
            args.dataset,
            CONFIG_NAME,
            split=SPLIT_NAME,
            streaming=args.streaming,
            token=True,
        )
        dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            if args.streaming:
                dataset = dataset.take(args.max_eval_samples)
            else:
                dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
        return dataset

    def benchmark(batch):
        audios = [audio["array"] for audio in batch["audio"]]
        minibatch_size = len(audios)
        batch["audio_length_s"] = [
            len(audio["array"]) / audio["sampling_rate"] for audio in batch["audio"]
        ]
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(
            batch, minibatch_size
        )

        # START TIMING
        on_cuda = device.type == "cuda"
        if on_cuda:
            torch.cuda.synchronize(device=device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        pred_text = transcribe(audios)

        # END TIMING
        if on_cuda:
            end_event.record()
            torch.cuda.synchronize(device=device)
            runtime = start_event.elapsed_time(end_event) / 1000.0
        else:
            runtime = time.perf_counter() - start_time

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        batch["predictions"] = pred_text  # raw; normalization applied at scoring time
        if "lattice" in batch:
            # Lattice reference (e.g. VoiceArena/Monsoon_hi_test): store the
            # lattice JSON-encoded in the reference field; scoring decodes it
            # and uses voi_oiwer (see normalizer/eval_utils.py).
            batch["references"] = [json.dumps(lat, ensure_ascii=False) for lat in batch["lattice"]]
        else:
            batch["references"] = batch["text"]  # raw; normalization applied at scoring time

        return batch

    print(f"Loading dataset: {args.dataset} with config: {CONFIG_NAME}")
    dataset = load_eval_dataset()

    if args.warmup_steps is not None and args.warmup_steps > 0:
        print(f"Running {args.warmup_steps} warmup steps...")
        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(
            warmup_dataset.map(
                benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"]
            )
        )
        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    # Reload the dataset for the timed run (resets the streaming pointer)
    dataset = load_eval_dataset()
    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"]
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
        "audio_filepath": [],
    }

    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    # Filter empty references (consistent with English pipeline)
    filtered = [
        (ref, pred, dur, time_s, fpath)
        for ref, pred, dur, time_s, fpath in zip(
            all_results["references"],
            all_results["predictions"],
            all_results["audio_length_s"],
            all_results["transcription_time_s"],
            all_results["audio_filepath"],
        )
        if data_utils.is_target_text_in_range(ref)
    ]
    if filtered:
        (
            all_results["references"],
            all_results["predictions"],
            all_results["audio_length_s"],
            all_results["transcription_time_s"],
            all_results["audio_filepath"],
        ) = zip(*filtered)
        all_results = {k: list(v) for k, v in all_results.items()}

    # Write manifest results (WER and RTFX)
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset,
        CONFIG_NAME or "",
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
        audio_filepaths=all_results["audio_filepath"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    from normalizer.eval_utils import OIWER_LANGUAGES, score_oiwer

    if LANGUAGE in OIWER_LANGUAGES:
        # Lattice-based, orthography-aware scoring (voi_oiwer applies its own
        # normalization internally).
        manifest = [
            {"text": ref, "pred_text": pred}
            for ref, pred in zip(all_results["references"], all_results["predictions"])
        ]
        wer, _ins, _del, _sub = score_oiwer(manifest, OIWER_LANGUAGES[LANGUAGE])
        wer = round(100 * wer, 2)
    else:
        norm_refs = [data_utils.ml_normalizer(r, lang=LANGUAGE) for r in all_results["references"]]
        norm_preds = [data_utils.ml_normalizer(p, lang=LANGUAGE) for p in all_results["predictions"]]
        wer_refs, wer_preds = normalize_compound_pairs(norm_refs, norm_preds)
        wer = wer_metric.compute(references=wer_refs, predictions=wer_preds)
        wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="bharatgenai/Shrutam-2",
        help="Model identifier. Should be a Shrutam-style repo on the Hub.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'VoiceArena/Monsoon_hi_test'`",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Config name for the dataset. *E.g.* `'fleurs_hi'` for Hindi FLEURS. "
        "Omit for single-config dataset repos (e.g. 'VoiceArena/Monsoon_hi_test').",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'hi'). If not provided, extracted from config_name.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override the transcription instruction. Defaults to "
        "'Transcribe speech to <Language> text.'",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'test'` for the test split.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream the dataset lazily over the network instead of downloading it in full before the evaluation. Off by default for reproducible benchmark timings.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for beam search (4 matches the reference inference script).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="The dtype to use for model loading and inference. E.g. 'float32', 'bfloat16'.",
    )
    parser.add_argument(
        "--disable_kv_cache",
        action="store_true",
        help="Honour the checkpoint's `use_cache: false`. Same transcriptions, much lower RTFx.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()

    main(args)
