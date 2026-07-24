#!/usr/bin/env python3
"""Evaluate Icefall Zipformer models for the ASR leaderboard."""

from __future__ import annotations

import argparse
import audioop
import math
import time
from pathlib import Path
from typing import Any

import evaluate
import kaldi_native_fbank as knf
import numpy as np
import sentencepiece as spm
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from tqdm import tqdm

from beam_search import modified_beam_search
from icefall.utils import AttributeDict
from train import get_model

from normalizer import data_utils

LOG_EPS = math.log(1e-10)

wer_metric = evaluate.load("wer")


class OfflineZipformerTransducer:
    """Decode 16 kHz audio with an offline Icefall Zipformer transducer."""

    def __init__(self, args: argparse.Namespace) -> None:

        self.model_dir = Path(
            snapshot_download(
                args.model_id, allow_patterns=("*.pt", "*.model", "*.yaml"),
            )
        ).resolve()

        model_config = OmegaConf.to_container(
            OmegaConf.load(self.model_dir / "config.yaml"),
        )

        self.min_encoder_input_frames = model_config["min_encoder_input_frames"]
        self.beam_size = model_config["decoding"]["beam_size"]

        bpe_model_path = (self.model_dir / model_config["tokenizer"]).resolve()
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(bpe_model_path))

        params = AttributeDict(model_config["model_params"])
        params.blank_id = self.sp.piece_to_id("<blk>")
        params.vocab_size = self.sp.get_piece_size() - 1
        self.params = params

        self.device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")

        model_path = (self.model_dir / model_config["file"]).resolve()
        self.model = get_model(self.params)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)["model"]

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model size: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B parameters")

        self.fbank_opts = knf.FbankOptions.from_dict(model_config["feature_opts"])

    def transcribe(self, audios: list, sampling_rate: int) -> list[str]:
        features = [
            self.pad_short_features(self.compute_features(audio)) for audio in audios
        ]
        feature_lengths = torch.tensor(
            [feature.size(0) for feature in features],
            dtype=torch.int64,
            device=self.device,
        )
        features = torch.nn.utils.rnn.pad_sequence(
            features,
            batch_first=True,
            padding_value=LOG_EPS,
        ).to(self.device)

        with (
            torch.no_grad(),
            torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16),
        ):
            encoder_out, encoder_out_lens = self.model.forward_encoder(
                features,
                feature_lengths,
            )
            hyp_tokens = self.decode(encoder_out, encoder_out_lens)

        return self.sp.decode(hyp_tokens)

    def pad_short_features(self, features):
        num_frames = features.size(0)
        if num_frames >= self.min_encoder_input_frames:
            return features

        pad = features.new_full(
            (self.min_encoder_input_frames - num_frames, self.params.feature_dim),
            LOG_EPS,
        )
        return torch.cat([features, pad], dim=0)

    def compute_features(self, audio) -> Any:

        audio = np.ascontiguousarray(audio)

        fbank = knf.OnlineFbank(self.fbank_opts)
        fbank.accept_waveform(self.fbank_opts.frame_opts.samp_freq, audio)
        fbank.input_finished()
        frames = [
            torch.from_numpy(fbank.get_frame(i)) for i in range(fbank.num_frames_ready)
        ]
        if not frames:
            return torch.empty(0, self.params.feature_dim, dtype=torch.float32)
        return torch.stack(frames).to(torch.float32)

    def decode(self, encoder_out, encoder_out_lens) -> list[list[int]]:
        return modified_beam_search(
            model=self.model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=self.beam_size,
        )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Hugging Face repo id or local model directory.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="hf-audio/esb-datasets-test-only-sorted",
        help="Hugging Face dataset path.",
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        default=True,
        help="Download/materialize the dataset instead of streaming it.",
    )
    parser.add_argument("--warmup_steps", type=int, default=1)
    return parser


def benchmark(
    batch: dict[str, list[Any]],
    backend: OfflineZipformerTransducer,
) -> dict[str, list[Any]]:
    audios = [
        resample_audioop(
            audio["array"],
            audio["sampling_rate"],
            backend.fbank_opts.frame_opts.samp_freq,
        )
        for audio in batch["audio"]
    ]
    minibatch_size = len(audios)

    start_time = time.time()
    pred_text = backend.transcribe(
        audios,
        sampling_rate=backend.fbank_opts.frame_opts.samp_freq,
    )
    runtime = time.time() - start_time

    if len(pred_text) != minibatch_size:
        raise RuntimeError(
            f"Zipformer returned {len(pred_text)} predictions for "
            f"{minibatch_size} audio samples."
        )

    batch["audio_length_s"] = [
        len(audio) / backend.fbank_opts.frame_opts.samp_freq for audio in audios
    ]
    batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
    batch["predictions"] = pred_text  # raw; normalization applied at scoring time
    batch["references"] = batch["original_text"]  # raw; normalization applied at scoring time
    return batch


def resample_audioop(
    audio: np.typing.NDArray[np.float32],
    sampling_rate: int,
    target_sampling_rate: int,
) -> np.typing.NDArray[np.float32]:

    audio = np.asarray(audio, dtype=np.float32)

    # Expected shape and dtype: (num_samples,), mono float32 audio in [-1.0, 1.0].
    if audio.ndim != 1:
        raise ValueError(f"Expected 1-D mono audio, got shape {audio.shape}.")
    max_signed_int = np.float32(32768.0)

    sampling_rate = int(sampling_rate)
    target_sampling_rate = int(target_sampling_rate)

    if sampling_rate == target_sampling_rate:
        return audio

    pcm16 = (np.clip(audio, -1.0, 1.0) * max_signed_int).astype(np.int16).tobytes()
    resampled_pcm16, _ = audioop.ratecv(
        pcm16,
        2,  # int16 sample width in bytes
        1,  # mono
        sampling_rate,
        target_sampling_rate,
        None,
    )
    resampled_audio = np.frombuffer(resampled_pcm16, dtype=np.int16).astype(np.float32)
    resampled_audio /= max_signed_int
    return resampled_audio


def limit_dataset(dataset, max_samples: int | None, streaming: bool):
    if max_samples is None or max_samples <= 0:
        return dataset
    if streaming:
        return dataset.take(max_samples)
    return dataset.select(range(min(max_samples, len(dataset))))


def main(args: argparse.Namespace) -> None:
    backend = OfflineZipformerTransducer(args)

    if args.warmup_steps is not None:
        warmup_dataset = data_utils.load_data(args)
        warmup_dataset = data_utils.prepare_data(warmup_dataset, sampling_rate=None)
        warmup_dataset = limit_dataset(
            warmup_dataset,
            args.warmup_steps * args.batch_size,
            args.streaming,
        )
        warmup_dataset = warmup_dataset.map(
            benchmark,
            batch_size=args.batch_size,
            batched=True,
            fn_kwargs={"backend": backend},
        )
        for _ in tqdm(iter(warmup_dataset), desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    dataset = limit_dataset(dataset, args.max_eval_samples, args.streaming)
    dataset = data_utils.prepare_data(dataset, sampling_rate=None)
    dataset = dataset.map(
        benchmark,
        batch_size=args.batch_size,
        batched=True,
        fn_kwargs={"backend": backend},
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    for result in tqdm(iter(dataset), desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", manifest_path)

    norm_refs = [data_utils.normalizer(r) for r in all_results["references"]]
    norm_preds = [data_utils.normalizer(p) for p in all_results["predictions"]]
    wer = wer_metric.compute(references=norm_refs, predictions=norm_preds)
    wer = round(100 * wer, 2)
    rtfx = round(
        sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]),
        2,
    )
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
