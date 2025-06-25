import argparse
import os
import torch
import evaluate
from normalizer import data_utils
import time
from tqdm import tqdm
import julius
from moshi import models

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision("high")


def load_model(model_path):

    info = models.loaders.CheckpointInfo.from_hf_repo(model_path)

    mimi = info.get_mimi(device="cuda")
    tokenizer = info.get_text_tokenizer()
    lm = info.get_moshi(
        device="cuda",
        dtype=torch.bfloat16,
    )
    lm_gen = models.LMGen(lm, temp=0, temp_text=0.0)

    padding_token_id = info.raw_config.get("text_padding_token_id", 3)
    # Putting in some conservative defaults
    audio_silence_prefix_seconds = info.stt_config.get(
        "audio_silence_prefix_seconds", 1.0
    )
    audio_delay_seconds = info.stt_config.get("audio_delay_seconds", 5.0)

    return (
        mimi,
        tokenizer,
        lm,
        lm_gen,
        padding_token_id,
        audio_silence_prefix_seconds,
        audio_delay_seconds,
    )


@torch.inference_mode
def get_padded_batch(
    audios, sample_rates, before_padding: float, after_padding: float, frame_size: int
):
    sample_rate = 24_000

    batch = []
    max_len = -1

    for audio, sr in zip(audios, sample_rates):
        audio = julius.resample.resample_frac(audio, old_sr=sr, new_sr=sample_rate)
        audio = torch.nn.functional.pad(
            audio, (int(before_padding * sample_rate), int(after_padding * sample_rate))
        )
        max_len = max(max_len, audio.shape[-1])
        batch.append(audio)

    target = max_len
    if target % frame_size != 0:
        target = target + (frame_size - max_len % frame_size)

    batch = torch.stack(
        [
            torch.nn.functional.pad(audio, (0, target - audio.shape[-1]))
            for audio in batch
        ]
    )
    return batch


def main(args):
    (
        mimi,
        tokenizer,
        _lm,
        lm_gen,
        padding_token_id,
        audio_silence_prefix_seconds,
        audio_delay_seconds,
    ) = load_model(args.model_id)

    mimi_frame_size = mimi.frame_size

    def benchmark(batch):
        # Load audio inputs
        audios = [torch.from_numpy(audio["array"]) for audio in batch["audio"]]
        sample_rates = [ex["sampling_rate"] for ex in batch["audio"]]

        batch["audio_length_s"] = [
            len(audio) / batch["audio"][0]["sampling_rate"] for audio in audios
        ]
        minibatch_size = len(audios)

        # Start timing
        start_time = time.time()

        padded_batch = get_padded_batch(
            audios,
            sample_rates,
            before_padding=audio_silence_prefix_seconds,
            after_padding=audio_delay_seconds,
            frame_size=mimi_frame_size,
        )
        padded_batch = padded_batch.to(args.device).float()

        bsz = padded_batch.shape[0]

        text_tokens_acc = []

        with mimi.streaming(bsz), lm_gen.streaming(bsz):
            for offset in range(0, padded_batch.shape[-1], mimi.frame_size):
                audio_chunk = padded_batch[:, offset : offset + mimi.frame_size].cuda()
                tokens = mimi.encode(audio_chunk[:, None, :])
                text_tokens = lm_gen.step(tokens)
                text_tokens_acc.append(text_tokens)

        pred_tokens = torch.concat(text_tokens_acc, axis=-1).squeeze(dim=1)
        pred_tokens = torch.unbind(pred_tokens, dim=0)

        pred_text = [
            tokenizer.decode(t[t > padding_token_id].cpu().numpy().tolist())
            for t in pred_tokens
        ]

        # End timing
        runtime = time.time() - start_time

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # normalize transcriptions with English normalizer
        batch["predictions"] = [data_utils.normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]
        return batch

    if args.warmup_steps is not None:
        warmup_dataset = data_utils.load_data(args)
        warmup_dataset = data_utils.prepare_data(warmup_dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = warmup_dataset.take(num_warmup_samples)
        else:
            warmup_dataset = warmup_dataset.select(
                range(min(num_warmup_samples, len(warmup_dataset)))
            )
        warmup_dataset = iter(
            warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True)
        )

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    dataset = data_utils.prepare_data(dataset)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    dataset = dataset.map(
        benchmark,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=["audio"],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    # Write manifest results (WER and RTFX)
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
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(
        sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2
    )
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="esb/datasets",
        help="Dataset path. By default, it is `esb/datasets`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
        "can be found at `https://huggingface.co/datasets/esb/datasets`",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation`' for the dev split, or `'test'` for the test split.",
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
        default=1,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
