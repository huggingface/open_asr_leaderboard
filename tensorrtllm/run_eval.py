"""Run evaluation for TensorRT-LLM whisper models."""""
import argparse
import os
import time

import evaluate
import torch

from tqdm import tqdm

from normalizer import data_utils
from tensorrt_llm.runtime import ModelRunnerCpp
from tensorrt_llm.bindings import GptJsonConfig
from whisper_utils import log_mel_spectrogram, get_tokenizer
from pathlib import Path

wer_metric = evaluate.load("wer")
    
class WhisperTRTLLM(object):

    def __init__(self,
                 engine_dir,
                 assets_dir="assets",
                 batch_size=64):
        tokenizer_name = "multilingual"
        assert (Path(assets_dir) / "multilingual.tiktoken").exists(
        ), "multilingual.tiktoken file is not existed in assets_dir"

        self.tokenizer = get_tokenizer(name=tokenizer_name,
                                       num_languages=100,
                                       tokenizer_dir=assets_dir)
        self.eot_id = self.tokenizer.encode(
            "<|endoftext|>",
            allowed_special=self.tokenizer.special_tokens_set)[0]
        json_config = GptJsonConfig.parse_file(Path(engine_dir) / 'decoder' / 'config.json')
        assert json_config.model_config.supports_inflight_batching
        runner_kwargs = dict(engine_dir=engine_dir,
                                is_enc_dec=True,
                                max_batch_size=batch_size,
                                max_input_len=3000,
                                max_output_len=96,
                                max_beam_width=1,
                                debug_mode=False,
                                kv_cache_free_gpu_memory_fraction=0.9)
        self.model_runner_cpp = ModelRunnerCpp.from_dir(**runner_kwargs)
        self.n_mels = 128

    def process_batch(
            self,
            mel,
            mel_input_lengths,
            text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            num_beams=1,
            max_new_tokens=96):
        prompt_id = self.tokenizer.encode(
            text_prefix, allowed_special=self.tokenizer.special_tokens_set)
        prompt_id = torch.tensor(prompt_id)
        batch_size = len(mel)
        decoder_input_ids = prompt_id.repeat(batch_size, 1)

        with torch.no_grad():
            if isinstance(mel, list):
                mel = [
                    m.transpose(1, 2).type(
                        torch.float16).squeeze(0)
                    for m in mel
                ]
            else:
                mel = mel.transpose(1, 2)
            outputs = self.model_runner_cpp.generate(
                batch_input_ids=decoder_input_ids,
                encoder_input_features=mel,
                encoder_output_lengths=mel_input_lengths // 2,
                max_new_tokens=max_new_tokens,
                end_id=self.eot_id,
                pad_id=self.eot_id,
                num_beams=num_beams,
                output_sequence_lengths=True,
                return_dict=True)
            torch.cuda.synchronize()
            output_ids = outputs['output_ids'].cpu().numpy().tolist()
        texts = []
        for i in range(len(output_ids)):
            text = self.tokenizer.decode(output_ids[i][0]).strip()
            texts.append(text)
        return texts
    
def main(args) -> None:
    """Main function to run evaluation on a dataset."""
    asr_model = WhisperTRTLLM(args.model_id)

    def benchmark(batch):
        start_time = time.time()
        print(batch)
        
        features = [
            log_mel_spectrogram(wave,
                                asr_model.n_mels,
                                padding=3000,
                                device='cuda').unsqueeze(0)
            for wave in batch["audio"]["array"]
        ]

        # pad to the even number of features, for remove_30s_padding option, conv layer padding corner case
        # for i, feature in enumerate(features):
        #     if feature.shape[2] % 2:
        #         features[i] = torch.nn.functional.pad(feature, (0, 1))

        features_input_lengths = torch.tensor([f.shape[2] for f in features],
                                              dtype=torch.int32,
                                              device='cuda')
        print(features[0].shape)
        texts = asr_model.process_batch(features, features_input_lengths)
        print(texts,features_input_lengths, 23333333333)

        batch["transcription_time_s"] = time.time() - start_time
        batch["predictions"] = [data_utils.normalizer(pred) for pred in texts]
        batch["references"] = batch["norm_text"]
        return batch

    if args.warmup_steps is not None:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset)

        if args.streaming:
            warmup_dataset = dataset.take(args.warmup_steps)
        else:
            warmup_dataset = dataset.select(range(min(args.warmup_steps, len(dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, remove_columns=["audio"]))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset)

    dataset = dataset.map(benchmark, remove_columns=["audio"])

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
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier.",
    )
    parser.add_argument(
        '--dataset_path', type=str, default='esb/datasets', help='Dataset path. By default, it is `esb/datasets`'
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
            "can be found at `https://huggingface.co/datasets/esb/datasets`"
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
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest='streaming',
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
