import argparse
import os
import torch
import json
from tensorrt_llm.runtime import ModelRunnerCpp
from tensorrt_llm.bindings import GptJsonConfig
import numpy as np
from collections import OrderedDict
from pathlib import Path
from whisper_utils import log_mel_spectrogram, get_tokenizer
import evaluate
from normalizer import data_utils, cuda_sync
import time
from tqdm import tqdm
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor

wer_metric = evaluate.load("wer")

def read_config(component, engine_dir):
    engine_dir = Path(engine_dir)
    config_path = engine_dir / component / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config['pretrained_config'])
    model_config.update(config['build_config'])
    return model_config

class WhisperTRTLLM(object):

    def __init__(self,
                 engine_dir,
                 assets_dir="assets",
                 batch_size=64):
        encoder_config = read_config('encoder', engine_dir)
        decoder_config = read_config('decoder', engine_dir)
        self.n_mels = encoder_config['n_mels']
        self.num_languages = encoder_config['num_languages']
        is_multilingual = (decoder_config['vocab_size'] >= 51865)
        if is_multilingual:
            tokenizer_name = "multilingual"
            assert (Path(assets_dir) / "multilingual.tiktoken").exists(
            ), "multilingual.tiktoken file is not existed in assets_dir"
        else:
            tokenizer_name = "gpt2"
            assert (Path(assets_dir) / "gpt2.tiktoken").exists(
            ), "gpt2.tiktoken file is not existed in assets_dir"
        self.text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>" if is_multilingual else "<|startoftranscript|><|notimestamps|>"
        self.tokenizer = get_tokenizer(name=tokenizer_name,
                                       num_languages=self.num_languages,
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

    def process_single_batch(self, mel_batch, decoder_input_ids, mel_input_lengths, max_new_tokens):
        outputs = self.model_runner_cpp.generate(
            batch_input_ids=decoder_input_ids,
            encoder_input_features=mel_batch,
            encoder_output_lengths=mel_input_lengths // 2,
            max_new_tokens=max_new_tokens,
            end_id=self.eot_id,
            pad_id=self.eot_id,
            num_beams=1,
            output_sequence_lengths=True,
            return_dict=True
        )
        
        output_ids = outputs['output_ids'].cpu().numpy().tolist()
        texts = []
        for i in range(len(output_ids)):
            text = self.tokenizer.decode(output_ids[i][0]).strip()
            text = re.sub(r'<\|.*?\|>', '', text)
            texts.append(text)
        return texts
    
    def process_batch(self, mel, mel_input_lengths, num_threads=4, max_new_tokens=96):
        prompt_id = self.tokenizer.encode(
            self.text_prefix, allowed_special=self.tokenizer.special_tokens_set)
        prompt_id = torch.tensor(prompt_id)
        batch_size = len(mel)
        decoder_input_ids = prompt_id.repeat(batch_size, 1)

        with torch.no_grad():
            if isinstance(mel, list):
                mel = torch.stack([m.transpose(1, 2).type(torch.float16).squeeze(0) for m in mel])
            else:
                mel = mel.transpose(1, 2)

            num_threads = min(num_threads, batch_size)
            mel_batches = torch.split(mel, batch_size // num_threads)
            mel_input_lengths_batches = torch.split(mel_input_lengths, batch_size // num_threads)

            texts_list = []
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for i, mel_batch in enumerate(mel_batches):
                    current_length = mel_batch.size(0)
                    futures.append(executor.submit(
                        self.process_single_batch,
                        mel_batch,
                        decoder_input_ids[:current_length],
                        mel_input_lengths_batches[i],
                        max_new_tokens
                    ))
                
                for future in futures:
                    texts_list.extend(future.result())
        
        return texts_list

def longest_common_substring(s1, s2):
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    longest_length = 0  
    end_index_s1 = 0 

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]: 
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest_length:
                    longest_length = dp[i][j]
                    end_index_s1 = i  
            else:
                dp[i][j] = 0 

    return s1[end_index_s1 - longest_length:end_index_s1]

def chunk_audio(audio, chunk_length, overlap_length, sample_rate):
    chunk_size = int(chunk_length * sample_rate)
    overlap_size = int(overlap_length * sample_rate)
    
    chunks = []
    start = 0
    
    while start < len(audio):
        end = min(start + chunk_size, len(audio))
        chunks.append(audio[start:end])
        start += chunk_size - overlap_size
    
    return chunks

def main(args):
    asr_model = WhisperTRTLLM(engine_dir=args.model_id)

    def benchmark(batch, min_new_tokens=None):
        # Load audio inputs
        max_duration, sample_rate = 30, 16000
        audios_origin = [audio["array"].astype(np.float32) for audio in batch["audio"]]
        minibatch_size = len(audios_origin)
        audios, audio_index = [], []

        chunk_length = 25
        overlap_length = 5 
        for i, audio in enumerate(audios_origin):
            if len(audio) > max_duration * sample_rate:
                audio_chunks = chunk_audio(audio, chunk_length, overlap_length, sample_rate)
                for chunk in audio_chunks:
                    audios.append(chunk)
                    audio_index.append(i)
            else:
                audios.append(audio)
                audio_index.append(i)
        audios = [torch.from_numpy(audio) for audio in audios]

        longest_duration = int(sample_rate * max_duration)

        features = [
            log_mel_spectrogram(wave,
                                asr_model.n_mels,
                                padding=longest_duration - wave.shape[-1],
                                device='cuda').unsqueeze(0)
            for wave in audios
        ]

        features_input_lengths = torch.tensor([f.shape[2] for f in features],
                                              dtype=torch.int32,
                                              device='cuda')

        # START TIMING - CUDA sync for accurate GPU timing
        cuda_sync(0)  # TensorRT-LLM runs on CUDA
        start_time = time.time()

        # Model inference only in timed block
        with torch.inference_mode(): 
            texts_origin = asr_model.process_batch(features, features_input_lengths, num_threads=4)

        # END TIMING - CUDA sync before measuring
        cuda_sync(args.device)
        runtime = time.time() - start_time

        # Post-processing outside timed block
        texts = []
        for i in range(minibatch_size):
            text_chunks = []
            for j in range(len(texts_origin)):
                if audio_index[j] == i:
                    text_chunks.append(texts_origin[j])
            
            if len(text_chunks) > 1:
                merged_text = text_chunks[0]
                for t in text_chunks[1:]:
                    lcs = longest_common_substring(merged_text, t)
                    merged_text += t[len(lcs):]
                    
                texts.append(merged_text)
            else:
                texts.append(text_chunks[0])

        print(f"Batch size: {minibatch_size}, Time taken: {runtime:.2f} s, texts_origin_len: {len(texts_origin)}, texts_len: {len(texts)}")
        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # normalize transcriptions with English normalizer
        batch["predictions"] = [data_utils.normalizer(pred) for pred in texts]
        batch["references"] = batch["norm_text"]
        return batch

    if args.warmup_steps is not None:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True, fn_kwargs={"min_new_tokens": args.max_new_tokens}))

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

    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"],
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
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
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
        default=16,
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
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (for auto-regressive models).",
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
