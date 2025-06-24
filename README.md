
# Open ASR Leaderboard

This repository contains the code for the Open ASR Leaderboard. The leaderboard is a Gradio Space that allows users to compare the accuracy of ASR models on a variety of datasets. The leaderboard is hosted at [hf-audio/open_asr_leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

# Requirements

Each library has its own set of requirements. We recommend using a clean conda environment, with Python 3.10 or above.

1) Clone this repository.
2) Install PyTorch by following the instructions here: https://pytorch.org/get-started/locally/
3) Install the common requirements for all library by running `pip install -r requirements/requirements.txt`.
4) Install the requirements for each library you wish to evaluate by running `pip install -r requirements/requirements_<library_name>.txt`.
5) Connect your Hugging Face account by running `huggingface-cli login`.

**Note:** If you wish to run NeMo, the benchmark currently needs CUDA 12.6 to fix a problem in previous drivers for RNN-T inference with cooperative kernels inside conditional nodes (see here: https://github.com/NVIDIA/NeMo/pull/9869). Running `nvidia-smi` should output "CUDA Version: 12.6" or higher.

# Evaluate a model

Each library has a script `run_eval.py` that acts as the entry point for evaluating a model. The script is run by the corresponding bash script for each model that is being evaluated. The script then outputs a JSONL file containing the predictions of the model on each dataset, and summarizes the Word Error Rate (WER) and Inverse Real-Time Factor (RTFx) of the model on each dataset after completion.

To reproduce existing results:

1) Change directory into the library you wish to evaluate. For example, `cd transformers`.
2) Run the bash script for the model you wish to evaluate. For example, `bash run_wav2vec2.sh`.

**Note**: All evaluations were run using an NVIDIA A100-SXM4-80GB GPU, with NVIDIA driver 560.28.03, CUDA 12.6, and PyTorch 2.4.0. You should ensure you use the same configuration when submitting results. If you are unable to create an equivalent machine, please request one of the maintainers to run your scripts for evaluation! 

# Add a new library

To add a new library for evaluation in this benchmark, please follow the steps below:

1) Fork this repository and create a new branch
2) Create a new directory for your library. For example, `mkdir transformers`.
3) Copy the template `run_eval.py` script below into your new directory. The script should be updated for the new library by making two modifications. Otherwise, please try to keep the structure of the script the same as in the template. In particular, the data loading, evaluation and manifest writing must be done in the same way as other libraries for consistency.
   1) Update the model loading logic in the `main` function
   2) Update the inference logic in the `benchmark` function

<details>

<summary> Template script for Transformers: </summary>

```python
import argparse
import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate
from normalizer import data_utils
import time
from tqdm import tqdm

wer_metric = evaluate.load("wer")

def main(args):
    # Load model (FILL ME!)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(args.device)
    processor = WhisperProcessor.from_pretrained(args.model_id)

    def benchmark(batch):
        # Load audio inputs
        audios = [audio["array"] for audio in batch["audio"]]
        batch["audio_length_s"] = [len(audio) / batch["audio"][0]["sampling_rate"] for audio in audios]
        minibatch_size = len(audios)

        # Start timing
        start_time = time.time()

        # INFERENCE (FILL ME! Replacing 1-3 with steps from your library)
        # 1. Pre-processing
        inputs = processor(audios, sampling_rate=16_000, return_tensors="pt").to(args.device)
        inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
        # 2. Generation
        pred_ids = model.generate(**inputs)
        # 3. Post-processing
        pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)

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
            warmup_dataset = warmup_dataset.select(range(min(num_warmup_samples, len(warmup_dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True))

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

```

</details>

4) Create one bash file per model type following the conversion `run_<model_type>.sh`.
    - The bash script should follow the same steps as other libraries. You can copy the example for [run_whisper.sh](./transformers/run_whisper.sh) and update it to your library
    - Different model sizes of the same type should share the script. For example `Wav2Vec` and `Wav2Vec2` would be two separate scripts, but different size of `Wav2Vec2` would be part of the same script.
    - **Important:** for a given model, you can tune decoding hyper-parameters to maximize benchmark performance (e.g. batch size, beam size, etc.). However, you must use the **same decoding hyper-parameters** for each dataset in the benchmark. For more details, refer to the [ESB paper](https://arxiv.org/abs/2210.13352).
5) Submit a PR for your changes.

# Add a new model

To add a model from a new library for evaluation in this benchmark, you can follow the steps noted above.

To add a model from an existing library, we can simplify the steps to:

1) If the model is already supported, but of a different size, simply add the new model size to the list of models run by the corresponding bash script.
2) If the model is entirely new, create a new bash script based on others of that library and add the new model and its sizes to that script.
3) Run the evaluation script to obtain a list of predictions for the new model on each of the datasets.
4) Submit a PR for your changes.

# Citation 


```bibtex
@misc{open-asr-leaderboard,
	title        = {Open Automatic Speech Recognition Leaderboard},
	author       = {Srivastav, Vaibhav and Majumdar, Somshubra and Koluguri, Nithin and Moumen, Adel and Gandhi, Sanchit and Hugging Face Team and Nvidia NeMo Team and SpeechBrain Team},
	year         = 2023,
	publisher    = {Hugging Face},
	howpublished = "\\url{https://huggingface.co/spaces/huggingface.co/spaces/open-asr-leaderboard/leaderboard}"
}
```
