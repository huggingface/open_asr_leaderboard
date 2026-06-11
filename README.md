# Open ASR Leaderboard

This repository contains the code for the Open ASR Leaderboard. The leaderboard is a Gradio Space that allows users to compare the accuracy of ASR models on a variety of datasets. The leaderboard is hosted at [hf-audio/open_asr_leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

# Datasets

The Open ASR Leaderboard evaluates models on a diverse set of publicly available ASR benchmarks hosted on the Hugging Face Hub. These datasets cover a wide range of domains, languages, and recording conditions to provide a fair and comprehensive comparison across models.

* **Main Test Sets (English, short-form):**
  The main benchmark datasets used for evaluation (short-form English) are available [here](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard).

* **English, long-form:**
  The [**ASR Longform benchmark**](https://huggingface.co/datasets/hf-audio/asr-leaderboard-longform) dataset includes earnings21 and earnings22. We also evaluate on [CORAAL](https://huggingface.co/datasets/bezzam/coraal), but it is stored as a separate dataset since it has multiple splits.

* **Multilingual Benchmark:**
  The [**ASR Multilingual benchmark**](https://huggingface.co/datasets/nithinraok/asr-leaderboard-datasets) dataset includes fleurs, mcv and mls multilingual.


* **Private datasets:** 
  After submitting a model to the leaderboard, the maintainers will evaluate on private sets, as described [here](https://huggingface.co/blog/open-asr-leaderboard-private-data).


# Evaluate a model (as of 10 June 2026)

English short-form evaluations use [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-overview) to guarantee reproducibility: every run executes the same Docker image on the same hardware, eliminating environment and driver differences. Multilingual and long-form evaluations will migrate to HF Jobs in the future. Local evaluation remains possible for contributors who want to test on their own hardware.

Jobs are launched on the following hardware ([flavor](https://huggingface.co/docs/hub/jobs-configuration#hardware-flavor) in HF Jobs terminology):
```
name             pretty name             cpu       ram      storage   accelerator               cost/min  cost/hour
h200             Nvidia H200             23 vCPU   256 GB   3000 GB   1x H200 (141 GB)          $0.0833   $5.00
```
Example costs for a full run over the main public datasets:
- $2.92 for `nvidia/parakeet-tdt-0.6b-v3`
- $2.68 for `openai/whisper-large-v3-turbo`
- $3.15 for `Qwen/Qwen3-ASR-1.7B`

Each model family has its own Docker image with the exact software stack required. Images are hosted on [HF Spaces](https://huggingface.co/collections/hf-audio/open-asr-leaderboard-eval-configurations):
- 🤗 [Transformers](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-transformers/tree/main): Whisper, Cohere, Voxtral, Voxtral Realtime, VibeVoice, Moonshine, Granite-Speech 3, GLM ASR, Crisper Whisper, Wav2Vec2, HuBERT, Data2Vec, MMS
- [NVIDIA NeMo](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-nemo/tree/main)
- [ESPnet](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-espnet/tree/main)
- [SpeechBrain](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-speechbrain/tree/main)
- [Granite 4](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-granite/tree/main)
- [Granite 4 NAR](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-granite-nar/tree/main)
- [Qwen3 ASR](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-qwen/tree/main)
- [Omnilingual ASR](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-omniasr/tree/main) by Meta
- [Lite-Whisper](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-lite-whisper/tree/main) by [Efficient Speech](https://huggingface.co/efficient-speech)
- [Phi4](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-phi4/tree/main) by Microsoft
- [Higgs Audio](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-boson/tree/main) by Boson AI
- [Kyutai](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-kyutai/tree/main)
- [Applied Brain Research](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-abr/tree/main)
- [API models](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-apis/tree/main)

**To launch an evaluation:**

1. **Hugging Face Hub setup**
   - Create an account at https://huggingface.co/ and add credits for HF Jobs: https://huggingface.co/settings/billing
   - Create a [WRITE token](https://huggingface.co/settings/tokens/new?tokenType=write) and copy it.
   - Create a Storage Bucket to store results: https://huggingface.co/new-bucket

2. **One-time local setup**
```bash
# Clone the repository
git clone git@github.com:huggingface/open_asr_leaderboard.git
cd open_asr_leaderboard

# Create a minimal conda environment (no GPU required locally)
conda create -n leaderboard_jobs python=3.10 -y
conda activate leaderboard_jobs
pip install -r requirements/requirements_jobs.txt
huggingface-cli login   # paste your WRITE token when prompted
```

3. **Launch an evaluation** 🚀
```bash
# Open the relevant submit_jobs script, uncomment the models/datasets you want, then run:
RESULTS_BUCKET="<your-bucket>" HF_TOKEN=hf_... bash qwen/submit_jobs.sh

# Jobs are submitted in parallel (one per dataset). The script waits for all
# jobs to finish, syncs results from the bucket, and prints a CSV summary.

# Billing to org
ORG_NAME="<org-name>" RESULTS_BUCKET="<your-bucket>" HF_TOKEN=hf_... bash qwen/submit_jobs.sh
```

## Local evaluation

For contributors who want to test locally or evaluate multilingual/long-form models before HF Jobs support is added, the `requirements/` folder contains per-family dependency files. The Dockerfiles in the HF Spaces can also be used to build a local container.

Each model family has a `run_eval.py` entry point driven by a corresponding bash script (e.g. `run_whisper.sh`). The script outputs a JSONL file with predictions and prints WER and RTFx after completion. See the sub-folders of this repo for examples; the latest scripts are in the HF Spaces linked above.

# Trade-off plots

For open-source models, you can plot tradeoff plots like below with `scripts/plot_all.sh`.

![EN Shortform RTFx vs WER](scripts/data/en_shortform_rtfx_wer.png)

You can highlight a particular model (see `scripts/data` for CSV results as of 26 March 2026):
```
./scripts/plot_all.sh --highlight "model_name"

# for example
./scripts/plot_all.sh --highlight "nvidia/parakeet-tdt-0.6b-v3"
```

![Highlight model](scripts/data/nvidia_parakeet-tdt-0.6b-v3_en_shortform_rtfx_wer.png)

You can also specify your own model and its performance as such:
```
./scripts/plot_all.sh --custom-model "MY MODEL" --model-size 2.0 --en-shortform-wer 5.5 --en-shortform-rtfx 1000
```

![Custom model](scripts/data/MY_MODEL_en_shortform_rtfx_wer.png)

# Contributing a model or dataset

Please follow the [pull request template](./.github/PULL_REQUEST_TEMPLATE.md) — it contains the full submission checklist and guidelines for both Transformers models and other libraries.

## Template `run_eval.py` script

<details>

<summary> Click to expand </summary>

```python
import argparse
import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate
from normalizer import data_utils
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
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # INFERENCE (FILL ME! Replacing 1-3 with steps from your library)
        # 1. Pre-processing
        inputs = processor(audios, sampling_rate=16_000, return_tensors="pt").to(args.device)
        inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
        # 2. Generation
        pred_ids = model.generate(**inputs)
        # 3. Post-processing
        pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # End timing
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0

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
        default="hf-audio/open-asr-leaderboard",
        help="Dataset path. By default, it is `hf-audio/open-asr-leaderboard`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
        "can be found at `https://huggingface.co/datasets/hf-audio/open-asr-leaderboard`",
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
        "--streaming",
        action="store_true",
        help="Stream the dataset lazily over the network instead of downloading it in full before the evaluation. Off by default for reproducible benchmark timings.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()

    main(args)

```

</details>

# Citation 


```bibtex
@misc{srivastav2025openasrleaderboardreproducible,
      title={Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation}, 
      author={Vaibhav Srivastav and Steven Zheng and Eric Bezzam and Eustache Le Bihan and Nithin Koluguri and Piotr Żelasko and Somshubra Majumdar and Adel Moumen and Sanchit Gandhi},
      year={2025},
      eprint={2510.06961},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.06961}, 
}
```
