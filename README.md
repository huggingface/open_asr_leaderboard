# Open ASR Leaderboard

This repository contains the code for the Open ASR Leaderboard. The leaderboard is a Gradio Space that allows users to compare the accuracy of ASR models on a variety of datasets. The leaderboard is hosted at [open-asr-leaderboard/leaderboard](https://huggingface.co/spaces/open-asr-leaderboard/leaderboard).

# Requirements

Each library has its own set of requirements. We recommend using a clean conda environment, with Python 3.10 or above.

1) Clone this repository.
2) Install PyTorch by following the instructions here: https://pytorch.org/get-started/locally/
3) Install the common requirements for all library by running `pip install -r requirements/requirements.txt`.
4) Install the requirements for each library you wish to evalaute by running `pip install -r requirements/requirements_<library_name>.txt`.
5) Connect your Hugging Face account by running `huggingface-cli login`.

# Evaluate a model

Each library has a script `run_eval.py` that acts as the entry point for evaluating a model. The script is run by the corresponding bash script for each model that is being evalauted. The script then outputs a JSONL file containing the predictions of the model on each dataset, and summarizes the Word Error Rate of the model on each dataset after completion. 

1) Change directory into the library you wish to evaluate. For example, `cd transformers`.
2) Run the bash script for the model you wish to evaluate. For example, `bash run_wav2vec2.sh`.
3) **Note**: All evaluations are done on single GPU. If you wish to run two scripts in parallel, please use `CUDA_VISIBLE_DEVICES=<0,1,...N-1>` prior to running the bash script, where `N` is the number of GPUs on your machine.

# Add a new library

To add a new library for evalution in this benchmark, please follow the steps below:

1) Fork this repository and create a new branch.
2) Create a new directory for your library. For example, `mkdir transformers`.
3) Copy the `run_eval.py` script from an existing library into your new directory. For example, `cp transformers/run_eval.py <your_library>/run_eval.py`.
    - Modify the script as needed, but please try to keep the structure of the script the same as others.
    - In particular, the data loading, evaluation and manifest writing must be done in the same way as other libraries.
4) Create one bash file per model type following the convesion `run_<model_type>.sh`.
    - The bash script should follow the same steps as other libraries.
    - Different model sizes of the same type should share the script. For example `Wav2Vec` and `Wav2Vec2` would be two separate scripts, but different size of `Wav2Vec2` would be part of the same script.
5) (Optional) You could also add a `calc_rtf.py` script for your library to evaluate the Real Time Factor of the model.
6) Submit a PR for your changes.

# Add a new model

To add a new model for evalution in this benchmark, you can follow most of the steps noted above. 

Since the library already exists in the benchmark, we can simplify the steps to:

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