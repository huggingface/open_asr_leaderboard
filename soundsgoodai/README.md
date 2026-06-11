# K2 Zipformer

This runner evaluates an offline K2 Zipformer model from a Hugging Face model
repo id.

The model package should contain:

- `model.pt`
- `bpe.model`
- `config.yaml` with the Zipformer architecture, feature, tokenizer,
  checkpoint file, and modified-beam-search decoding settings

Install the shared requirements, then the Zipformer-specific requirements, and
point `ICEFALL_PATH` at an Icefall checkout:

```bash
pip install -r ../requirements/requirements.txt
pip install -r ../requirements/requirements_zipformer.txt
git clone https://github.com/k2-fsa/icefall
```

Run the English short-form benchmark:

```bash
bash run_zipformer.sh
```
