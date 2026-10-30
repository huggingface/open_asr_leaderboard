"""Load Fluid Speech: NVIDIA Parakeet TDT 0.6B v3 with one small extra block between subsampling and the first
Conformer layer. The .nemo file holds that block's weights next to the usual ones, so a plain
`ASRModel.from_pretrained` / `restore_from` stops with "Unexpected key(s) ... encoder.fv_insert" instead of quietly
running without it; `load()` below adds the block and then loads every tensor strictly.

Clean mode is an optional transcription style for edited instead of verbatim text (the style of SPGISpeech's
references): it removes repeated words and filler words. Turn it on with
`load(..., clean_mode=True)`; without it the model is exactly the .nemo.

    import fluid_speech
    model = fluid_speech.load("<org>/<model>")                    # Hub repo id, or a local .nemo path
    model = fluid_speech.load("<org>/<model>", clean_mode=True)   # edited, non-verbatim text
"""
import os
import tarfile
import tempfile

import torch
import torch.nn as nn
from nemo.collections.asr.models import ASRModel

CLEAN_MODE_FILE = "clean-mode.safetensors"


class FrameCleaner(nn.Module):
    """Residual block on the subsampled frames: LayerNorm, projection, depthwise time convolution, GELU, projection."""

    def __init__(self, width=1024, dim=1024, kernel=15):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.down = nn.Linear(width, dim)
        self.mix = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.up = nn.Linear(dim, width)

    def forward(self, x):  # x: [batch, time, width]
        h = self.down(self.ln(x)).transpose(1, 2)
        return x + self.up(nn.functional.gelu(self.mix(h)).transpose(1, 2))


def attach(model, dim, kernel):
    """Add the block to `model.encoder` and run it on the output of the subsampling module."""
    encoder = model.encoder
    encoder.fv_insert = FrameCleaner(encoder.d_model, dim, kernel).to(next(encoder.parameters()).device)

    def clean_frames(module, args, out):
        if isinstance(out, tuple):
            return (encoder.fv_insert(out[0]),) + tuple(out[1:])
        return encoder.fv_insert(out)

    encoder.pre_encode.register_forward_hook(clean_frames)
    return encoder.fv_insert


def state_dict_of(nemo_path):
    with tarfile.open(nemo_path) as tar, tempfile.TemporaryDirectory() as tmp:
        member = next(m for m in tar.getmembers() if os.path.basename(m.name) == "model_weights.ckpt")
        tar.extract(member, tmp, filter="data")
        return torch.load(os.path.join(tmp, member.name), map_location="cpu", weights_only=True)


def resolve(model_id, filename=None):
    if os.path.exists(model_id):
        return os.path.join(os.path.dirname(os.path.abspath(model_id)), filename) if filename else model_id
    from huggingface_hub import hf_hub_download, list_repo_files
    if filename is None:
        nemo_files = [f for f in list_repo_files(model_id) if f.endswith(".nemo")]
        assert len(nemo_files) == 1, f"expected one .nemo file in {model_id}, found {nemo_files}"
        filename = nemo_files[0]
    return hf_hub_download(model_id, filename)


def fold_clean_mode(state, adapter_path):
    """Add scale * B @ A to every adapted encoder weight (the adapter's tensors are named `<module>.A` / `<module>.B`)."""
    from safetensors import safe_open

    with safe_open(adapter_path, framework="pt") as f:
        scale = float(f.metadata()["alpha"]) / int(f.metadata()["rank"])
        modules = sorted({k.rsplit(".", 1)[0] for k in f.keys()})
        for name in modules:
            key = f"encoder.{name}.weight"
            assert key in state, f"clean mode adapts {key}, which the model does not have"
            delta = scale * (f.get_tensor(f"{name}.B") @ f.get_tensor(f"{name}.A"))
            state[key] = state[key] + delta.view_as(state[key]).to(state[key].dtype)
    return state


def load(model_id, map_location=None, clean_mode=False):
    path = resolve(model_id)
    model = ASRModel.restore_from(path, map_location=map_location, strict=False)
    spec = model.cfg.fluid_speech
    attach(model, spec.insert_dim, spec.insert_kernel)
    state = state_dict_of(path)
    if clean_mode:
        state = fold_clean_mode(state, resolve(model_id, CLEAN_MODE_FILE))
    model.load_state_dict(state, strict=True)
    return model.eval()
