"""
Drop-in stand-in for the flash_attn package on systems where the real
flash-attn can't be installed (Windows native, torch+CUDA combos without
prebuilt wheels).

Implements `flash_attn_varlen_func` via per-segment torch SDPA. Only
correctness-equivalent for FULL attention (window_size=(-1,-1)). Sliding
window calls will raise so we fail loudly rather than silently wrong.

Inject before any `import flash_attn` happens:

    import sys, flash_attn_shim
    sys.modules['flash_attn'] = flash_attn_shim

For MiMo-V2.5-ASR specifically: the audio tokenizer's encoder + decoder
configs use [-1,-1] (full attention) per the shipped config.json. Only
the vocoder uses sliding window — but the vocoder is for TTS / speech
generation, never invoked during asr_sft, so this shim is safe for ASR.
"""
import torch
import torch.nn.functional as F


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    **kwargs,
):
    """SDPA-based stand-in. Inputs shaped (total_tokens, num_heads, head_dim)."""
    # Normalize: window_size may arrive as list/tuple from JSON configs.
    ws = tuple(window_size) if window_size is not None else (-1, -1)
    if ws != (-1, -1):
        raise NotImplementedError(
            f"flash_attn_shim: sliding window attention {window_size} not supported. "
            "This shim only handles full attention (encoder/decoder paths)."
        )
    if alibi_slopes is not None or softcap != 0.0:
        raise NotImplementedError(
            "flash_attn_shim: ALiBi slopes / softcap not supported."
        )

    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    n_segments = len(cu_q) - 1

    outputs = []
    for i in range(n_segments):
        sq, eq = cu_q[i], cu_q[i + 1]
        sk, ek = cu_k[i], cu_k[i + 1]
        if eq == sq:
            # Empty segment — emit empty tensor with matching shape.
            outputs.append(q.new_empty((0, q.shape[1], q.shape[2])))
            continue

        # SDPA wants (batch, heads, seq, dim); we have (seq, heads, dim).
        qi = q[sq:eq].transpose(0, 1).unsqueeze(0).contiguous()
        ki = k[sk:ek].transpose(0, 1).unsqueeze(0).contiguous()
        vi = v[sk:ek].transpose(0, 1).unsqueeze(0).contiguous()

        out = F.scaled_dot_product_attention(
            qi, ki, vi,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )  # (1, heads, seq, dim)
        outputs.append(out.squeeze(0).transpose(0, 1))  # back to (seq, heads, dim)

    if not outputs:
        return q.new_empty((0, q.shape[1], q.shape[2]))
    return torch.cat(outputs, dim=0)


# Optional helpers some flash_attn callers expect — provide stubs.
def flash_attn_func(*args, **kwargs):
    raise NotImplementedError("flash_attn_shim: flash_attn_func (non-varlen) not implemented yet.")


__version__ = "shim-0.1"
