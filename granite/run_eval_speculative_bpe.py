"""
Cascade ASR BPE: Self-speculative decoding with dual-head CTC BPE encoder.
Based on v32 - replaces grapheme CTC draft with BPE CTC draft.

The full Granite Speech encoder already has grapheme out/out_mid heads; only out_llm
(BPE linear head) needs to be loaded separately from the dual-head encoder safetensors.

Single encoder pass: embeddings from the CTC draft step are reused for verify/fallback.
Importance for posterior_weighted_pool uses mid-layer (layer num_layers//2) grapheme
blank probability, captured via a forward hook on the encoder.
"""

import argparse
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
from normalizer import data_utils
import time
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, models
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

assert hasattr(models, "granite_speech")

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision('high')

LLM_DOWNSAMPLE_WINDOW = 4
LLM_OUT_DIM = 100353  # 100352 Granite BPE tokens + 1 CTC blank (label 0)


def posterior_weighted_pool(hidden, importance, window_size=4):
    """Importance-weighted downsampling. importance[b,t] = 1 - blank_prob."""
    B, T, D = hidden.shape
    pad_len = (window_size - T % window_size) % window_size
    if pad_len > 0:
        hidden = F.pad(hidden, (0, 0, 0, pad_len))
        importance = F.pad(importance, (0, pad_len))
    num_windows = hidden.shape[1] // window_size
    hidden = hidden.view(B, num_windows, window_size, D)
    importance = importance.view(B, num_windows, window_size)
    weights = importance / (importance.sum(dim=-1, keepdim=True) + 1e-8)
    return (hidden * weights.unsqueeze(-1)).sum(dim=2), window_size


def main(args):
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

    # Full Granite Speech model
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    logits_scaling = getattr(model.language_model.config, 'logits_scaling', 1.0)

    # BPE CTC head (loaded separately, plugs on top of model.encoder)
    out_llm = nn.Linear(model.encoder.config.hidden_dim, LLM_OUT_DIM, bias=True)
    llm_weights = load_file(hf_hub_download(repo_id=args.model_id, filename="out_llm.safetensors"))
    out_llm.load_state_dict(llm_weights)
    out_llm.to(torch.bfloat16).eval().to(device)

    # Forward hook to capture mid-layer hidden state for importance weighting.
    # The dual-head encoder applies grapheme feedback at layer num_layers//2; we hook
    # the output of that layer (before feedback addition) to compute blank probability.
    num_enc_layers = model.encoder.config.num_layers
    mid_layer_idx = num_enc_layers // 2 - 1  # 0-based index into model.encoder.layers
    _mid_hidden = {}

    def _save_mid_hidden(module, input, output):
        _mid_hidden['h'] = output[0] if isinstance(output, tuple) else output

    _hook = model.encoder.layers[mid_layer_idx].register_forward_hook(_save_mid_hidden)

    # ========== Chat Template Setup ==========
    text_instruction = "<|audio|>can you transcribe the speech into a written format?"

    # Build chat message and apply template
    message = [
        {"role": "user", "content": text_instruction},
    ]
    text_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    # Derive prefix and suffix from the formatted prompt
    prompt_prefix, prompt_suffix = text_prompt.split("<|audio|>")

    # Cache prompt embeddings (suffix excludes <|audio|> since we insert audio embeds separately)
    embed_layer = model.language_model.get_input_embeddings()
    prefix_ids = tokenizer.encode(prompt_prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(prompt_suffix, add_special_tokens=False)
    cached_prefix_embeds = embed_layer(torch.tensor([prefix_ids], device=device))
    cached_suffix_embeds = embed_layer(torch.tensor([suffix_ids], device=device))

    HOP_LENGTH = 160
    confidence_threshold = args.confidence_threshold
    ctc_threshold = args.ctc_threshold

    @torch.no_grad()
    def ctc_decode_bpe(audios):
        """BPE CTC draft: single encoder pass, reuse embeddings for verify/fallback."""
        texts = [text_prompt] * len(audios)
        model_inputs = processor(texts, audios, device=device, return_tensors="pt").to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            encoder_output = model.encoder(model_inputs["input_features"])
            # Full-resolution encoder hidden states — passed unchanged to model.projector
            # for LLM verification and fallback; never modified by posterior_weighted_pool.
            enc_hidden = encoder_output.last_hidden_state if hasattr(encoder_output, 'last_hidden_state') else encoder_output

            # Mid-layer hidden state captured by hook; compute grapheme logits for importance
            mid_h = _mid_hidden['h']
            mid_grapheme_logits = model.encoder.out(mid_h)
            grapheme_probs_mid = F.softmax(mid_grapheme_logits.float(), dim=-1)
            importance = 1.0 - grapheme_probs_mid[:, :, 0]  # 1 - blank_prob

            # Pool enc_hidden 4x for BPE CTC head only; enc_hidden itself is not subsampled
            x_pooled, _ = posterior_weighted_pool(enc_hidden, importance, window_size=LLM_DOWNSAMPLE_WINDOW)

            # Compute per-sample valid pooled lengths
            # After pooling, T_pooled = ceil((T_enc + pad) / window_size)
            # Valid frames for sample i: ceil(enc_len_i / window_size)
            pooled_lengths = []
            for i in range(len(audios)):
                enc_len = len(audios[i]) // HOP_LENGTH // 2 + 1
                enc_len = min(enc_len, enc_hidden.shape[1])
                pooled_len = math.ceil(enc_len / LLM_DOWNSAMPLE_WINDOW)
                pooled_lengths.append(min(pooled_len, x_pooled.shape[1]))

            # Gather non-padded positions and apply BPE head
            valid_positions = []
            for i, plen in enumerate(pooled_lengths):
                for t in range(plen):
                    valid_positions.append((i, t))

            batch_idx = torch.tensor([p[0] for p in valid_positions], device=device)
            time_idx = torch.tensor([p[1] for p in valid_positions], device=device)
            x_valid = x_pooled[batch_idx, time_idx, :]  # [N_valid, D]
            bpe_logits_valid = out_llm(x_valid)  # [N_valid, LLM_OUT_DIM]
            bpe_probs_valid = F.softmax(bpe_logits_valid.float(), dim=-1)  # [N_valid, V]

        # Decode each sample from its slice of valid probs
        bpe_texts, bpe_entropies, embed_lengths = [], [], []
        offset = 0
        for i in range(len(audios)):
            plen = pooled_lengths[i]
            probs_i = bpe_probs_valid[offset:offset + plen]  # [plen, V]
            offset += plen

            _, idx = probs_i.max(dim=-1)
            entropy_i = -(probs_i * torch.log(probs_i + 1e-10)).sum(dim=-1)

            dedup = torch.unique_consecutive(idx, dim=-1)
            non_blank = dedup[dedup > 0]
            token_ids = [t.item() - 1 for t in non_blank]  # label i -> Granite token (i-1)
            text = tokenizer.decode(token_ids) if token_ids else ""
            bpe_texts.append(text)
            bpe_entropies.append(entropy_i.max().item() if token_ids else float('inf'))
            embed_lengths.append(len(audios[i]) // HOP_LENGTH // 2 + 1)

        return bpe_texts, bpe_entropies, enc_hidden, embed_lengths

    @torch.no_grad()
    def verify(ctc_texts, embeddings, embed_lengths):
        """Verify BPE draft tokens with LLM. Identical to v32 except inputs are already BPE text."""
        batch_sz = len(ctc_texts)

        ctc_token_ids = []
        for text in ctc_texts:
            text = text.strip() if text else ""
            ctc_token_ids.append(tokenizer.encode(text, add_special_tokens=False) if text else [])

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            audio_embeds = model.projector(embeddings)
        max_proj_len = audio_embeds.shape[1]

        window_size, downsample_rate = model.config.window_size, model.config.downsample_rate
        num_queries = window_size // downsample_rate
        proj_lengths = [min(math.ceil(enc_len / window_size) * num_queries, max_proj_len) for enc_len in embed_lengths]

        if not any(ctc_token_ids):
            return [(False, ctc_texts[i]) for i in range(batch_sz)], audio_embeds, proj_lengths

        audio_token_id = model.config.audio_token_id
        all_input_ids, prompt_lens, audio_ranges = [], [], []

        for i, proj_len in enumerate(proj_lengths):
            audio_start = len(prefix_ids)
            audio_ranges.append((audio_start, audio_start + proj_len))
            prompt_part = prefix_ids + [audio_token_id] * proj_len + suffix_ids
            prompt_lens.append(len(prompt_part))
            all_input_ids.append(prompt_part + ctc_token_ids[i])

        max_len = max(len(ids) for ids in all_input_ids)
        padded_ids = torch.full((batch_sz, max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros(batch_sz, max_len, dtype=torch.long, device=device)
        for i, ids in enumerate(all_input_ids):
            padded_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attn_mask[i, :len(ids)] = 1

        inputs_embeds = model.language_model.get_input_embeddings()(padded_ids)
        for i in range(batch_sz):
            s, e = audio_ranges[i]
            inputs_embeds[i, s:e, :] = audio_embeds[i, :e-s, :]

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            hidden = model.language_model.model(attention_mask=attn_mask, inputs_embeds=inputs_embeds, use_cache=False).last_hidden_state

        # Gather hidden states at verification positions
        sample_idx, pos_idx, ctc_flat = [], [], []
        sample_ranges, sample_valid = [], []
        offset = 0

        for i in range(batch_sz):
            ctc_tokens = ctc_token_ids[i]
            if not ctc_tokens or prompt_lens[i] - 1 + len(ctc_tokens) > hidden.shape[1]:
                sample_ranges.append((offset, offset))
                sample_valid.append(False)
                continue
            verify_start = prompt_lens[i] - 1
            for k in range(len(ctc_tokens)):
                sample_idx.append(i)
                pos_idx.append(verify_start + k)
                ctc_flat.append(ctc_tokens[k])
            sample_ranges.append((offset, offset + len(ctc_tokens)))
            sample_valid.append(True)
            offset += len(ctc_tokens)

        if pos_idx:
            gathered = hidden[torch.tensor(sample_idx, device=device), torch.tensor(pos_idx, device=device), :]
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model.language_model.lm_head(gathered) / logits_scaling
            probs = F.softmax(logits.float(), dim=-1)
            ctc_probs = probs[torch.arange(len(ctc_flat), device=device), torch.tensor(ctc_flat, device=device)]

        results = []
        for i in range(batch_sz):
            s, e = sample_ranges[i]
            if not sample_valid[i]:
                results.append((False, ctc_texts[i]))
                continue
            token_probs = ctc_probs[s:e]
            accepted = (token_probs >= confidence_threshold).all().item()
            results.append((accepted, ctc_texts[i]))

        return results, audio_embeds, proj_lengths

    @torch.no_grad()
    def fallback(audio_embeds, indices, proj_lengths):
        """AR fallback for failed samples."""
        if not indices:
            return []

        batch_sz = len(indices)
        hidden_dim = audio_embeds.shape[-1]
        all_embeds, all_lengths = [], []

        for i in indices:
            sample_embeds = audio_embeds[i, :proj_lengths[i], :].unsqueeze(0)
            combined = torch.cat([cached_prefix_embeds, sample_embeds, cached_suffix_embeds], dim=1)
            all_embeds.append(combined.squeeze(0))
            all_lengths.append(combined.shape[1])

        max_len = max(all_lengths)
        padded = torch.zeros(batch_sz, max_len, hidden_dim, device=device, dtype=audio_embeds.dtype)
        attn_mask = torch.zeros(batch_sz, max_len, dtype=torch.long, device=device)
        for i, (emb, length) in enumerate(zip(all_embeds, all_lengths)):
            padded[i, max_len - length:] = emb
            attn_mask[i, max_len - length:] = 1

        outputs = model.language_model.generate(
            inputs_embeds=padded, attention_mask=attn_mask,
            bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams, early_stopping=True,
            do_sample=False, use_cache=True
        )

        return [tokenizer.decode(outputs[i], skip_special_tokens=True) for i in range(batch_sz)]

    def benchmark(batch):
        audios = [audio["array"] for audio in batch["audio"]]
        batch_sz = len(audios)

        start_time = time.time()

        # Step 1: BPE CTC draft (single encoder pass; embeddings reused below)
        bpe_texts, bpe_entropies, embeddings, embed_lengths = ctc_decode_bpe(audios)

        # Step 2: Gate by BPE CTC entropy
        predictions = [None] * batch_sz
        verify_idx = []

        for i, (text, ent) in enumerate(zip(bpe_texts, bpe_entropies)):
            if ent <= ctc_threshold and text.strip():
                predictions[i] = text.strip()
            else:
                verify_idx.append(i)

        # Step 3: Verify remaining with LLM (reuses encoder embeddings from step 1)
        if verify_idx:
            verify_emb = embeddings[verify_idx]
            verify_embed_lengths = [embed_lengths[i] for i in verify_idx]
            verify_bpe_texts = [bpe_texts[i] for i in verify_idx]

            results, audio_embeds, proj_lengths = verify(verify_bpe_texts, verify_emb, verify_embed_lengths)

            fail_idx = []
            for j, (accepted, text) in enumerate(results):
                i = verify_idx[j]
                if accepted:
                    predictions[i] = text.strip()
                else:
                    fail_idx.append(j)

            # Step 4: Fallback
            if fail_idx:
                fallback_texts = fallback(audio_embeds, fail_idx, proj_lengths)
                for k, j in enumerate(fail_idx):
                    predictions[verify_idx[j]] = fallback_texts[k]

        runtime = time.time() - start_time

        batch["transcription_time_s"] = [runtime / batch_sz] * batch_sz
        batch["predictions"] = [data_utils.normalizer(p) for p in predictions]
        batch["references"] = batch["norm_text"]
        return batch

    # Load and process dataset
    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling to {args.max_eval_samples} samples")
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset)

    dataset = dataset.map(benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"], desc="Processing")

    all_results = {"audio_length_s": [], "transcription_time_s": [], "predictions": [], "references": []}
    for result in tqdm(dataset, desc="Samples"):
        for key in all_results:
            all_results[key].append(result[key])

    _hook.remove()  # clean up forward hook

    # Write results
    manifest_path = data_utils.write_manifest(
        all_results["references"], all_results["predictions"], args.model_id,
        args.dataset_path, args.dataset, args.split,
        audio_length=all_results["audio_length_s"], transcription_time=all_results["transcription_time_s"]
    )
    print("Results saved at:", os.path.abspath(manifest_path))

    wer = round(100 * wer_metric.compute(references=all_results["references"], predictions=all_results["predictions"]), 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print(f"{args.model_id} - WER: {wer}%, RTFx: {rtfx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True,
                        help="Full Granite Speech model (provides encoder, projector, LLM)")
    parser.add_argument("--dataset_path", type=str, default="esb/datasets")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--confidence_threshold", type=float, default=0.2)
    parser.add_argument("--ctc_threshold", type=float, default=0.7)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    args = parser.parse_args()
    args.streaming = False
    main(args)
