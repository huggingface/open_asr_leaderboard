"""
Self-speculative decoding for Speech LLMs.
"""

import argparse
import math
import os
import torch
import torch.nn.functional as F
import evaluate
from normalizer import data_utils
import time
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, models

assert hasattr(models, "granite_speech")

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision('high')


def main(args):
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    logits_scaling = getattr(model.language_model.config, 'logits_scaling', 1.0)

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
    def ctc_decode(audios):
        """CTC decode with entropy-based confidence."""
        texts = [text_prompt] * len(audios)
        model_inputs = processor(texts, audios, device=device, return_tensors="pt").to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            encoder_output = model.encoder(model_inputs["input_features"])
            embeddings = encoder_output.last_hidden_state if hasattr(encoder_output, 'last_hidden_state') else encoder_output
            ctc_logits = model.encoder.out(embeddings)
            ctc_probs = F.softmax(ctc_logits.float(), dim=-1)

        _, idx_batch = ctc_probs.max(dim=-1)
        entropy = -(ctc_probs * torch.log(ctc_probs + 1e-10)).sum(dim=-1)

        ctc_texts, ctc_entropies, embed_lengths = [], [], []
        for i, idx in enumerate(idx_batch):
            dedup = torch.unique_consecutive(idx, dim=-1)
            non_blank = dedup[dedup > 0].tolist()
            ctc_texts.append(''.join(chr(c) for c in non_blank))
            ctc_entropies.append(entropy[i].max().item() if non_blank else float('inf'))
            embed_lengths.append(len(audios[i]) // HOP_LENGTH // 2 + 1)

        return ctc_texts, ctc_entropies, embeddings, embed_lengths

    @torch.no_grad()
    def verify(ctc_texts, embeddings, embed_lengths):
        """Verify CTC outputs with LLM."""
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
            num_beams=args.num_beams, early_stopping=args.num_beams > 1,
            do_sample=False, use_cache=True
        )

        return [tokenizer.decode(outputs[i], skip_special_tokens=True) for i in range(batch_sz)]

    def benchmark(batch):
        audios = [audio["array"] for audio in batch["audio"]]
        batch_sz = len(audios)

        start_time = time.time()

        # Step 1: CTC decode
        ctc_texts, ctc_entropies, embeddings, embed_lengths = ctc_decode(audios)

        # Step 2: Gate by CTC entropy
        predictions = [None] * batch_sz
        verify_idx = []

        for i, (text, ent) in enumerate(zip(ctc_texts, ctc_entropies)):
            if ent <= ctc_threshold and text.strip():
                predictions[i] = text.strip()
            else:
                verify_idx.append(i)

        # Step 3: Verify remaining
        if verify_idx:
            verify_emb = embeddings[verify_idx]
            verify_lens = [embed_lengths[i] for i in verify_idx]
            verify_texts = [ctc_texts[i] for i in verify_idx]

            results, audio_embeds, proj_lengths = verify(verify_texts, verify_emb, verify_lens)

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

    # Write results
    manifest_path = data_utils.write_manifest(
        all_results["references"], all_results["predictions"], args.model_id,
        args.dataset_path, args.dataset, args.split,
        audio_length=all_results["audio_length_s"], transcription_time=all_results["transcription_time_s"]
    )
    print("Results saved at:", os.path.abspath(manifest_path))

    wer = round(100 * wer_metric.compute(references=all_results["references"], predictions=all_results["predictions"]), 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print(f"WER: {wer}%, RTFx: {rtfx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="esb/datasets")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--confidence_threshold", type=float, default=0.01)
    parser.add_argument("--ctc_threshold", type=float, default=0.5)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    args = parser.parse_args()
    args.streaming = False
    main(args)
