import torch
import torchaudio
import numpy as np
import librosa
import math
from pathlib import Path
from typing import Union, List, Optional
from dataclasses import dataclass
from torch.nn import functional as F
from torch import nn, Tensor
from transformers import (
    WhisperModel, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig,
    AutoConfig, WhisperFeatureExtractor, AutoTokenizer
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchaudio.functional import forced_align, merge_tokens

class ModelConfig(PretrainedConfig):
    model_type = "whisper_qwen"

    def __init__(self, whisper_model_name=None, qwen_model_name=None, **kwargs):
        super().__init__(**kwargs)
        self.whisper_model_name = whisper_model_name
        self.qwen_model_name = qwen_model_name
        self.freeze_audio_encoder = False

@dataclass
class TranscriptBatch:
    text: List[List[str]]
    start: Optional[List[List[float]]] = None
    end: Optional[List[List[float]]] = None
    tokens: Optional[List[List[int]]] = None

def _set_trainable(module, train: bool):
    for p in module.parameters():
        p.requires_grad = train

class WhisperQwenModel(PreTrainedModel):
    config_class = ModelConfig

    def __init__(self, torch_dtype=torch.bfloat16):
        config = ModelConfig(
            whisper_model_name='openai/whisper-large-v3',
            qwen_model_name='babs/decoder_fusion_lm_v5'
        )
        super().__init__(config)

        whisper = WhisperModel.from_pretrained(config.whisper_model_name, torch_dtype=torch_dtype)
        self.whisper_encoder = whisper.encoder
        _set_trainable(self.whisper_encoder, False)

        qwen_config = AutoConfig.from_pretrained(config.qwen_model_name)
        self.qwen = None

        self.audio_proj = nn.Linear(whisper.config.d_model, qwen_config.hidden_size)
        self.qwen_bos_id = 151650
        self.qwen_eos_id = qwen_config.eos_token_id
        self.blank_id = self.qwen_bos_id

        self.encoder_norm = nn.RMSNorm(qwen_config.hidden_size)
        self.ctc_head = nn.Linear(qwen_config.hidden_size, qwen_config.vocab_size, bias=False)
        self.encoder_pool = nn.Conv1d(
            in_channels=self.whisper_encoder.config.d_model,
            out_channels=self.whisper_encoder.config.d_model,
            kernel_size=2, stride=2, groups=2, bias=False
        ).to(torch_dtype)

        del whisper
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3')
        self.tokenizer = AutoTokenizer.from_pretrained('babs/decoder_fusion_lm_v3')
        self.torch_dtype = torch_dtype
        self.load()

    def _downsample(self, x: Tensor):
        return self.encoder_pool(x.transpose(1, 2)).transpose(1, 2)

    def _encode_audio(self, mels: Tensor) -> Tensor:
        mels = mels.to(dtype=self.torch_dtype)
        h = self.whisper_encoder(mels).last_hidden_state
        h = self._downsample(h)
        h = self.encoder_norm(self.audio_proj(h))
        return h

    def prepare_for_generation(self, audio: Union[np.ndarray, List[np.ndarray]],
                             chunk_sec=30, sample_rate=16000):
        if isinstance(audio, np.ndarray):
            audio = [audio]

        features = []
        self._enc_lens = []
        chunk_len = int(chunk_sec * sample_rate)

        for wav in audio:
            if wav.ndim == 2:
                wav = wav.mean(axis=0)

            for start in range(0, len(wav), chunk_len):
                chunk = wav[start:start + chunk_len]
                num_samples = len(chunk)
                self._enc_lens.append(math.ceil(num_samples / 640))
                mel = self.feature_extractor(
                    chunk, sampling_rate=sample_rate, return_tensors="pt"
                ).input_features
                features.append(mel)

        return torch.cat(features, dim=0)

    def _build_prompt(self):
        PROMPT_TOKEN = "<|quad_start|>"
        inputs = f'{PROMPT_TOKEN}\nSpecial_words-><>\nTranscription:\n'
        inputs = self.tokenizer(inputs, return_tensors='pt')
        return inputs['input_ids'], inputs['attention_mask']

    def load_audio(self, audio_input: Union[str, bytes, np.ndarray], target_sr=16000):
        if isinstance(audio_input, str):
            array, sr = librosa.load(audio_input, sr=target_sr, duration=30)
        elif isinstance(audio_input, bytes):
            import io
            audio_io = io.BytesIO(audio_input)
            array, sr = librosa.load(audio_io, sr=target_sr, duration=30)
        elif isinstance(audio_input, np.ndarray):
            array = audio_input[:(16000*30)]
        else:
            raise ValueError("Audio input must be file path, bytes, or numpy array")

        return array

    @torch.no_grad()
    def generate(self, audio: Union[str, bytes, np.ndarray], max_new_tokens=50,
                temperature=0.1, return_timestamps=False):
        audio_array = self.load_audio(audio)
        audio_tensor = self.prepare_for_generation(audio_array)
        audio_tensor = audio_tensor.to(self.torch_device)

        input_ids, attention_mask = self._build_prompt()
        input_ids = input_ids.to(self.torch_device)
        attention_mask = attention_mask.to(self.torch_device)

        batch_size = input_ids.size(0)
        device = input_ids.device

        pad_token_id = self.qwen.config.pad_token_id or self.qwen_eos_id
        eos_token_id = self.qwen_eos_id

        audio_embd = self._encode_audio(audio_tensor)
        audio_len = audio_embd.size(1)

        text_embd = self.qwen.model.embed_tokens(input_ids)
        combined_embs = torch.cat([audio_embd, text_embd], dim=1)

        dummy_ids = torch.full((batch_size, combined_embs.size(1)), pad_token_id, device=device)
        dummy_ids[:, audio_len:] = input_ids

        audio_attention_mask = torch.ones((batch_size, audio_len), device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([audio_attention_mask, attention_mask], dim=1)

        out = self.qwen.generate(
            input_ids=dummy_ids,
            inputs_embeds=combined_embs,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            #temperature=temperature,
            max_new_tokens=max_new_tokens
        )

        new_tokens = out[:, combined_embs.size(1):]
        seqs = []
        for row in new_tokens:
            if eos_token_id in row:
                cut = (row == eos_token_id).nonzero(as_tuple=True)[0][0] + 1
                seqs.append(row[:cut])
            else:
                seqs.append(row)

        if return_timestamps:
            logp = F.log_softmax(self.ctc_head(audio_embd), dim=-1)
            transcript = TranscriptBatch(text=[], start=[], end=[])

            for i, (chunk_ids, enc_len) in enumerate(zip(seqs, self._enc_lens)):
                path = forced_align(
                    logp[i:i+1, :enc_len, :].cpu(),
                    chunk_ids[:-1].unsqueeze(0).cpu(),
                    blank=self.blank_id,
                )
                ts = self.get_timestamps(path)
                transcript.text.append(ts.text[0])
                transcript.start.append(ts.start[0])
                transcript.end.append(ts.end[0])

            return transcript

        decoded = [self.tokenizer.decode(s, skip_special_tokens=True).strip() for s in seqs]
        text_lists = [[clean_asr_artifacts(t)] for t in decoded]

        return TranscriptBatch(text=text_lists, start=None, end=None)

    def get_timestamps(self, alignment, level="sentence"):
        batch_text, batch_start, batch_end = [], [], []
        aligned_tokens, scores = alignment

        for path, sc in zip(aligned_tokens, scores):
            span = merge_tokens(path, sc, blank=self.blank_id)
            starts = [s.start * 0.040 for s in span]
            ends = [(s.end + 1) * 0.040 for s in span]
            toks = [s.token for s in span]

            words, w_start, w_end = self.merge_tokens_to_words(toks, starts, ends)
            if level == "sentence":
                units, u_start, u_end = self.merge_words_to_sentences(words, w_start, w_end)
            else:
                units, u_start, u_end = words, w_start, w_end

            batch_text.append(units)
            batch_start.append(u_start)
            batch_end.append(u_end)

        return TranscriptBatch(text=batch_text, start=batch_start, end=batch_end)

    def merge_tokens_to_words(self, token_ids, starts, ends, space_prefix=("Ġ", "▁")):
        toks = self.tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
        words, w_start, w_end = [], [], []
        buf, buf_start = [], None

        for i, tok in enumerate(toks):
            new_word = tok.startswith(space_prefix) or buf == []

            if new_word and buf:
                words.append(self.tokenizer.convert_tokens_to_string(buf).strip())
                w_start.append(buf_start)
                w_end.append(ends[i - 1])
                buf, buf_start = [], None

            if new_word:
                buf_start = starts[i]

            buf.append(tok)

        if buf:
            words.append(self.tokenizer.convert_tokens_to_string(buf).strip())
            w_start.append(buf_start)
            w_end.append(ends[-1])

        return words, w_start, w_end

    def merge_words_to_sentences(self, words, w_start, w_end):
        PUNCT = {".", "?", "!"}
        sents, s_start, s_end = [], [], []
        buf, buf_start = [], None

        for w, st, en in zip(words, w_start, w_end):
            if not buf:
                buf_start = st

            if w in PUNCT and buf:
                buf[-1] += w
            else:
                buf.append(w)

            if w.endswith(tuple(PUNCT)) or w in PUNCT:
                sents.append(" ".join(buf))
                s_start.append(buf_start)
                s_end.append(en)
                buf = []

        if buf:
            sents.append(" ".join(buf))
            s_start.append(buf_start)
            s_end.append(w_end[-1])

        return sents, s_start, s_end

    def _split_and_load(self, enc_state, device: torch.device):
        def _sub(prefix):
            return {k[len(prefix):]: v for k, v in enc_state.items() if k.startswith(prefix)}

        self.whisper_encoder.load_state_dict(_sub("whisper_encoder."))
        self.audio_proj.load_state_dict(_sub("audio_proj."))
        self.encoder_norm.load_state_dict(_sub("encoder_norm."))
        self.encoder_pool.load_state_dict(_sub("encoder_pool."))
        self.ctc_head.load_state_dict(_sub("ctc_head."))

        for m in (self.whisper_encoder, self.audio_proj, self.encoder_norm, self.encoder_pool):
            m.to(device=device, dtype=self.torch_dtype)

    def load(self, model_dir='babs/fusion_lm_v5', device="cuda"):
        self.torch_device = torch.device(device)

        self.qwen = AutoModelForCausalLM.from_pretrained(
            "babs/decoder_fusion_lm_v5",
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            attn_implementation = 'flash_attention_2'
        ).to(self.torch_device)

        path = hf_hub_download(filename='encoder.safetensors', repo_id=model_dir)
        enc_state = load_file(path)
        self._split_and_load(enc_state, self.torch_device)
        return self

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'float32', 'bfloat16'])
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--timestamps', action='store_true')

    args = parser.parse_args()

    dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16}

    model = WhisperQwenModel(torch_dtype=dtype_map[args.dtype])
    model = model.to(args.device, dtype=dtype_map[args.dtype])
    model.eval()

    result = model.generate(
        audio=args.audio,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        return_timestamps=False
    )

    print(result.text[0][0])

if __name__ == "__main__":
    main()
