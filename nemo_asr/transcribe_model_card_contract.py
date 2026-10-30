from __future__ import annotations

import re
from typing import Any


MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
MODEL_REVISION = "e6ed81343993185d8c7617c6029c7378b4116a26"

DEFAULT_PROMPT = (
    "Audio transcription. Language: as spoken. Wording: verbatim, including "
    "fillers and repetitions. Formatting: punctuation and capitalization. "
    "Output: transcript only."
)

SPGISPEECH_PROMPT = (
    "Audio transcription. Language: as spoken. Wording: omit fillers and "
    "accidental repetitions; keep the remaining spoken wording unchanged. "
    "Formatting: punctuation and capitalization. Output: transcript only."
)

MONSOON_ENGLISH_PROMPT = (
    "Audio transcription. Language: as spoken. Wording: verbatim, including "
    "fillers and repetitions. Formatting: punctuation and capitalization. "
    "Output: transcript only. Use Latin script only."
)

MONSOON_HINDI_PROMPT = (
    "Audio transcription. Language: as spoken. Wording: verbatim, including "
    "fillers and repetitions. Formatting: punctuation and capitalization. "
    "Output: transcript only. Use Devanagari for Hindi words, never romanized "
    "Hindi. Keep English words in their original spelling."
)

FLEURS_PROMPT = (
    "Audio transcription. Language: as spoken, in its standard writing system. "
    "Wording: verbatim, including fillers and repetitions. Formatting: "
    "punctuation and capitalization; numbers in their standard written form. "
    "Output: transcript only."
)

PROMPTS = {
    "default": DEFAULT_PROMPT,
    "spgispeech": SPGISPEECH_PROMPT,
    "monsoon_en": MONSOON_ENGLISH_PROMPT,
    "monsoon_hi": MONSOON_HINDI_PROMPT,
    "fleurs": FLEURS_PROMPT,
}


def get_prompt(profile: str) -> str:
    try:
        return PROMPTS[profile]
    except KeyError as exc:
        valid = ", ".join(sorted(PROMPTS))
        raise ValueError(
            f"Unknown prompt profile {profile!r}; choose one of: {valid}"
        ) from exc


CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)


def cleanup_generation_loops(text: str) -> str:
    text = CHAR_RUN.sub(lambda match: match.group(1), text)
    return BLOCK_RUN.sub(lambda match: match.group(1), text)


def build_chat_completion_request(
    audio_data_url: str,
    prompt_profile: str,
    *,
    model: str = MODEL_ID,
) -> dict[str, Any]:
    if not audio_data_url.startswith("data:audio/wav;base64,"):
        raise ValueError("audio_data_url must be a base64-encoded WAV data URL")

    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": get_prompt(prompt_profile)},
                    {"type": "audio_url", "audio_url": {"url": audio_data_url}},
                ],
            }
        ],
        "stream": False,
        "n": 1,
        "max_tokens": 16384,
        "temperature": 0,
        "top_p": 1.0,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "seed": 0,
        },
    }
