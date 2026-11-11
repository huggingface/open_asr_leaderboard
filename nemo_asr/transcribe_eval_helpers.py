from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def decoded_audio_fields(audio: Any) -> tuple[Any, int]:
    try:
        samples = audio["array"]
        sampling_rate = audio["sampling_rate"]
    except (KeyError, TypeError, AttributeError) as exc:
        raise TypeError(
            "Expected decoded audio with ['array'] and ['sampling_rate'] fields"
        ) from exc
    return samples, int(sampling_rate)


def audio_chunk_ranges(
    sample_count: int,
    sampling_rate: int,
    chunk_duration_s: float = 55 * 60,
    chunk_overlap_s: float = 5,
) -> list[tuple[int, int]]:
    if sample_count < 1:
        raise ValueError("audio must contain at least one frame")
    if sampling_rate < 1:
        raise ValueError("sampling rate must be positive")
    if not math.isfinite(chunk_duration_s) or chunk_duration_s <= 0:
        raise ValueError("chunk duration must be positive and finite")
    if not math.isfinite(chunk_overlap_s) or chunk_overlap_s < 0:
        raise ValueError("chunk overlap must be non-negative and finite")

    chunk_samples = math.floor(chunk_duration_s * sampling_rate)
    overlap_samples = round(chunk_overlap_s * sampling_rate)
    if chunk_samples < 1:
        raise ValueError("chunk duration is shorter than one audio frame")
    if overlap_samples >= chunk_samples:
        raise ValueError("chunk overlap must be shorter than a chunk")
    if sample_count <= chunk_samples:
        return [(0, sample_count)]

    stride_samples = chunk_samples - overlap_samples
    return [
        (start, min(start + chunk_samples, sample_count))
        for start in range(0, sample_count, stride_samples)
    ]


def longform_chunk_plan(
    sample_count: int,
    sampling_rate: int,
    chunk_duration_s: float = 55 * 60,
    chunk_overlap_s: float = 5,
) -> tuple[list[tuple[int, int]], bool]:
    ranges = audio_chunk_ranges(
        sample_count,
        sampling_rate,
        chunk_duration_s,
        chunk_overlap_s,
    )
    return ranges, len(ranges) > 1


def _merge_transcript_boundary(
    left_text: str,
    right_text: str,
    search_words: int = 128,
) -> str:
    left = left_text.split()
    right = right_text.split()
    if not left:
        return right_text.strip()
    if not right:
        return left_text.strip()

    left_window = left[-search_words:]
    right_window = right[:search_words]
    lengths = [[0] * (len(right_window) + 1) for _ in range(len(left_window) + 1)]
    for left_index in range(1, len(left_window) + 1):
        for right_index in range(1, len(right_window) + 1):
            if left_window[left_index - 1] == right_window[right_index - 1]:
                lengths[left_index][right_index] = (
                    lengths[left_index - 1][right_index - 1] + 1
                )
            else:
                lengths[left_index][right_index] = max(
                    lengths[left_index - 1][right_index],
                    lengths[left_index][right_index - 1],
                )

    left_index = len(left_window)
    right_index = len(right_window)
    alignment: list[tuple[int, int]] = []
    while left_index and right_index:
        if left_window[left_index - 1] == right_window[right_index - 1]:
            alignment.append((left_index - 1, right_index - 1))
            left_index -= 1
            right_index -= 1
        elif (
            lengths[left_index - 1][right_index] >= lengths[left_index][right_index - 1]
        ):
            left_index -= 1
        else:
            right_index -= 1

    if not alignment:
        return f"{left_text.rstrip()}\n{right_text.lstrip()}"
    alignment.reverse()
    left_end = len(left) - len(left_window) + alignment[-1][0] + 1
    right_start = alignment[-1][1] + 1
    return " ".join(left[:left_end] + right[right_start:])


def aggregate_transcripts(parts: Sequence[str], chunked: bool) -> str:
    if chunked:
        if not parts:
            raise ValueError("chunked transcription must contain a response")
        merged = parts[0].strip()
        for part in parts[1:]:
            merged = _merge_transcript_boundary(merged, part)
        return merged
    if len(parts) != 1:
        raise ValueError("unchunked transcription must contain one response")
    return parts[0]
