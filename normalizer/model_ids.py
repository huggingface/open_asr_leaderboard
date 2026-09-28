"""Recover a model's Hub id from the name the results bucket uses.

A manifest filename cannot carry the ``/`` of a Hub id, so the eval scripts write
``model_id.replace("/", "-")`` (see :func:`normalizer.eval_utils`), and the bucket
directory is named the same way. That transform is lossy: once the separator is a
hyphen, nothing in ``abr-ai-niagara-19m-batch.en`` says whether the org is ``abr``
or ``abr-ai``. The tables below carry the information the filename dropped.

    >>> to_hub_id("microsoft-Phi-4-multimodal-instruct")
    'microsoft/Phi-4-multimodal-instruct'
    >>> to_hub_id("ibm-granite-granite-speech-3.3-8b")
    'ibm-granite/granite-speech-3.3-8b'
"""

from __future__ import annotations

import collections

__all__ = ["HYPHENATED_ORGS", "BUCKET_ID_OVERRIDES", "to_hub_id", "to_hub_ids"]

# Orgs whose own name contains a hyphen. Splitting on the first hyphen would put
# half the org into the model name, so these are matched as a whole prefix.
# Only orgs that actually appear in the results bucket are listed; add to this
# set when onboarding a model from a new hyphenated org.
HYPHENATED_ORGS = frozenset(
    {
        "ARTPARK-IISc",
        "AutoArk-AI",
        "OpenMOSS-Team",
        "abr-ai",
        "distil-whisper",
        "efficient-speech",
        "ibm-granite",
        "zai-org",
    }
)

# Manifests whose name was never a Hub id to begin with, so no rule recovers it.
# The eval scripts for these models take a bare or framework-specific model card
# on the command line: `omniasr/run_eval_ml.py` maps
# "facebook/omniASR-LLM-7B" to the "omniASR_LLM_7B" card NeMo/omnilingual want,
# and `nemo_asr/run_parakeet.sh` passes NVIDIA's names without their org.
BUCKET_ID_OVERRIDES = {
    "omniASR_CTC_7B_v2": "facebook/omniASR-CTC-7B-v2",
    "omniASR_LLM_7B_v2": "facebook/omniASR-LLM-7B-v2",
    "stt_en_conformer_transducer_small": "nvidia/stt_en_conformer_transducer_small",
}


def to_hub_id(bucket_id: str) -> str:
    """The Hub id ``bucket_id`` was written from, e.g. ``microsoft/Phi-4-multimodal-instruct``.

    Returns ``bucket_id`` unchanged when it carries no separator to restore and no
    override claims it, so an unrecognised name passes through rather than being
    mangled into a plausible-looking wrong id.
    """
    if bucket_id in BUCKET_ID_OVERRIDES:
        return BUCKET_ID_OVERRIDES[bucket_id]
    # Longest org first, so a hyphenated org still wins where another org's name
    # is a prefix of it.
    for org in sorted(HYPHENATED_ORGS, key=len, reverse=True):
        if bucket_id.startswith(f"{org}-"):
            return f"{org}/{bucket_id[len(org) + 1 :]}"
    org, sep, name = bucket_id.partition("-")
    return f"{org}/{name}" if sep else bucket_id


def to_hub_ids(models: list[str]) -> dict[str, str]:
    """Map each bucket model name in ``models`` to the Hub id to report it under.

    :func:`to_hub_id` applied one at a time cannot see that two bucket names
    landed on the same Hub id, which would silently merge two models' rows into
    one. Checked here instead: on a collision, report it and fall back to the
    bucket names for the whole set, since a recognisable wrong name beats a
    plausible-looking wrong number.
    """
    out = {model: to_hub_id(model) for model in models}
    collisions = collections.Counter(out.values())
    clashing = sorted(hub for hub, n in collisions.items() if n > 1)
    if clashing:
        for hub in clashing:
            sources = sorted(m for m, h in out.items() if h == hub)
            print(f"  warning: {' and '.join(sources)} both map to {hub}; reporting bucket names")
        return {model: model for model in models}
    return out
