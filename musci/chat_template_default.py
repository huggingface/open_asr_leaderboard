from dataclasses import dataclass
from typing import Literal, Optional
import numpy


@dataclass
class MusciChatTemplateSegment:
    type: Literal["constant_text_token", "text_token", "audio_token", "audio_contiguous"]
    add_loss: bool = True
    text_ids: Optional[numpy.ndarray] = None
    text_token_idx: Optional[int] = None
    text_token_key: Optional[str] = None

    def __post_init__(self) -> None:
        if self.type == "constant_text_token":
            assert self.text_ids is not None
        elif self.type == "text_token":
            assert self.text_token_key is not None and self.text_token_idx is not None
        elif self.type in ("audio_token", "audio_contiguous"):
            assert not self.add_loss


STYLE_CONTROL_TEXT = ""

chat_template = [
    # <|im_start|>user\n<|audio_start|>
    MusciChatTemplateSegment(
        type="constant_text_token",
        text_ids=numpy.array([151644, 872, 198, 151669]),
        add_loss=False,
    ),
    MusciChatTemplateSegment(
        type="audio_contiguous",
        add_loss=False,
    ),
    # <|audio_end|><|im_end|>\n<|im_start|>assistant\n
    MusciChatTemplateSegment(
        type="constant_text_token",
        text_ids=numpy.array([151670, 151645, 198, 151644, 77091, 198]),
        add_loss=False,
    ),
    MusciChatTemplateSegment(
        type="text_token",
        text_token_key="text_token_transcript",
        text_token_idx=0,
        add_loss=True,
    ),
    # <|im_end|>
    MusciChatTemplateSegment(
        type="constant_text_token",
        text_ids=numpy.array([151645]),
        add_loss=True,
    ),
]

__all__ = ["MusciChatTemplateSegment", "STYLE_CONTROL_TEXT", "chat_template"]
