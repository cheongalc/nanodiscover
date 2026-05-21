from __future__ import annotations

from dataclasses import dataclass


KNOWN_RENDERER_NAMES = (
    "plain_text",
    "qwen_chat",
    "qwen_instruct_chat",
    "gpt_oss_harmony",
)


@dataclass(frozen=True)
class RendererSpec:
    """Base prompt-rendering contract for supported model families."""

    name: str
    stop_sequences: tuple[str, ...]
    system_prompt_text: str = ""

    @property
    def system_prompt(self) -> str | None:
        return self.system_prompt_text or None

    @property
    def default_thinking_model(self) -> bool:
        return False

    def render_prompt(self, prompt: str) -> str:
        return prompt


@dataclass(frozen=True)
class PlainTextRenderer(RendererSpec):
    """Renderer that forwards prompts as plain text."""

    name: str = "plain_text"
    stop_sequences: tuple[str, ...] = ()
    system_prompt_text: str = ""

    def render_prompt(self, prompt: str) -> str:
        if self.system_prompt:
            return f"{self.system_prompt}{prompt}"
        return prompt


@dataclass(frozen=True)
class QwenChatRenderer(RendererSpec):
    """Renderer for the Qwen chat formats used by NanoDiscover."""

    include_think_tag: bool = True
    name: str = "qwen_chat"
    stop_sequences: tuple[str, ...] = ()
    system_prompt_text: str = ""

    @property
    def default_thinking_model(self) -> bool:
        return self.include_think_tag

    def render_prompt(self, prompt: str) -> str:
        # Parity: TTT-Discover does not inject a default Qwen system prompt for
        # the common single-turn flow, so we only render one when the launcher
        # explicitly provides it.
        rendered = (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            if self.system_prompt
            else ""
        )
        think_prefix = "<think>\n" if self.include_think_tag else ""
        return f"{rendered}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{think_prefix}"


@dataclass(frozen=True)
class GptOssHarmonyRenderer(RendererSpec):
    """Render GPT-OSS prompts in Harmony format with analysis-first generation."""

    name: str = "gpt_oss_harmony"
    stop_sequences: tuple[str, ...] = ()
    system_prompt_text: str = ""

    @property
    def default_thinking_model(self) -> bool:
        return True

    def render_prompt(self, prompt: str) -> str:
        # Parity: the original GptOssRenderer.build_generation_prompt() creates a
        # partial Message(role="assistant", content="") and only appends its prefix,
        # which is "<|start|>assistant".  The model itself generates the channel
        # header (<|channel|>analysis<|message|> or <|channel|>final<|message|>).
        # We must NOT force the analysis channel here.
        rendered = ""
        if self.system_prompt:
            rendered += self.system_prompt
        rendered += f"<|start|>user<|message|>{prompt}<|end|>"
        rendered += "<|start|>assistant"
        return rendered


def build_qwen_renderer(
    *,
    name: str,
    include_think_tag: bool,
    system_prompt: str,
    stop_sequences: tuple[str, ...],
) -> QwenChatRenderer:
    """Build one of the supported Qwen chat renderer variants."""

    return QwenChatRenderer(
        include_think_tag=include_think_tag,
        name=name,
        stop_sequences=stop_sequences,
        system_prompt_text=system_prompt,
    )


def resolve_renderer(renderer_name: str, *, system_prompt: str, stop_sequence: str) -> RendererSpec:
    """Build the renderer implementation for the configured model family."""

    normalized = str(renderer_name or "").strip().lower()
    stop_sequences = (stop_sequence,) if stop_sequence else ()
    if normalized == "plain_text":
        return PlainTextRenderer(stop_sequences=stop_sequences, system_prompt_text=system_prompt)
    if normalized == "qwen_chat":
        return build_qwen_renderer(
            name="qwen_chat",
            include_think_tag=True,
            system_prompt=system_prompt,
            stop_sequences=stop_sequences,
        )
    if normalized in {"qwen_instruct_chat", "qwen_chat_instruct"}:
        return build_qwen_renderer(
            name="qwen_instruct_chat",
            include_think_tag=False,
            system_prompt=system_prompt,
            stop_sequences=stop_sequences,
        )
    if normalized == "gpt_oss_harmony":
        return GptOssHarmonyRenderer(stop_sequences=stop_sequences, system_prompt_text=system_prompt)
    expected = ", ".join(KNOWN_RENDERER_NAMES)
    raise ValueError(f"Unsupported renderer_name={renderer_name!r}; expected one of: {expected}")