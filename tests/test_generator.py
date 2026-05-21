import hashlib
import os
from pathlib import Path
import sys
from dataclasses import dataclass
from types import ModuleType

import pytest

import core.generator as generator_module
from core.generator import (
    GeneratorConfig,
    Generator,
    PhaseOutput,
    RAY_LOCAL_SOCKET_PATH_LIMIT,
    RayDataLLMBackend,
    postprocess_ray_llm_row,
    resolve_local_ray_temp_dir,
)
from core.renderer import GptOssHarmonyRenderer, QwenChatRenderer, resolve_renderer


@dataclass
class _FakeBackend:
    scripted: list[PhaseOutput]
    sampled_prompts: list[list[str]] | None = None
    sampled_prompt_token_id_lists: list[list[list[int]]] | None = None
    sampled_max_tokens: list[int | list[int]] | None = None
    stop_sequences_value: list[str] | None = None

    def render_prompt(self, prompt: str) -> str:
        return prompt

    def stop_sequences(self) -> list[str]:
        return list(self.stop_sequences_value or ["STOP"])

    def tokenize(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def sample(
        self,
        prompts,
        *,
        max_tokens: int | list[int],
        temperature: float,
        stop_sequences=None,
        prompt_token_id_lists=None,
    ):
        if self.sampled_prompts is not None:
            self.sampled_prompts.append(list(prompts))
        if self.sampled_prompt_token_id_lists is not None:
            token_id_lists = []
            if prompt_token_id_lists is not None:
                token_id_lists = [list(token_ids) for token_ids in prompt_token_id_lists]
            self.sampled_prompt_token_id_lists.append(token_id_lists)
        if self.sampled_max_tokens is not None:
            self.sampled_max_tokens.append(list(max_tokens) if isinstance(max_tokens, list) else int(max_tokens))
        count = len(prompt_token_id_lists) if prompt_token_id_lists is not None else len(prompts)
        _ = (prompts, max_tokens, temperature, stop_sequences, prompt_token_id_lists)
        return [self.scripted.pop(0) for _ in range(count)]

    def reload_adapter(self, adapter_path):
        _ = adapter_path

    def teardown(self):
        return None


_GPT_OSS_SYSTEM_PROMPT = (
    "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: 2026-03-21\n\n"
    "Reasoning: high\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
)
_GPT_OSS_STOP_SEQUENCE = "<|return|>"
_GPT_OSS_FINAL_ANSWER_MARKER = "<|channel|>final<|message|>"
_GPT_OSS_PHASE1_END_MARKER = "<|end|>"
_GPT_OSS_FORCED_FINAL_SUFFIX = (
    "\n\n... okay, I am out of thinking tokens. I need to send my final message now."
    "<|end|><|start|>assistant<|channel|>final<|message|>"
)
_GPT_OSS_FORCED_FINAL_SUFFIX_AFTER_PHASE1_END_MARKER = (
    "\n\n... okay, I am out of thinking tokens. I need to send my final message now."
    "<|start|>assistant<|channel|>final<|message|>"
)


def _gpt_oss_renderer() -> GptOssHarmonyRenderer:
    renderer = resolve_renderer(
        "gpt_oss_harmony",
        system_prompt=_GPT_OSS_SYSTEM_PROMPT,
        stop_sequence=_GPT_OSS_STOP_SEQUENCE,
    )
    assert isinstance(renderer, GptOssHarmonyRenderer)
    return renderer


def _generator_config(**overrides) -> GeneratorConfig:
    """Build a minimal generator config for unit tests."""

    payload = {
        "model_name_or_path": "fake",
        "tokenizer_name_or_path": None,
        "renderer_name": "plain_text",
        "renderer_system_prompt": "",
        "renderer_stop_sequence": "",
        "temperature": 1.0,
        "phase1_max_tokens": 128,
        "context_window": 256,
        "context_buffer": 0,
        "gpu_memory_utilization": 0.98,
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 4,
        "request_parallelism": 2,
        "request_timeout_s": 60.0,
        "backend_name": "ray_data_llm",
        "data_parallel_size": 1,
        "tensor_parallel_size": 1,
        "final_answer_marker": None,
        "forced_final_suffix": None,
        "phase1_end_marker": None,
        "forced_final_suffix_after_phase1_end_marker": None,
    }
    payload.update(overrides)
    return GeneratorConfig(**payload)


def _clear_local_tmpdir_env(monkeypatch) -> None:
    """Remove local-scratch env vars so tests can control candidate ordering."""

    for name in ("SLURM_TMPDIR", "TMPDIR", "TMP", "TEMP", "SLURM_JOB_ID"):
        monkeypatch.delenv(name, raising=False)


def test_resolve_local_ray_temp_dir_prefers_explicit_override(monkeypatch):
    _clear_local_tmpdir_env(monkeypatch)
    monkeypatch.setattr(generator_module, "prepare_local_ray_temp_dir", lambda path: path.resolve())
    config = _generator_config(
        ray_temp_dir="/tmp/ndrayoverride",
        run_dir="/tmp/nd-run",
    )

    temp_dir, source = resolve_local_ray_temp_dir(config)

    assert temp_dir == Path("/tmp/ndrayoverride").resolve()
    assert source == "env"


def test_resolve_local_ray_temp_dir_prefers_slurm_tmpdir(monkeypatch):
    _clear_local_tmpdir_env(monkeypatch)
    run_dir = Path("/network/run").resolve()
    slurm_tmpdir = Path("/local/slurm")
    tmpdir = Path("/local/tmpdir")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    expected_name = generator_module.local_ray_temp_dir_name(str(run_dir))
    seen: list[Path] = []

    monkeypatch.setenv("SLURM_TMPDIR", str(slurm_tmpdir))
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setenv("TMP", "/local/tmp")
    monkeypatch.setenv("TEMP", "/local/temp")
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setattr(generator_module, "prepare_local_ray_temp_dir", lambda path: seen.append(path) or path)

    temp_dir, source = resolve_local_ray_temp_dir(_generator_config(run_dir=str(run_dir)))

    assert temp_dir == slurm_tmpdir / expected_name
    assert source == "slurm_tmpdir"
    assert seen == [slurm_tmpdir / expected_name]


def test_resolve_local_ray_temp_dir_falls_back_to_tmpdir_when_slurm_tmpdir_fails(monkeypatch):
    _clear_local_tmpdir_env(monkeypatch)
    run_dir = Path("/network/run").resolve()
    slurm_tmpdir = Path("/local/slurm")
    tmpdir = Path("/local/tmpdir")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    expected_name = generator_module.local_ray_temp_dir_name(str(run_dir))
    slurm_candidate = slurm_tmpdir / expected_name
    tmp_candidate = tmpdir / expected_name

    monkeypatch.setenv("SLURM_TMPDIR", str(slurm_tmpdir))
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setenv("USER", "tester")

    def fake_prepare(path: Path) -> Path:
        if path == slurm_candidate:
            raise PermissionError("slurm tmpdir not writable")
        return path

    monkeypatch.setattr(generator_module, "prepare_local_ray_temp_dir", fake_prepare)

    temp_dir, source = resolve_local_ray_temp_dir(_generator_config(run_dir=str(run_dir)))

    assert temp_dir == tmp_candidate
    assert source == "tmpdir"


def test_resolve_local_ray_temp_dir_falls_back_to_run_dir_parent_after_local_candidates_fail(monkeypatch):
    _clear_local_tmpdir_env(monkeypatch)
    run_dir = Path("/network/shared/run").resolve()
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    run_dir_parent_candidate = run_dir.parent / generator_module.local_ray_temp_dir_name(str(run_dir))

    monkeypatch.setenv("SLURM_TMPDIR", "/local/slurm")
    monkeypatch.setenv("TMPDIR", "/local/tmpdir")
    monkeypatch.setenv("TMP", "/local/tmp")
    monkeypatch.setenv("TEMP", "/local/temp")
    monkeypatch.setenv("USER", "tester")

    def fake_prepare(path: Path) -> Path:
        if path == run_dir_parent_candidate:
            return path
        raise PermissionError("candidate not writable")

    monkeypatch.setattr(generator_module, "prepare_local_ray_temp_dir", fake_prepare)

    temp_dir, source = resolve_local_ray_temp_dir(_generator_config(run_dir=str(run_dir)))

    assert temp_dir == run_dir_parent_candidate
    assert source == "run_dir_parent"


def test_resolve_local_ray_temp_dir_rejects_paths_that_exceed_socket_limit(monkeypatch):
    _clear_local_tmpdir_env(monkeypatch)
    too_long_root = "/" + ("x" * RAY_LOCAL_SOCKET_PATH_LIMIT)

    with pytest.raises(RuntimeError, match="NANODISCOVER_RAY_TMPDIR"):
        resolve_local_ray_temp_dir(_generator_config(ray_temp_dir=too_long_root))


def test_resolve_local_ray_temp_dir_rejects_relative_explicit_override(monkeypatch):
    _clear_local_tmpdir_env(monkeypatch)

    with pytest.raises(RuntimeError, match="absolute"):
        resolve_local_ray_temp_dir(_generator_config(ray_temp_dir="."))


def test_resolve_local_ray_temp_dir_skips_relative_tmpdir_candidate(monkeypatch):
    _clear_local_tmpdir_env(monkeypatch)
    monkeypatch.setenv("TMPDIR", ".")
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(generator_module, "prepare_local_ray_temp_dir", generator_module.validate_local_ray_temp_dir)

    temp_dir, source = resolve_local_ray_temp_dir(_generator_config())

    assert source == "tmp_user"
    assert temp_dir == Path("/tmp") / "tester" / generator_module.local_ray_temp_dir_name(None)


def test_ensure_ray_runtime_prepares_local_env_for_ray_data_llm_init(monkeypatch, tmp_path):
    class _FakeRayModule(ModuleType):
        def __init__(self) -> None:
            super().__init__("ray")
            self.initialized = False

        def is_initialized(self) -> bool:
            return self.initialized

        def init(self, **kwargs) -> None:
            raise AssertionError(f"ensure_ray_runtime should not call ray.init: {kwargs}")

    fake_ray = _FakeRayModule()
    explicit_temp_dir = tmp_path / "ray-explicit"

    def fake_resolve(_config: GeneratorConfig) -> tuple[Path, str]:
        explicit_temp_dir.mkdir(parents=True, exist_ok=True)
        return explicit_temp_dir, "env"

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(generator_module, "resolve_local_ray_temp_dir", fake_resolve)

    backend = RayDataLLMBackend.__new__(RayDataLLMBackend)
    backend.config = _generator_config(
        ray_temp_dir=str(explicit_temp_dir),
        run_dir=str(tmp_path / "run"),
    )
    backend._owns_ray_runtime = False
    backend._ray_temp_dir = None
    backend._ray_temp_dir_source = None

    ray_module = backend.ensure_ray_runtime()

    assert ray_module is fake_ray
    assert os.environ["RAY_ADDRESS"] == "local"
    assert os.environ["RAY_TMPDIR"] == str(explicit_temp_dir)
    assert backend._owns_ray_runtime is False
    assert backend._ray_temp_dir == explicit_temp_dir
    assert backend._ray_temp_dir_source == "env"

    backend.teardown()

    assert "RAY_ADDRESS" not in os.environ
    assert "RAY_TMPDIR" not in os.environ
    assert explicit_temp_dir.exists()


def test_build_processor_pins_local_reinit_address_for_ray_data_llm(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class _FakeRayModule(ModuleType):
        def __init__(self) -> None:
            super().__init__("ray")
            self.__path__ = []
            self.initialized = False
            self.shutdown_called = False
            self.init_calls: list[dict[str, object]] = []

        def is_initialized(self) -> bool:
            return self.initialized

        def init(self, **kwargs) -> None:
            self.init_calls.append(dict(kwargs))
            expected_tmpdir = str(captured["expected_tmpdir"])
            if os.environ.get("RAY_ADDRESS") != "local" or os.environ.get("RAY_TMPDIR") != expected_tmpdir:
                raise PermissionError(13, "Permission denied", "/tmp/ray/ray_current_cluster")
            self.initialized = True

        def shutdown(self) -> None:
            self.shutdown_called = True
            self.initialized = False

    class _StubProcessorConfig:
        def __init__(self, **kwargs) -> None:
            self.runtime_env = kwargs.get("runtime_env")
            captured["config_kwargs"] = kwargs

    auto_temp_dir = tmp_path / "ray-auto"
    captured["expected_tmpdir"] = auto_temp_dir
    fake_ray = _FakeRayModule()
    fake_ray_data = ModuleType("ray.data")
    fake_ray_data.__path__ = []
    fake_ray_data_llm = ModuleType("ray.data.llm")

    def fake_build_processor(config, **kwargs):
        fake_ray.init(runtime_env=getattr(config, "runtime_env", None), ignore_reinit_error=True)
        captured["processor_config"] = config
        captured["build_kwargs"] = kwargs
        return "processor"

    fake_ray_data_llm.vLLMEngineProcessorConfig = _StubProcessorConfig
    fake_ray_data_llm.build_processor = fake_build_processor
    fake_ray.data = fake_ray_data
    fake_ray_data.llm = fake_ray_data_llm

    def fake_resolve(_config: GeneratorConfig) -> tuple[Path, str]:
        auto_temp_dir.mkdir(parents=True, exist_ok=True)
        return auto_temp_dir, "tmpdir"

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.data", fake_ray_data)
    monkeypatch.setitem(sys.modules, "ray.data.llm", fake_ray_data_llm)
    monkeypatch.setattr(generator_module, "resolve_local_ray_temp_dir", fake_resolve)

    backend = RayDataLLMBackend.__new__(RayDataLLMBackend)
    backend.config = _generator_config(run_dir=str(tmp_path / "run"), batch_size=64)
    backend.adapter_path = None
    backend._logged_config = False
    backend._owns_ray_runtime = False
    backend._ray_temp_dir = None
    backend._ray_temp_dir_source = None

    processor = backend.build_processor(use_lora=False)

    assert processor == "processor"
    assert len(fake_ray.init_calls) == 1
    assert fake_ray.init_calls[0]["ignore_reinit_error"] is True
    assert "address" not in fake_ray.init_calls[0]
    assert os.environ["RAY_ADDRESS"] == "local"
    assert os.environ["RAY_TMPDIR"] == str(auto_temp_dir)
    assert backend._owns_ray_runtime is True

    backend.teardown()

    assert fake_ray.shutdown_called is True
    assert "RAY_ADDRESS" not in os.environ
    assert "RAY_TMPDIR" not in os.environ
    assert not auto_temp_dir.exists()


def test_build_processor_cleans_auto_selected_temp_dir_when_ray_init_fails(monkeypatch, tmp_path):
    class _FakeRayModule(ModuleType):
        def __init__(self) -> None:
            super().__init__("ray")
            self.__path__ = []
            self.initialized = False

        def is_initialized(self) -> bool:
            return self.initialized

        def init(self, **kwargs) -> None:
            _ = kwargs
            raise RuntimeError("ray init failed")

    class _StubProcessorConfig:
        def __init__(self, **kwargs) -> None:
            self.runtime_env = kwargs.get("runtime_env")

    auto_temp_dir = tmp_path / "ray-auto-fail"
    fake_ray = _FakeRayModule()
    fake_ray_data = ModuleType("ray.data")
    fake_ray_data.__path__ = []
    fake_ray_data_llm = ModuleType("ray.data.llm")

    def fake_build_processor(config, **kwargs):
        _ = (config, kwargs)
        fake_ray.init(runtime_env=getattr(config, "runtime_env", None), ignore_reinit_error=True)
        return "unreachable"

    fake_ray_data_llm.vLLMEngineProcessorConfig = _StubProcessorConfig
    fake_ray_data_llm.build_processor = fake_build_processor
    fake_ray.data = fake_ray_data
    fake_ray_data.llm = fake_ray_data_llm

    def fake_resolve(_config: GeneratorConfig) -> tuple[Path, str]:
        auto_temp_dir.mkdir(parents=True, exist_ok=True)
        return auto_temp_dir, "tmpdir"

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.data", fake_ray_data)
    monkeypatch.setitem(sys.modules, "ray.data.llm", fake_ray_data_llm)
    monkeypatch.setattr(generator_module, "resolve_local_ray_temp_dir", fake_resolve)

    backend = RayDataLLMBackend.__new__(RayDataLLMBackend)
    backend.config = _generator_config(run_dir=str(tmp_path / "run"), batch_size=64)
    backend.adapter_path = None
    backend._logged_config = False
    backend._owns_ray_runtime = False
    backend._ray_temp_dir = None
    backend._ray_temp_dir_source = None

    with pytest.raises(RuntimeError, match="ray init failed"):
        backend.build_processor(use_lora=False)

    assert not auto_temp_dir.exists()
    assert backend._ray_temp_dir is None
    assert backend._ray_temp_dir_source is None

    backend.teardown()

    assert "RAY_ADDRESS" not in os.environ
    assert "RAY_TMPDIR" not in os.environ


def test_teardown_restores_previous_ray_address_when_runtime_is_external(monkeypatch):
    class _FakeRayModule(ModuleType):
        def __init__(self) -> None:
            super().__init__("ray")
            self.initialized = True

        def is_initialized(self) -> bool:
            return self.initialized

    fake_ray = _FakeRayModule()
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setenv("RAY_ADDRESS", "auto")

    backend = RayDataLLMBackend.__new__(RayDataLLMBackend)
    backend.config = _generator_config(run_dir="/tmp/nd-run")
    backend._owns_ray_runtime = False
    backend._ray_temp_dir = None
    backend._ray_temp_dir_source = None

    backend.ensure_ray_runtime()

    assert os.environ["RAY_ADDRESS"] == "local"

    backend.teardown()

    assert os.environ["RAY_ADDRESS"] == "auto"


def test_teardown_keeps_explicit_ray_temp_dir_override(monkeypatch, tmp_path):
    class _FakeRayModule(ModuleType):
        def __init__(self) -> None:
            super().__init__("ray")
            self.initialized = False

        def is_initialized(self) -> bool:
            return self.initialized

        def init(self, **kwargs) -> None:
            _ = kwargs
            self.initialized = True

        def shutdown(self) -> None:
            self.initialized = False

    fake_ray = _FakeRayModule()
    explicit_temp_dir = tmp_path / "ray-explicit"
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    def fake_resolve(_config: GeneratorConfig) -> tuple[Path, str]:
        explicit_temp_dir.mkdir(parents=True, exist_ok=True)
        return explicit_temp_dir, "env"

    monkeypatch.setattr(generator_module, "resolve_local_ray_temp_dir", fake_resolve)

    backend = RayDataLLMBackend.__new__(RayDataLLMBackend)
    backend.config = _generator_config(run_dir=str(tmp_path / "run"))
    backend._owns_ray_runtime = False
    backend._ray_temp_dir = None
    backend._ray_temp_dir_source = None

    backend.ensure_ray_runtime()
    assert explicit_temp_dir.exists()

    backend.teardown()

    assert explicit_temp_dir.exists()


def test_gpt_oss_renderer_uses_explicit_harmony_prompt_and_stop_sequence():
    renderer = _gpt_oss_renderer()

    assert isinstance(renderer, GptOssHarmonyRenderer)
    assert renderer.name == "gpt_oss_harmony"
    assert renderer.stop_sequences == (_GPT_OSS_STOP_SEQUENCE,)
    assert renderer.default_thinking_model is True
    assert renderer.system_prompt == _GPT_OSS_SYSTEM_PROMPT
    assert renderer.render_prompt("solve this") == (
        _GPT_OSS_SYSTEM_PROMPT
        + "<|start|>user<|message|>solve this<|end|><|start|>assistant"
    )


def test_generator_uses_explicit_gpt_oss_config_for_phase2_forcing():
    class _RendererBackedBackend(_FakeBackend):
        def __init__(self, *, renderer: GptOssHarmonyRenderer, **kwargs):
            super().__init__(**kwargs)
            self.renderer = renderer

        def render_prompt(self, prompt: str) -> str:
            return self.renderer.render_prompt(prompt)

        def stop_sequences(self) -> list[str]:
            return list(self.renderer.stop_sequences)

    sampled_prompts: list[list[str]] = []
    sampled_prompt_token_id_lists: list[list[list[int]]] = []
    sampled_max_tokens: list[int | list[int]] = []
    renderer = _gpt_oss_renderer()

    prompt = "solve this"
    phase1_text = "thinking"
    forced_suffix = _GPT_OSS_FORCED_FINAL_SUFFIX
    phase2_text = f"```python\npass\n```{_GPT_OSS_STOP_SEQUENCE}"
    rendered_prompt = renderer.render_prompt(prompt)
    phase1_max_tokens = len(rendered_prompt) + len(phase1_text)
    context_window = phase1_max_tokens + len(forced_suffix) + len(phase2_text) + 64

    backend = _RendererBackedBackend(
        renderer=renderer,
        scripted=[
            PhaseOutput(
                text=phase1_text,
                token_ids=[ord(ch) for ch in phase1_text],
                logprobs=[-1.0] * len(phase1_text),
                finish_reason="length",
            ),
            PhaseOutput(
                text=phase2_text,
                token_ids=[ord(ch) for ch in phase2_text],
                logprobs=[-0.5] * len(phase2_text),
                finish_reason="stop",
            ),
        ],
        sampled_prompts=sampled_prompts,
        sampled_prompt_token_id_lists=sampled_prompt_token_id_lists,
        sampled_max_tokens=sampled_max_tokens,
    )
    stage = Generator(
        GeneratorConfig(
            model_name_or_path="openai/gpt-oss-120b",
            tokenizer_name_or_path="openai/gpt-oss-120b",
            renderer_name="gpt_oss_harmony",
            renderer_system_prompt=_GPT_OSS_SYSTEM_PROMPT,
            renderer_stop_sequence=_GPT_OSS_STOP_SEQUENCE,
            temperature=1.0,
            phase1_max_tokens=phase1_max_tokens,
            context_window=context_window,
            context_buffer=50,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            request_parallelism=2,
            request_timeout_s=60.0,
            backend_name="ray_data_llm",
            data_parallel_size=1,
            tensor_parallel_size=1,
            final_answer_marker=_GPT_OSS_FINAL_ANSWER_MARKER,
            forced_final_suffix=_GPT_OSS_FORCED_FINAL_SUFFIX,
            phase1_end_marker=_GPT_OSS_PHASE1_END_MARKER,
            forced_final_suffix_after_phase1_end_marker=_GPT_OSS_FORCED_FINAL_SUFFIX_AFTER_PHASE1_END_MARKER,
        ),
        backend=backend,
    )

    output = stage.generate([prompt])[0]

    assert sampled_prompts == [[rendered_prompt], []]
    assert sampled_max_tokens[0] == [len(phase1_text)]
    assert sampled_prompt_token_id_lists[0] == []
    assert sampled_prompt_token_id_lists[1][0][-len(forced_suffix) :] == [ord(ch) for ch in forced_suffix]
    assert output.response_text == phase1_text + forced_suffix + phase2_text
    assert output.completion_mask == (
        ([1.0] * len(phase1_text))
        + ([0.0] * len(forced_suffix))
        + ([1.0] * len(phase2_text))
    )
    assert output.completion_logprobs[len(phase1_text) : len(phase1_text) + len(forced_suffix)] == [0.0] * len(forced_suffix)


def test_generator_does_not_fallback_to_renderer_strings_when_config_omits_them():
    class _RendererBackedBackend(_FakeBackend):
        def __init__(self, *, renderer: GptOssHarmonyRenderer, **kwargs):
            super().__init__(**kwargs)
            self.renderer = renderer

        def render_prompt(self, prompt: str) -> str:
            return self.renderer.render_prompt(prompt)

        def stop_sequences(self) -> list[str]:
            return list(self.renderer.stop_sequences)

    sampled_prompt_token_id_lists: list[list[list[int]]] = []
    renderer = _gpt_oss_renderer()
    prompt = "solve this"
    phase1_text = "thinking"
    phase2_text = f"```python\npass\n```{_GPT_OSS_STOP_SEQUENCE}"
    rendered_prompt = renderer.render_prompt(prompt)
    phase1_max_tokens = len(rendered_prompt) + len(phase1_text)
    context_window = phase1_max_tokens + len(phase2_text) + 64

    backend = _RendererBackedBackend(
        renderer=renderer,
        scripted=[
            PhaseOutput(
                text=phase1_text,
                token_ids=[ord(ch) for ch in phase1_text],
                logprobs=[-1.0] * len(phase1_text),
                finish_reason="length",
            ),
            PhaseOutput(
                text=phase2_text,
                token_ids=[ord(ch) for ch in phase2_text],
                logprobs=[-0.5] * len(phase2_text),
                finish_reason="stop",
            ),
        ],
        sampled_prompt_token_id_lists=sampled_prompt_token_id_lists,
    )
    stage = Generator(
        GeneratorConfig(
            model_name_or_path="openai/gpt-oss-120b",
            tokenizer_name_or_path="openai/gpt-oss-120b",
            renderer_name="gpt_oss_harmony",
            renderer_system_prompt=_GPT_OSS_SYSTEM_PROMPT,
            renderer_stop_sequence=_GPT_OSS_STOP_SEQUENCE,
            temperature=1.0,
            phase1_max_tokens=phase1_max_tokens,
            context_window=context_window,
            context_buffer=50,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            request_parallelism=2,
            request_timeout_s=60.0,
            backend_name="ray_data_llm",
            data_parallel_size=1,
            tensor_parallel_size=1,
            final_answer_marker=None,
            forced_final_suffix=None,
            phase1_end_marker=None,
            forced_final_suffix_after_phase1_end_marker=None,
        ),
        backend=backend,
    )

    output = stage.generate([prompt])[0]

    assert sampled_prompt_token_id_lists[1][0][-len(_GPT_OSS_FORCED_FINAL_SUFFIX) :] != [ord(ch) for ch in _GPT_OSS_FORCED_FINAL_SUFFIX]
    assert output.response_text == phase1_text + phase2_text
    assert output.completion_mask == ([1.0] * len(phase1_text)) + ([1.0] * len(phase2_text))


def test_generator_forced_suffix_is_zero_masked():
    backend = _FakeBackend(
        scripted=[
            PhaseOutput(text="abcd", token_ids=[1, 2, 3, 4], logprobs=[-1.0, -1.0, -1.0, -1.0], finish_reason="length"),
            PhaseOutput(text="z", token_ids=[9], logprobs=[-0.5], finish_reason="stop"),
        ]
    )
    stage = Generator(
        GeneratorConfig(
            model_name_or_path="fake",
            tokenizer_name_or_path=None,
            renderer_name="plain_text",
            renderer_system_prompt="",
            renderer_stop_sequence="",
            temperature=1.0,
            phase1_max_tokens=5,
            context_window=20,
            context_buffer=0,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            request_parallelism=2,
            request_timeout_s=60.0,
            backend_name="ray_data_llm",
            data_parallel_size=1,
            tensor_parallel_size=1,
            final_answer_marker="FINAL",
            forced_final_suffix="XY",
            phase1_end_marker=None,
            forced_final_suffix_after_phase1_end_marker=None,
        ),
        backend=backend,
    )
    output = stage.generate(["p"])[0]
    assert output.response_text == "abcdXYz"
    assert output.completion_mask == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    assert output.completion_logprobs[4:6] == [0.0, 0.0]


def test_generator_without_forcing_just_continues_phase2():
    backend = _FakeBackend(
        scripted=[
            PhaseOutput(text="abcd", token_ids=[1, 2, 3, 4], logprobs=[-1.0, -1.0, -1.0, -1.0], finish_reason="length"),
            PhaseOutput(text="zz", token_ids=[8, 9], logprobs=[-0.5, -0.6], finish_reason="stop"),
        ]
    )
    stage = Generator(
        GeneratorConfig(
            model_name_or_path="fake",
            tokenizer_name_or_path=None,
            renderer_name="plain_text",
            renderer_system_prompt="",
            renderer_stop_sequence="",
            temperature=1.0,
            phase1_max_tokens=5,
            context_window=20,
            context_buffer=0,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            request_parallelism=2,
            request_timeout_s=60.0,
            backend_name="ray_data_llm",
            data_parallel_size=1,
            tensor_parallel_size=1,
            final_answer_marker=None,
            forced_final_suffix=None,
            phase1_end_marker=None,
            forced_final_suffix_after_phase1_end_marker=None,
        ),
        backend=backend,
    )
    output = stage.generate(["p"])[0]
    assert output.response_text == "abcdzz"
    assert output.completion_mask == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_qwen_renderer_couples_prompt_and_stop_sequence():
    renderer = resolve_renderer("qwen_chat", system_prompt="", stop_sequence="<|im_end|>")

    assert renderer.name == "qwen_chat"
    assert renderer.stop_sequences == ("<|im_end|>",)
    assert renderer.default_thinking_model is True
    assert renderer.system_prompt is None
    assert renderer.render_prompt("solve this") == "<|im_start|>user\nsolve this<|im_end|>\n<|im_start|>assistant\n<think>\n"


def test_generator_renders_qwen3_prompts_as_chat_turns():
    class _QwenBackend(_FakeBackend):
        def render_prompt(self, prompt: str) -> str:
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"

    sampled_prompts: list[list[str]] = []
    backend = _QwenBackend(
        scripted=[
            PhaseOutput(text="```python\npass\n```<|im_end|>", token_ids=[1, 2, 3], logprobs=[-1.0] * 3, finish_reason="stop"),
        ],
        sampled_prompts=sampled_prompts,
        stop_sequences_value=["<|im_end|>"],
    )
    stage = Generator(
        GeneratorConfig(
            model_name_or_path="Qwen/Qwen3-8B",
            tokenizer_name_or_path="Qwen/Qwen3-8B",
            renderer_name="qwen_chat",
            renderer_system_prompt="",
            renderer_stop_sequence="<|im_end|>",
            temperature=1.0,
            phase1_max_tokens=128,
            context_window=256,
            context_buffer=0,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            request_parallelism=2,
            request_timeout_s=60.0,
            backend_name="ray_data_llm",
            data_parallel_size=1,
            tensor_parallel_size=1,
            final_answer_marker=None,
            forced_final_suffix=None,
            phase1_end_marker=None,
            forced_final_suffix_after_phase1_end_marker=None,
        ),
        backend=backend,
    )

    output = stage.generate(["solve this"])[0]

    assert sampled_prompts == [["<|im_start|>user\nsolve this<|im_end|>\n<|im_start|>assistant\n<think>\n"]]
    assert output.prompt_text == "solve this"


def test_generator_renders_qwen35_prompts_with_same_qwen_chat_template():
    rendered = resolve_renderer("qwen_chat", system_prompt="", stop_sequence="<|im_end|>").render_prompt("future model")

    assert rendered == "<|im_start|>user\nfuture model<|im_end|>\n<|im_start|>assistant\n<think>\n"


def test_qwen_instruct_renderer_omits_think_tag():
    renderer = resolve_renderer("qwen_instruct_chat", system_prompt="", stop_sequence="<|im_end|>")

    assert isinstance(renderer, QwenChatRenderer)
    assert renderer.default_thinking_model is False
    assert renderer.system_prompt is None
    assert renderer.render_prompt("answer") == "<|im_start|>user\nanswer<|im_end|>\n<|im_start|>assistant\n"


def test_qwen_renderer_matches_original_single_turn_shape_without_system_prompt():
    renderer = resolve_renderer("qwen_chat", system_prompt="", stop_sequence="<|im_end|>")

    assert renderer.render_prompt("What can you help me with?") == (
        "<|im_start|>user\n"
        "What can you help me with?<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
    )


def test_generator_raises_when_prompt_exceeds_phase1_budget():
    stage = Generator(
        GeneratorConfig(
            model_name_or_path="fake",
            tokenizer_name_or_path=None,
            renderer_name="plain_text",
            renderer_system_prompt="",
            renderer_stop_sequence="",
            temperature=1.0,
            phase1_max_tokens=3,
            context_window=20,
            context_buffer=0,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            request_parallelism=2,
            request_timeout_s=60.0,
            backend_name="ray_data_llm",
            data_parallel_size=1,
            tensor_parallel_size=1,
            final_answer_marker=None,
            forced_final_suffix=None,
            phase1_end_marker=None,
            forced_final_suffix_after_phase1_end_marker=None,
        ),
        backend=_FakeBackend(scripted=[]),
    )

    try:
        stage.generate(["prompt"])
    except ValueError as exc:
        assert "exceeds phase1_max_tokens" in str(exc)
    else:
        raise AssertionError("Expected generate() to reject prompts that exhaust phase1 budget")


def test_generator_continues_to_phase2_when_finish_reason_is_missing_at_budget():
    backend = _FakeBackend(
        scripted=[
            PhaseOutput(text="abcd", token_ids=[1, 2, 3, 4], logprobs=[-1.0, -1.0, -1.0, -1.0], finish_reason=None),
            PhaseOutput(text="zz", token_ids=[8, 9], logprobs=[-0.5, -0.6], finish_reason="stop"),
        ]
    )
    stage = Generator(
        GeneratorConfig(
            model_name_or_path="fake",
            tokenizer_name_or_path=None,
            renderer_name="plain_text",
            renderer_system_prompt="",
            renderer_stop_sequence="",
            temperature=1.0,
            phase1_max_tokens=5,
            context_window=20,
            context_buffer=0,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            request_parallelism=2,
            request_timeout_s=60.0,
            backend_name="ray_data_llm",
            data_parallel_size=1,
            tensor_parallel_size=1,
            final_answer_marker=None,
            forced_final_suffix=None,
            phase1_end_marker=None,
            forced_final_suffix_after_phase1_end_marker=None,
        ),
        backend=backend,
    )

    output = stage.generate(["p"])[0]

    assert output.response_text == "abcdzz"
    assert output.completion_mask == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_generator_stops_after_phase1_when_stop_sequence_hits_at_budget():
    backend = _FakeBackend(
        scripted=[
            PhaseOutput(text="abcSTOP", token_ids=[97, 98, 99, 83, 84, 79, 80], logprobs=[-1.0] * 7, finish_reason=None),
        ]
    )
    stage = Generator(
        GeneratorConfig(
            model_name_or_path="fake",
            tokenizer_name_or_path=None,
            renderer_name="plain_text",
            renderer_system_prompt="",
            renderer_stop_sequence="",
            temperature=1.0,
            phase1_max_tokens=8,
            context_window=20,
            context_buffer=0,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            request_parallelism=2,
            request_timeout_s=60.0,
            backend_name="ray_data_llm",
            data_parallel_size=1,
            tensor_parallel_size=1,
            final_answer_marker=None,
            forced_final_suffix=None,
            phase1_end_marker=None,
            forced_final_suffix_after_phase1_end_marker=None,
        ),
        backend=backend,
    )

    output = stage.generate(["p"])[0]

    assert output.response_text == "abcSTOP"
    assert output.completion_token_ids == [97, 98, 99, 83, 84, 79, 80]
    assert output.completion_mask == [1.0] * 7


def test_generator_uses_alternate_forced_suffix_after_phase1_end_marker():
    backend = _FakeBackend(
        scripted=[
            PhaseOutput(text="abcEND", token_ids=[97, 98, 99, 69, 78, 68], logprobs=[-1.0] * 6, finish_reason="length"),
            PhaseOutput(text="z", token_ids=[9], logprobs=[-0.5], finish_reason="stop"),
        ]
    )
    stage = Generator(
        GeneratorConfig(
            model_name_or_path="fake",
            tokenizer_name_or_path=None,
            renderer_name="plain_text",
            renderer_system_prompt="",
            renderer_stop_sequence="",
            temperature=1.0,
            phase1_max_tokens=7,
            context_window=20,
            context_buffer=0,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            request_parallelism=2,
            request_timeout_s=60.0,
            backend_name="ray_data_llm",
            data_parallel_size=1,
            tensor_parallel_size=1,
            final_answer_marker="FINAL",
            forced_final_suffix="XYFINAL",
            phase1_end_marker="END",
            forced_final_suffix_after_phase1_end_marker="FINAL",
        ),
        backend=backend,
    )

    output = stage.generate(["p"])[0]

    assert output.response_text == "abcENDFINALz"
    assert output.completion_mask == [1.0] * 6 + [0.0] * 5 + [1.0]


def test_generator_batches_phase1_and_phase2_requests():
    sampled_prompts: list[list[str]] = []
    sampled_prompt_token_id_lists: list[list[list[int]]] = []
    sampled_max_tokens: list[int | list[int]] = []
    backend = _FakeBackend(
        scripted=[
            PhaseOutput(text="a", token_ids=[1], logprobs=[-0.1], finish_reason="stop"),
            PhaseOutput(text="bbbb", token_ids=[2, 2, 2, 2], logprobs=[-0.2] * 4, finish_reason="length"),
            PhaseOutput(text="c", token_ids=[3], logprobs=[-0.3], finish_reason="stop"),
        ],
        sampled_prompts=sampled_prompts,
        sampled_prompt_token_id_lists=sampled_prompt_token_id_lists,
        sampled_max_tokens=sampled_max_tokens,
    )
    stage = Generator(
        GeneratorConfig(
            model_name_or_path="fake",
            tokenizer_name_or_path=None,
            renderer_name="plain_text",
            renderer_system_prompt="",
            renderer_stop_sequence="",
            temperature=1.0,
            phase1_max_tokens=5,
            context_window=20,
            context_buffer=0,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            request_parallelism=2,
            request_timeout_s=60.0,
            backend_name="ray_data_llm",
            data_parallel_size=1,
            tensor_parallel_size=1,
            final_answer_marker=None,
            forced_final_suffix=None,
            phase1_end_marker=None,
            forced_final_suffix_after_phase1_end_marker=None,
        ),
        backend=backend,
    )

    outputs = stage.generate(["p", "q"])

    assert len(sampled_prompts) == 2
    assert sampled_prompts[0] == ["p", "q"]
    assert sampled_max_tokens[0] == [4, 4]
    assert sampled_prompts[1] == []
    assert sampled_prompt_token_id_lists[0] == []
    assert sampled_prompt_token_id_lists[1] == [[ord("q"), 2, 2, 2, 2]]
    assert sampled_max_tokens[1] == [15]
    assert [output.response_text for output in outputs] == ["a", "bbbbc"]


def test_generator_batches_contiguous_identical_prompts_in_phase1():
    backend = _FakeBackend(
        scripted=[
            PhaseOutput(text="a", token_ids=[1], logprobs=[-1.0], finish_reason="stop"),
            PhaseOutput(text="b", token_ids=[2], logprobs=[-1.1], finish_reason="stop"),
        ]
    )
    stage = Generator(
        GeneratorConfig(
            model_name_or_path="fake",
            tokenizer_name_or_path=None,
            renderer_name="plain_text",
            renderer_system_prompt="",
            renderer_stop_sequence="",
            temperature=1.0,
            phase1_max_tokens=5,
            context_window=20,
            context_buffer=0,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            request_parallelism=2,
            request_timeout_s=60.0,
            backend_name="vllm_service",
            data_parallel_size=1,
            tensor_parallel_size=1,
            final_answer_marker=None,
            forced_final_suffix=None,
            phase1_end_marker=None,
            forced_final_suffix_after_phase1_end_marker=None,
        ),
        backend=backend,
    )

    outputs = stage.generate(["p", "p"])

    assert [output.response_text for output in outputs] == ["a", "b"]


def test_extract_token_logprobs_from_ray_row():
    row = {
        "logprobs": [
            {11: {"logprob": -0.1, "rank": 1, "decoded_token": "a"}},
            {"22": {"logprob": -0.2, "rank": 1, "decoded_token": "b"}},
        ]
    }

    assert RayDataLLMBackend.extract_token_logprobs(row, [11, 22]) == [-0.1, -0.2]


def test_postprocess_ray_llm_row_strips_raw_logprobs_and_keeps_compact_values():
    row = {
        "row_index": 3,
        "prompt": "prompt",
        "tokenized_prompt": [1, 2],
        "sampling_params": {"logprobs": 1},
        "generated_tokens": [11, 22],
        "generated_text": "ab",
        "finish_reason": "stop",
        "logprobs": [
            {11: {"logprob": -0.1, "rank": 1, "decoded_token": "a"}},
            {22: {"logprob": -0.2, "rank": 1, "decoded_token": "b"}},
        ],
    }

    processed = postprocess_ray_llm_row(row)

    assert processed["row_index"] == 3
    assert processed["generated_tokens"] == [11, 22]
    assert processed["generated_text"] == "ab"
    assert processed["finish_reason"] == "stop"
    assert processed["completion_logprobs"] == [-0.1, -0.2]
    assert processed["logprobs"] is None
    assert "prompt" not in processed
    assert "tokenized_prompt" not in processed
    assert "sampling_params" not in processed


def test_generator_selects_ray_backend(monkeypatch):
    selected = {}

    class _StubRayBackend:
        def __init__(self, config):
            selected["backend_name"] = config.backend_name

        def reload_adapter(self, adapter_path):
            selected["adapter_path"] = adapter_path

        def teardown(self):
            return None

    monkeypatch.setattr("core.generator.RayDataLLMBackend", _StubRayBackend)
    stage = Generator(
        GeneratorConfig(
            model_name_or_path="fake",
            tokenizer_name_or_path=None,
            renderer_name="plain_text",
            renderer_system_prompt="",
            renderer_stop_sequence="",
            temperature=1.0,
            phase1_max_tokens=5,
            context_window=20,
            context_buffer=0,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            request_parallelism=2,
            request_timeout_s=60.0,
            backend_name="ray_data_llm",
            data_parallel_size=2,
            tensor_parallel_size=1,
            final_answer_marker=None,
            forced_final_suffix=None,
            phase1_end_marker=None,
            forced_final_suffix_after_phase1_end_marker=None,
        )
    )

    stage.reload_adapter("/tmp/adapter")

    assert selected == {"backend_name": "ray_data_llm", "adapter_path": "/tmp/adapter"}


def test_generator_rejects_removed_legacy_backend_name():
    stage = Generator(
        GeneratorConfig(
            model_name_or_path="fake",
            tokenizer_name_or_path=None,
            renderer_name="plain_text",
            renderer_system_prompt="",
            renderer_stop_sequence="",
            temperature=1.0,
            phase1_max_tokens=5,
            context_window=20,
            context_buffer=0,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            request_parallelism=2,
            request_timeout_s=60.0,
            backend_name="vllm_service",
            data_parallel_size=1,
            tensor_parallel_size=1,
            final_answer_marker=None,
            forced_final_suffix=None,
            phase1_end_marker=None,
            forced_final_suffix_after_phase1_end_marker=None,
        )
    )

    with pytest.raises(RuntimeError, match="only supports ray_data_llm"):
        stage.reload_adapter(None)

def test_ray_backend_builds_pretokenized_processor(monkeypatch):
    pytest.importorskip("ray.data.llm")
    captured = {}

    class _StubProcessorConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs

    def _fake_build_processor(config, **kwargs):
        captured["processor_config"] = config
        captured["kwargs"] = kwargs
        return "processor"

    monkeypatch.setattr("ray.data.llm.vLLMEngineProcessorConfig", _StubProcessorConfig)
    monkeypatch.setattr("ray.data.llm.build_processor", _fake_build_processor)
    monkeypatch.setattr("ray.is_initialized", lambda: True)

    backend = object.__new__(RayDataLLMBackend)
    backend.config = GeneratorConfig(
        model_name_or_path="fake",
        tokenizer_name_or_path=None,
        renderer_name="plain_text",
        renderer_system_prompt="",
        renderer_stop_sequence="",
        temperature=1.0,
        phase1_max_tokens=32,
        context_window=2048,
        context_buffer=0,
        gpu_memory_utilization=0.98,
        max_num_batched_tokens=16384,
        max_num_seqs=64,
        request_parallelism=8,
        request_timeout_s=60.0,
        backend_name="ray_data_llm",
        data_parallel_size=4,
        tensor_parallel_size=2,
        final_answer_marker=None,
        forced_final_suffix=None,
        phase1_end_marker=None,
        forced_final_suffix_after_phase1_end_marker=None,
        batch_size=64,
    )
    backend.adapter_path = None
    backend._logged_config = False
    backend._owns_ray_runtime = False
    backend._ray_temp_dir = None
    backend._ray_temp_dir_source = None

    processor = backend.build_processor(use_lora=False)

    assert processor == "processor"
    assert captured["processor_config"] is not None
    assert callable(captured["kwargs"]["postprocess"])
    assert captured["config_kwargs"]["chat_template_stage"] is False
    assert captured["config_kwargs"]["tokenize_stage"] is False
    assert captured["config_kwargs"]["detokenize_stage"] is False
    assert captured["config_kwargs"]["concurrency"] == 4
    assert captured["config_kwargs"]["engine_kwargs"] == {
        "tensor_parallel_size": 2,
        "max_model_len": 2048,
        "trust_remote_code": True,
        "enable_prefix_caching": True,
    }

    backend.teardown()


def test_ray_backend_decodes_generated_tokens_when_generated_text_missing():
    backend = object.__new__(RayDataLLMBackend)
    backend.decode = lambda token_ids: f"decoded:{token_ids}"

    assert backend.response_text_from_row({}, [1, 2, 3]) == "decoded:[1, 2, 3]"