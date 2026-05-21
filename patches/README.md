# Patches

## Ray Data LLM LoRA fix (`vllm_engine_stage.py`)

`ray[llm]==2.54.0` has a bug where per-row LoRA requests are not dispatched correctly during generation, causing LoRA adapters to have no effect. This means test-time training (TTT) does not work without this patch.

We have filed an upstream issue and fix: https://github.com/ray-project/ray/pull/62609.

After installing `requirements.txt`, apply the patch by replacing the affected file in your Ray installation:

```bash
cp patches/vllm_engine_stage.py \
  "$(python -c 'import ray, os; print(os.path.dirname(ray.__file__))')/llm/_internal/batch/stages/vllm_engine_stage.py"
```

Run this from the `nanodiscover` root with your runtime venv activated.
