# Known Differences from TTT-Discover

## Strict Mode (`strict` branch)

The `strict` branch removes parent construction passdown entirely from `ac1`, `ac2`, and `erdos`. In TTT-Discover's non-strict setting, the parent solution's mathematical construction is injected into the child's execution namespace (as `height_sequence_1` for ac1/ac2, and `initial_h_values` for erdos), allowing the child to start its search from the parent's known-good construction. In strict mode:

- `height_sequence_1` is not injected into the ac1/ac2 execution namespace.
- `initial_h_values` is not injected into the erdos execution namespace.
- The starter code templates start from random initialization instead.
- The prompts do not mention these variables or suggest starting from the parent construction.

The child must produce a construction entirely from scratch. This is a stricter notion of discovery. Circle packing is unaffected because it never had construction passdown.

## AC1 Prompt Text

`nanodiscover` intentionally differs from the released AC1 prompt in the current-value wording for seed/no-parent states:

- Released TTT-Discover renders `Current upper bound (higher is better)` for AC1 seed prompts.
- `nanodiscover` renders `Current upper bound (lower is better)`.

This is an intentional correction. AC1 minimizes the upper bound, and the released text is inconsistent with the task objective. See [`ttt_discover/tinker_utils/state.py` line 103](https://github.com/test-time-training/discover/blob/6c40e82dab9d5de7416ac873ad5cd3106084aaed/ttt_discover/tinker_utils/state.py#L103) where the seed/no-parent state branch hardcodes `"higher is better"` instead of using the `improvement_direction` variable that is correctly computed on line 82. Parent/child prompts already use `improvement_direction` and are therefore correct.

Other AC1 prompt differences are formatting-only:

- `nanodiscover` removes trailing whitespace on blank lines inside the displayed starter program.
- The starter algorithm, evaluation function, task instructions, CPU count, timeout wording, and `height_sequence_1` wording are otherwise aligned with the released prompt.

## AC2 Prompt Text

All differences from the released AC2 prompt are text cleanup only. The starter program, task objective, evaluator logic, `construct_function` entrypoint, `height_sequence_1` availability, allowed-library list, CPU count, and timeout are aligned with the released TTT-Discover implementation.

Specific cleanup changes:

- The released AC2 prompt begins with a stray leading apostrophe before `Act as...`; `nanodiscover` removes it.
- `nanodiscover` keeps a normal blank line between `import numpy as np` and `def evaluate_sequence(...)` in the displayed evaluator code.
- `nanodiscover` keeps a blank line between the literature block and the next task-instruction paragraph.
- `nanodiscover` uses ASCII `x2` instead of Unicode `×2` in two starter-code comments.
- `nanodiscover` removes trailing spaces from prompt lines.

## Erdos Prompt Text

`nanodiscover` intentionally corrects two released Erdos prompt inaccuracies:

- Released TTT-Discover can render `Current C5 bound (higher is better)` for seed/no-parent states even though Erdos minimizes the C5 bound. `nanodiscover` renders `lower is better`.
- Released TTT-Discover says `2 CPUs available` while configuring the Erdos evaluator with one CPU per task. `nanodiscover` says `1 available` to match the actual evaluator resource envelope.

## Erdos Evaluator Timeout

The initial TTT-Discover release used a 530-second evaluator timeout for Erdos while the prompt tells the model it has a 1000-second budget. This was updated to 1100 seconds in [commit bf20511](https://github.com/test-time-training/discover/commit/bf205118d27fbb6b25be71ae16126aac581a61b5) (March 30, 2026). `nanodiscover` mirrors the 1100-second timeout to give the model sufficient time to use its full budget plus overhead.

## Erdos Evaluator Scoring

The released TTT-Discover Erdos evaluator returns the model's self-reported C5 bound directly after verifying it passes a tolerance check (`atol=1e-4`). Because the tolerance is nonzero, a model can report an optimistically low C5 value that still passes verification, causing the archive and global-best tracker to record a score that is better than the true computed bound.

`nanodiscover` instead returns the C5 value computed by the verifier, so `raw_score`, the archive, and global-best tracking always reflect the true bound rather than the model's self-report.

## Training Backend

TTT-Discover uses the [Tinker API](https://thinkingmachines.ai/tinker/) as a persistent training client that remains alive across all epochs. `nanodiscover` replaces this with a self-contained DeepSpeed + LoRA training pipeline that spawns a fresh subprocess each epoch.

Each training step proceeds in two phases:

**Phase 1: Reference policy scoring.** Before DeepSpeed is initialized, each GPU independently loads a frozen copy of the base model (no LoRA adapter) and scores its assigned shard of the epoch's rollouts. This produces per-token log-probabilities under the base model, which are used to compute a per-token KL divergence between the current policy and the base model. The KL diffs are all-reduced across ranks to get a global average, and the rollout advantages are adjusted in-place by the KL penalty before training begins. The base model is then deleted from GPU memory. This matches TTT-Discover's shaping of the advantages before RL training.

**Phase 2: LoRA training.** DeepSpeed initializes with ZeRO-2 and Ulysses sequence parallelism. The policy gradient loss uses the `sampling_logprobs` stored at generation time (i.e., the log-probabilities the LoRA-adapted model assigned to each token when it generated the rollout), not the reference policy log-probabilities from phase 1. The loss is TTT-Discover's modified GRPO-style objective that is designed to chase maximum-reward actions. To make training more efficient, we also implement sequence packing to reduce padding. First we define a token budget for **all** ranks as `NANODISCOVER_TRAINER_MAX_TOKENS_PER_RANK * NANODISCOVER_SEQUENCE_PARALLEL_SIZE`. Samples longer than this budget are truncated from the end (therefore it is important to ensure that you have enough GPUs to handle the full context length of the model, this way you will not have any truncation). The remaining samples are sorted by length and greedily packed into microbatches that stay within the budget. Then sequence parallelism splits the microbatches across the ranks. Gradients are accumulated across all microbatches with a single optimizer step at the end.

**Generation and training stack.** TTT-Discover uses Tinker as an integrated client for both inference and training. `nanodiscover` keeps these on entirely separate stacks: generation uses Ray Data LLM with vLLM as the inference backend, while training uses DeepSpeed. This can introduce minor differences in how arithmetic is performed during forward passes. Whether Tinker integrates inference and training more tightly internally is not publicly documented.

**Reference logprob precision.** Log-probability computation during reference policy scoring uses float32 (`NANODISCOVER_TRAINER_LOGPROB_COMPUTE_DTYPE=float32`) rather than bfloat16. KL divergence involves subtracting log-probabilities that can be small in magnitude, and bfloat16's reduced mantissa precision can cause significant cancellation error in this calculation, so we chose float32. The vocab dimension is also chunked during logprob gathering (`NANODISCOVER_REFERENCE_LOGPROB_VOCAB_CHUNK_SIZE`) to avoid materializing the full logit tensor in float32 at once, as that can cause OOMs.

**Optimizer state persistence.** Because TTT-Discover's Tinker client is persistent, Adam optimizer state (first and second moments) carries over naturally between epochs. `nanodiscover`'s subprocess-per-epoch design requires explicit save and reload of the ZeRO-2 optimizer state to match this behavior. Optimizer states are saved at the end of each epoch and reloaded at the start of the next. `NANODISCOVER_OPTIMIZER_STATE_KEEP_WINDOW` controls how many past epochs of optimizer state are retained on disk.
