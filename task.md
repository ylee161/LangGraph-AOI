# LangGraph AOI MLE-STAR Agent — Task Checklist

> **LangGraph quick-ref (verified June 2026)**
> - Install: `pip install -U langgraph`
> - Checkpointer pkg: `pip install langgraph-checkpoint-sqlite`
> - Studio CLI: `pip install --upgrade "langgraph-cli[inmem]"` then `langgraph dev`
> - Studio URL: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`
> - `SqliteSaver` import: `from langgraph.checkpoint.sqlite import SqliteSaver`
> - `MemorySaver` import: `from langgraph.checkpoint.memory import MemorySaver`
> - Checkpointer is passed to **`g.compile(checkpointer=...)`**, not `build_graph()`'s return
> - Python ≥ 3.11 required for Studio (project uses 3.13 ✓)

---

## Component 1: Shared Utilities

- [x] **1.1** Update `state.py` — all Phase 1-4 fields, `Annotated` reducers, canonical metric keys (`ng_recall`, `miss_rate`, `overkill_rate`, `accuracy`, `f1`), counter names (`outer_iteration` / `inner_iteration` / `ensemble_iteration`)
- [x] **1.2** Update `config.py` — lazy `require_api_key()`, `NO_IMPROVE_MAX_CONSTRAINED=5`, `DEBUG_MODE`/`DEBUG_CHECK_TIMEOUT_SECONDS`, `BOARD_CODE_PATTERN`, `CKPT_L0`, all ADK thresholds
  - Model names: `MODEL_FLASH = "deepseek/deepseek-v4-flash"` (thinking disabled), `MODEL_PRO = "deepseek/deepseek-v4-pro"` (extended thinking, `budget_tokens=16000`, `reasoning_effort="max"`)
  - `TOKEN_BUDGET = 10_000_000`, `TOKEN_LITE_THRESHOLD = 7_000_000`
- [x] **1.3** Port `shared/checkpoint_io.py` — atomic JSON read/write, numpy-safe encoder
- [x] **1.8** Port `shared/acceptance_scoring.py` — `passes_relaxed_acceptance`, `passes_final_acceptance`, `acceptance_distance`, `is_acceptance_improvement`
- [x] **1.4** Port `shared/code_runner.py` — subprocess executor with configurable timeout; honours `DEBUG_MODE` (1 epoch / 10 samples / `DEBUG_CHECK_TIMEOUT_SECONDS`)
- [x] **1.5** Port `shared/metrics_parser.py` — parses `ng_recall`, `miss_rate`, `overkill_rate`, `accuracy`, `f1` from stdout; also parses `PREDICTIONS` per-sample blocks and `CALIBRATION_STATS`; exports `REQUIRED_GENERATED_SCRIPT_MARKERS`
- [x] **1.6** Port `shared/metric_guard.py` ⚠️ — guards against degenerate metrics (flat probabilities, zero splits); called on every metric parse path; correctness-critical
- [x] **1.7** Port `shared/data_split.py` — grouped board-level train/val/test split using `BOARD_CODE_PATTERN`; boards never split across partitions
- [x] **1.9** Port `shared/labels.py` — reads `.xlsx` label sheets; maps raw labels to G/NG via `FAIL_LABELS` / `PASS_LABELS`
- [x] **1.10** Create `shared/llm.py` — LiteLLM call wrapper; prompt builder; structured JSON response parser; token counter; switches to `MODEL_FLASH` above `TOKEN_LITE_THRESHOLD`; built-in rate-limit retry with backoff (replaces ADK `callbacks.py`); token budget stop (raises if `tokens_used >= TOKEN_BUDGET`)
- [x] **1.11** Port `shared/aoi_smoke_triage.py` — parses script stdout into structured smoke diagnostics dict (called by both Phase 1 candidate evaluator and Phase 2 evaluator); detects missing required output markers, degenerate probabilities, zero-sample splits; feeds into `metric_guard`

---

## Component 2: Code Validator Guard

- [x] **2.1** Port `guards/code_validator.py` from ADK `guards/code_validator_agent.py` — **as a plain callable function** (not an ADK agent):
  - [x] 2.1.1 Inject dry-run env vars (`DRY_RUN=1`, `DRY_RUN_EPOCHS=1`, `DRY_RUN_SAMPLES=10`, seeds)
  - [x] 2.1.2 **Run the script** via `code_runner` with `_VALIDATOR_TIMEOUT=120s` — reject scripts that error, hang, or produce no output
  - [x] 2.1.3 Check dry-run ternary contract: `epochs = DRY_RUN_EPOCHS if DRY_RUN else N` must be present
  - [x] 2.1.4 Static check: `data_usage_validator` — both L+R stereo images must be loaded
  - [x] 2.1.5 Static check: `lr_schedule_validator` — `scheduler.step()` must be present
  - [x] 2.1.6 Static check: `difference_feature_validator` — stereo difference features handled correctly
  - [x] 2.1.7 Static check: `small_data_strategy_validator` — no `KNOWN_FAILED_STRATEGY_FINGERPRINTS`
  - [x] 2.1.8 Check required metric output markers from `REQUIRED_GENERATED_SCRIPT_MARKERS`
  - [x] 2.1.9 Validation cache — SHA-256 hash identical scripts; skip re-validation; persist cache to `CKPT_VALIDATION_CACHE`

---

## Component 3: Phase 1 — Initialization

- [x] **3.1** Create `nodes/phase1_init.py`:
  - [x] 3.1.1 **Skip check + state restore** — check `CKPT_CANDIDATE_SCORES` + `CKPT_L0`; if both exist, restore `current_best_score`, `best_miss_rate`, `best_overkill_rate`, `best_accuracy`, `best_f1`, `best_candidate_name` from `L0.json` into state then return early
  - [x] 3.1.2 **Data split** — load lot folders → `labels.py` → `data_split.py` grouped board split → save to `CKPT_DATA_SPLIT`
  - [x] 3.1.3 **Retriever** — web search (Tavily → Serper → DuckDuckGo keyless fallback) for **M=4** candidate architectures; returns `{model_name, description, example_code}` pairs; falls back to LLM knowledge of current (2024-25) backbones if all search methods fail
  - [x] 3.1.4 **Baseline coder** — for each of the M candidates, LLM generates a full PyTorch training script with stereo L+R fusion; pass each through `code_validator` before running
  - [x] 3.1.5 **Candidate evaluator** — run all M scripts via `code_runner` + `aoi_smoke_triage`; parse metrics with `metric_guard`; **permanently ban failed architectures** (add to `HARD_EXCLUDED_ARCHITECTURES`); sort scored `{name, script, metrics, architecture}` best-first
  - [x] 3.1.6 **Merger** — start with top candidate as `s_0`; for each remaining candidate LLM generates a merged script (simple ensemble integration); if merged score > current best, replace `s_0`; stop when no merge improves
  - [x] 3.1.7 **Save L0** — write `CKPT_L0` (script + all 6 best-snapshot fields); write `CKPT_CANDIDATE_SCRIPTS` + `CKPT_CANDIDATE_SCORES`
- [x] **3.2** Write unit test for Phase 1 in `DRY_RUN=1` mode (validates skip-check, data split, L0 save)

---

## Component 4: Phase 2 — Refinement (Nested Loops)

- [x] **4.1** Create `nodes/phase2_ablation.py`:
  - [x] 4.1.1 Implement the **6 fixed AOI-specific ablation variants** from `ABLATION_VARIANTS`: `no_stereo_fusion`, `no_weighted_loss`, `no_threshold_sweep`, `no_augmentation`, `threshold_acceptance_distance`, `fp_penalty_loss`
  - [x] 4.1.2 Run each variant as its own step (mirroring ADK's `_make_variant_step_agent`); checkpoint each result individually so partial ablations can be resumed
  - [x] 4.1.3 Run each variant script via `code_runner` + `metric_guard`; compute delta vs current best
  - [x] 4.1.4 Pass previous ablation summaries as context so outer iterations target different components
  - [x] 4.1.5 Save to `ckpt_ablation(outer_iteration)` (also added `ckpt_ablation()` and `ckpt_ablation_variant()` helpers to config.py)
  - [x] 4.1.6 **Also set `stop_outer_loop = True`** when patience/cap exit conditions are met — ablation is the outer loop's exit signal source
  - [x] 4.1.7 Return `{ablation_results, target_component, stop_outer_loop}` ranked by impact

- [x] **4.2** Create `nodes/phase2_diagnosis.py`:
  - [x] 4.2.1 **Checkpoint gate with lineage check** — load `diagnosis_N.json` if ablation SHA-256 matches; recompute if stale or empty ranking
  - [x] 4.2.2 LLM reads ablation summary → identifies target code block `c_t` (prefers untargeted blocks) + initial plan `p_0`
  - [x] 4.2.3 Return `{diagnosis_report, target_component, target_block_code, refinement_plan: p_0}`

- [x] **4.3** Create `nodes/phase2_error_analysis_gate.py`:
  - [x] 4.3.1 Iteration 0: skip (set `inner_iteration = 0`), pass through to planner
  - [x] 4.3.2 Subsequent iterations: check if last script emitted `PREDICTIONS` per-sample output
  - [x] 4.3.3 First missing evidence: set `error_analysis_instrumentation_required = True`; allow one repair iteration; coder must emit `PREDICTIONS`
  - [x] 4.3.4 Second missing evidence (repair attempted): escalate — block inner loop ("blind refinement" prevention)

- [x] **4.4** Create `nodes/phase2_error_analysis.py`:
  - [x] 4.4.1 Parse `PREDICTIONS` blocks from last stdout: FP/FN counts, probability distribution summary, capped sample lists (`ERROR_ANALYSIS_SAMPLE_CAP`)
  - [x] 4.4.2 Lineage-check against last script run; load from `ckpt_error_analysis(outer, inner)` if current
  - [x] 4.4.3 Return `{error_analysis_report, latest_error_analysis}`

- [x] **4.5** Create `nodes/phase2_planner.py`:
  - [x] 4.5.1 Load-on-demand: target block `c_t`, all prior `{plan, score}` attempts this outer step, error analysis, strategy history
  - [x] 4.5.2 (v2) Integrate `kb_semantic` — rank semantically similar records from persistent KB
  - [x] 4.5.3 (v2) Integrate `ideator_agent` — use `retrieved_technique_hints` from arXiv search if stagnation triggered
  - [x] 4.5.4 Propose plan `p_k`; return `{refinement_plan: p_k}`

- [x] **4.6** Create `nodes/phase2_strategy_gate.py`:
  - [x] 4.6.1 Validate proposed strategy against `KNOWN_FAILED_STRATEGY_FINGERPRINTS` blacklist and small-data constraints
  - [x] 4.6.2 Accept or request re-plan

- [x] **4.7** Create `nodes/phase2_coder.py`:
  - [x] 4.7.1 Load-on-demand: full best pipeline script (state or `CKPT_BEST_PIPELINE` fallback), diagnosis, error analysis, `p_k`, FP/FN per-sample evidence (capped), population summary
  - [x] 4.7.2 LLM implements `p_k` as refined block `c_t^k`; surgical replacement: `s_t^k = s_t.replace(c_t, c_t^k)`
  - [x] 4.7.3 Pass through `code_validator`; return rejection reasons if fails
  - [x] 4.7.4 Return `{candidate_scripts: [s_t^k]}`

- [x] **4.8** Create `nodes/phase2_evaluator.py`:
  - [x] 4.8.1 Check validation cache (SHA-256 of script)
  - [x] 4.8.2 Run via `code_runner`; parse metrics with `metric_guard`; reject degenerate outputs
  - [x] 4.8.3 (v2) `curve_extrapolation` early abort — forecast final performance from epoch logs; abort if clearly below best
  - [x] 4.8.4 `is_acceptance_improvement` comparison; update `CKPT_BEST_PIPELINE` + `best_*` snapshot if improved
  - [x] 4.8.5 (v2) Persistent KB update — append `{tags, target_component, mechanism_class, outcome}` to `kb_semantic`
  - [x] 4.8.6 (v2) Stagnation → `ideator_agent.trigger_ideation`: arXiv search keyed to diagnosed failure mode; populate `retrieved_technique_hints`
  - [x] 4.8.7 Manage `inner_iteration`, `no_improve_count`; set `stop_outer_loop = True` when outer exit conditions met
  - [x] 4.8.8 Return `{latest_metrics, inner_iteration, no_improve_count, best_pipeline_script, best_*snapshot, stop_outer_loop}`

- [x] **4.9** Create routing functions:
  - [x] 4.9.1 `route_inner_loop` — "continue" → `phase2_error_analysis_gate` | "exit" → `phase2_outer_gate`; checks `INNER_LOOP_MAX` + `error_analysis_blocked` + early-stop signal
  - [x] 4.9.2 `route_outer_loop` — "continue" → `phase2_ablation` | "exit" → `phase3_ensemble_coder`; checks `stop_outer_loop` + `OUTER_LOOP_MAX` + patience caps + token budget

- [x] **4.10** Write unit tests for Phase 2:
  - [x] 4.10.1 Ablation variant runner (mock `code_runner`)
  - [x] 4.10.2 Error analysis gate state machine (all 3 branches)
  - [x] 4.10.3 `route_inner_loop` + `route_outer_loop` routing logic

---

## Component 5: Phase 3 — Ensemble

- [x] **5.1** Create `nodes/phase3_ensemble_coder.py`:
  - [x] 5.1.1 Load-on-demand: best pipeline script, Phase 1 candidate scores summary, ablation results, diagnosis, calibration stats, `tried_ensemble_approaches` history with fingerprints
  - [x] 5.1.2 Iteration 0: propose baseline `e_0` (simple averaging of all candidate models)
  - [x] 5.1.3 Subsequent iterations: propose `e_r` based on full `{strategy, score}` history; avoid previously-tried fingerprints
  - [x] 5.1.4 Pass through `code_validator`
  - [x] 5.1.5 Return `{ensemble_script, ensemble_strategy: {strategy_name, combination_method, strategy_fingerprint}}`

- [x] **5.2** Create `nodes/phase3_ensemble_evaluator.py`:
  - [x] 5.2.1 Check validation cache
  - [x] 5.2.2 Run script via `code_runner` + `metric_guard`
  - [x] 5.2.3 `is_acceptance_improvement`; update best if improved
  - [x] 5.2.4 Append to `tried_ensemble_approaches` with fingerprint; persist to `CKPT_TRIED_ENSEMBLE_APPROACHES`
  - [x] 5.2.5 Increment `ensemble_iteration`; set exit signal if cap hit or no-improvement
  - [x] 5.2.6 Return `{ensemble_models, latest_metrics, ensemble_iteration, tried_ensemble_approaches}`

- [x] **5.3** Create routing function `route_ensemble_loop` — "continue" → `phase3_ensemble_coder` | "exit" → `phase4_submit`

- [x] **5.4** Write unit test for Phase 3 (mock script execution, strategy fingerprint deduplication)

---

## Component 6: Phase 4 — Submission

- [x] **6.1** Create `nodes/phase4_submit.py`:
  - [x] 6.1.1 Lineage check — load `CKPT_SUBMISSION` directly if pipeline script SHA-256 matches
  - [x] 6.1.2 Run final best pipeline on **test split** (not val)
  - [x] 6.1.3 Parse metrics with `metric_guard`
  - [x] 6.1.4 Check both tiers: relaxed §9.1 + final §9.2 acceptance
  - [x] 6.1.5 Save `CKPT_SUBMISSION`; set `submission_passed`, `submission_report`

- [x] **6.2** Create `route_after_submit` — `END` (passed) | `phase2_ablation` (retry) | `END` (max retries)
- [x] **6.3** Implement retry reset logic (mirrors ADK `RetryLoopAgent._reset_for_retry`):
  - [x] 6.3.1 Archive Phase 2/3/4 checkpoints to `checkpoints/retry_archives/attempt_N/`
  - [x] 6.3.2 Reset loop counters: `outer_iteration`, `inner_iteration`, `ensemble_iteration`, `no_improve_count`, `stop_outer_loop`, `ensemble_no_improve_count`, token counter
  - [x] 6.3.3 **Preserve** `best_pipeline_script` + all `best_*` snapshot fields — start from best found, not L0
  - [x] 6.3.4 **Preserve** `tried_approaches` — planner avoids repeating failed strategies across retries
  - [x] 6.3.5 Do NOT re-run Phase 1 — data split and L0 skip-gated on disk

- [x] **6.4** Write unit test for Phase 4 acceptance logic (both tiers, edge cases) + retry reset (preserved vs cleared fields)

---

---

> **Note on the project summary vs actual code**: The ADK summary mentions `reflexion_agent`, `script_template.py`, `factorial_ablation.py`, and "Optuna HPO" — **none of these exist in the actual ADK file system**. The inner loop has no reflexion agent (it's `error_analysis_gate → planner → strategy_gate → coder → evaluator → error_analysis`). Ablation is 6 fixed named variants (not a factorial grid). `warm_restart.py` is a plateau-triggered optimizer/LR jolt, not Optuna. This plan tracks what's actually built.

---

## Component 7: Top-Level Graph Assembly

- [x] **7.1** Update `graph.py` — replace stub nodes with real imports from all node files
- [x] **7.2** Wire full topology per Components 3–6; `phase2_outer_gate` must be registered with `add_node`
- [x] **7.3** Add SQLite checkpointer — `build_graph()` must accept an optional `checkpointer` param and pass it to `g.compile(checkpointer=checkpointer)`:
  - Real runs: `SqliteSaver.from_conn_string("checkpoints/langgraph.db")` (package: `langgraph-checkpoint-sqlite`, import: `from langgraph.checkpoint.sqlite import SqliteSaver`)
  - Dry-run / tests: `MemorySaver()` (import: `from langgraph.checkpoint.memory import MemorySaver`)
  - Studio mode: `MemorySaver()` by default (Studio manages its own thread state)
  - ⚠️ `SqliteSaver` is **single-threaded only** — not safe for concurrent calls; fine for this sequential agent
- [x] **7.4** CLI entry point: `--dataset`, `--goal`, `--dry-run` flags; `--dry-run` sets `DRY_RUN=1` + uses `MemorySaver`; real runs use `SqliteSaver`; correct `initial_state` with all required fields
- [x] **7.5** Update `requirements.txt`:
  - Add `langgraph-checkpoint-sqlite` (for `SqliteSaver`)
  - Add `httpx>=0.27.0` (currently missing — used by LiteLLM)
  - Add `langgraph-cli[inmem]` (for LangGraph Studio dev server, installs `langgraph-cli` with in-memory mode, no Docker needed)
  - `litellm`, `openpyxl` already present ✓
  - Python ≥ 3.11 required by LangGraph Studio (project runs 3.13 ✓)

---

## Component 8: Integration Testing

- [x] **8.1** Graph compilation test — `build_graph()` succeeds without API key
- [x] **8.2** Dry-run smoke test — `DRY_RUN=1` end-to-end through all 4 phases in minutes
- [x] **8.3** Checkpoint resume test — kill after Phase 2 outer iteration 1, restart, verify counters and best-state are restored
- [x] **8.4** Full run on SUP046 dataset — validate relaxed §9.1 acceptance met

---

## Component 9: LangGraph Studio Setup

> LangGraph Studio is a **free** web-based visual debugger. No Docker needed with `inmem` mode.
> It connects to a local dev server you run with `langgraph dev`.
> Access URL: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`
> Requires a free LangSmith account (sign up at smith.langchain.com) for the API key.

- [x] **9.1** Create `langgraph.json` in project root — tells the CLI where to find the compiled graph:
  ```json
  {
    "dependencies": ["."],
    "graphs": {
      "mle_star": "./mle_star_agent/graph.py:build_graph"
    },
    "env": ".env"
  }
  ```
  Notes:
  - `graphs` value format is `path/to/file.py:callable` — callable must return a compiled graph
  - `build_graph()` must return `g.compile(...)`, not the `StateGraph` object (already does ✓)
  - Multiple graphs can be listed (e.g. add `"phase1_only"` for isolated debugging)

- [x] **9.2** Add Studio-required env vars to `.env`:
  - `LANGSMITH_API_KEY=<your-key>` (get from smith.langchain.com → Settings → API Keys)
  - Optionally: `LANGSMITH_TRACING=false` to disable telemetry while keeping Studio functional

- [x] **9.3** Verify `build_graph()` works for Studio:
  - Studio calls `build_graph()` with no arguments — ensure signature has safe defaults: `def build_graph(checkpointer=None)`
  - When `checkpointer is None`, use `MemorySaver()` so Studio gets persistent thread state within a session
  - Studio passes its own `thread_id` via `config={"configurable": {"thread_id": "..."}}` — the checkpointer enables this

- [x] **9.4** Workflow to run Studio locally:
  ```bash
  pip install --upgrade "langgraph-cli[inmem]"
  langgraph dev          # starts server at http://127.0.0.1:2024
  # Then open: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
  ```
  - Safari users: use `langgraph dev --tunnel` (Safari blocks localhost cross-origin)
  - Hot-reload is on by default — edit nodes and re-invoke without restarting
  - Each "Run" in Studio creates a new thread; use "Threads" panel to inspect state at each step
