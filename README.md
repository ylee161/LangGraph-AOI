# LangGraph AOI — MLE-STAR Agent

An AOI (Automated Optical Inspection) agent built with **LangGraph** that implements the **MLE-STAR loop**: given a dataset and a goal metric, it autonomously trains models, evaluates them, diagnoses failures, and iterates until the goal is hit.

> Migrated from the Google ADK implementation in `../AOI agent ` — same loop logic, new graph-based orchestration.

## Project Layout

```
LangGraph AOI/
├── mle_star_agent/          # Main agent package (to be built)
│   ├── graph.py             # LangGraph StateGraph definition
│   ├── state.py             # AgentState TypedDict
│   ├── nodes/               # One file per graph node
│   │   ├── phase1_init.py
│   │   ├── phase2_refine.py
│   │   ├── phase3_ensemble.py
│   │   └── phase4_submit.py
│   ├── tools/               # Tool functions callable by nodes
│   ├── config.py            # Hyper-params, paths, thresholds
│   └── __init__.py
├── checkpoints/             # Persistent run state (JSON)
├── MLE-STAR-paper.pdf       # → symlink to reference paper
├── MLE-STAR-paper-summary.md
├── dataset_SUP046_lot1/     # → symlink to lot data
├── dataset_SUP046_lot2/
├── dataset_SUP046_lot3/
├── .env.example             # → symlink to env template
├── .env                     # your secrets (git-ignored)
└── requirements.txt
```

## Quick Start

```bash
# 1. Create & activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env      # then fill in DEEPSEEK_API_KEY etc.

# 4. Run the agent (entry point TBD)
python -m mle_star_agent.graph
```

## MLE-STAR Loop (high level)

```
Phase 1  →  Initialise: data split, baseline script
     ↓
Phase 2  →  Refine: diagnose errors, generate new training scripts
     ↓  (loop until goal or patience exhausted)
Phase 3  →  Ensemble: search over candidate models
     ↓
Phase 4  →  Submit: evaluate vs acceptance criteria
     ↑  (retry loop if criteria not met)
```

## Reference

- **MLE-STAR paper**: [MLE-STAR-paper.pdf](./MLE-STAR-paper.pdf)
- **ADK reference implementation**: `../AOI agent /mle_star_agent/`
