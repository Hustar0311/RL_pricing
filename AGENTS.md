# AGENTS.md - compatibility baseline worktree

## Current role

D:\Code\RL_Pricing\RL_pricing_baseline is the compatibility baseline worktree. It retains the upstream algorithm behavior while carrying the minimum compatibility fixes needed by the current environment.

Use this worktree for:

- reproducing baseline experiments
- running baseline smoke tests
- generating baseline comparison results
- checking original-code behavior against the modified worktree

This worktree is not the place to develop bounded rationality mechanisms.

## Git context

- Branch: codex/baseline-compatibility
- Original local history is retained on baseline-original; upstream main is 96d0ffa.
- Shared bare repository: D:\Code\RL_Pricing\RL_pricing_worktrees.git
- Modified counterpart: D:\Code\RL_Pricing\RL_pricing_modified
- Tooling repository: D:\Code\RL_Pricing\RL_pricing_tools
- Maintained documentation: D:\Code\RL_Pricing\RL文档数据库

## Hard boundaries

1. Treat this worktree as the baseline reference.
2. Do not develop BR logic here.
3. Do not modify algorithm logic here unless the user explicitly approves a baseline fix.
4. Do not write to old D:\Code\RL_Pricing\RL_pricing\Origin.
5. Do not commit generated outputs: results/, runs/, logs/, model weights, .csv, .npy, .npz, .pkl, .pt, .pth, .pyc, .log.
6. Keep training and testing outputs ignored by Git.
7. Put maintained project documentation in D:\Code\RL_Pricing\RL文档数据库, not in this worktree.
8. All new source, experiments, exports, and temporary files must stay under D:\Code\RL_Pricing. Never recreate a project under a C-drive user profile. Use D:\Code\RL_Pricing\tmp for TEMP/TMP/TMPDIR. Installed tools may reside elsewhere.
9. Keep the upstream input arrays under grid_small_static/4/4 and grid_small_dynamic/4/4 versioned; they are inputs, not generated experiment outputs.

## Baseline validation commands

    cd D:\Code\RL_Pricing\RL_pricing_baseline

    conda run -n rl_pricing_py38 python -m py_compile `
      main.py runner.py sim.py center.py model.py pricer.py

    conda run -n rl_pricing_py38 python main.py --help
    git status --short

## Comparison rules

Baseline results should only be compared with modified results when the data, algorithm, seed, epoch count, batch size, learning rates, and other relevant hyperparameters match.

Single-seed or one-epoch runs are pipeline checks only. They are not research conclusions.
