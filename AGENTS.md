# AGENTS.md - br-modified worktree

## Current role

D:\Code\RL_Pricing\RL_pricing_modified is the active worktree for the bounded-rationality extension.

Use this worktree for:

- BR mechanism implementation
- algorithm-side debugging
- training and testing the modified model
- process audit and visual trace generation
- short smoke runs and formal Modified experiments

Do not write new work into the legacy D:\Code\RL_Pricing\RL_pricing\Modified directory.

## Git context

- Feature branch: codex/joint-price-compensation, based on codex/visualization-hooks at 2424174.
- The joint-pay-contract, joint-pay-td3, joint-pay-training and joint-pay-interfaces branches are the preceding review stages; preserve them until the user decides how to merge the PR stack.
- Pre-review local history is retained on br-modified.
- Shared bare repository: D:\Code\RL_Pricing\RL_pricing_worktrees.git
- Baseline counterpart: D:\Code\RL_Pricing\RL_pricing_baseline
- Tooling repository: D:\Code\RL_Pricing\RL_pricing_tools
- Maintained documentation: D:\Code\RL_Pricing\RL文档数据库

## Hard boundaries

1. Do not modify D:\Code\RL_Pricing\RL_pricing_baseline from this worktree.
2. Do not write to old D:\Code\RL_Pricing\RL_pricing\Origin or D:\Code\RL_Pricing\RL_pricing\Modified.
3. Do not commit generated outputs: results/, runs/, logs/, model weights, .csv, .npy, .npz, .pkl, .pt, .pth, .pyc, .log.
4. Keep training and testing outputs ignored by Git.
5. In paper-facing text use bounded-rationality extension, bounded-rationality model, or BR.
6. Put maintained project documentation in D:\Code\RL_Pricing\RL文档数据库, not in this worktree.
7. All new source, experiments, exports, and temporary files must stay under D:\Code\RL_Pricing. Never recreate a project under a C-drive user profile. Use D:\Code\RL_Pricing\tmp for TEMP/TMP/TMPDIR. Installed tools may reside elsewhere.
8. Keep upstream input arrays under grid_small_static/4/4 and grid_small_dynamic/4/4 versioned. The independent UI source remains in RL_pricing_tools; this worktree owns its simulator trace interfaces.

## Current bounded rationality model

New results use `model_family: bounded_rationality` and numeric `schema_version: 2`.

Describe the current implementation by listing its active mechanisms:

1. Passenger entry into the system.
2. Driver online / active supply.

Optional TD3 joint control is enabled with `--action_mode joint`; the default remains `price_only`. Joint actions contain separate price and order-compensation vectors. Result schema stays 2; `action_space` records action protocol 1 and joint state version 2. Joint observations retain the full waiting-age price and compensation histories. Maintained design, interface and acceptance notes live in the vault, including `20-模型/TD3价格与报酬联合决策.md`.

## Algorithm validation commands

    cd D:\Code\RL_Pricing\RL_pricing_modified

    conda run -n rl_pricing_py38 python -m py_compile `
      main.py runner.py sim.py center.py model.py pricer.py `
      process_audit.py visual_trace.py tools\analyze_process_accuracy.py

    conda run -n rl_pricing_py38 python main.py --help
    conda run -n rl_pricing_py38 python -m unittest discover -s tests -v
    git status --short

## Experiment comparison rules

When comparing baseline and modified results, keep the following identical:

- data folder
- pricing algorithm
- epoch count
- seed
- batch size
- actor and critic learning rates
- kernel size
- searching mode
- policy delay
- position encoding
- other relevant hyperparameters

The only experimental difference should be that the modified worktree enables the bounded-rationality model.

For joint-control comparisons, also record action_mode, compensation bounds, action/state versions and learner scheduling. The new joint runner starts learning at batch size, unlike the legacy runner's full-period replay threshold. Engineering acceptance is not a controlled profit-improvement experiment; use matched training schedules before claiming an action-space effect.

Single-seed or one-epoch runs are pipeline checks only. They are not research conclusions.
