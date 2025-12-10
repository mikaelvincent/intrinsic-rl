#!/bin/bash
# Generate key figures for the paper (Figures 1a/1b and ablation/mechanism plots).
# Assumes 'python -m irl.experiments full' has already been run to populate runs_suite.

set -e

# Default output directory for final paper-ready figures
RESULTS_DIR="paper_results"
RUNS_DIR="runs_suite"

echo "Generating Figure 1a (Extrinsic Return) and 1b (Total Return)..."
python -m irl.experiments plots \
    --runs-root "$RUNS_DIR" \
    --results-dir "$RESULTS_DIR" \
    --metric reward_mean \
    --metric reward_total_mean \
    --smooth 10 \
    --shade

echo "Generating Figure 3 (Ablation Study - Extrinsic)..."
# Filter to ablations if possible, or rely on overlay picking them up if they exist in runs_suite.
# This re-runs plotting specifically for ablation analysis if specific configs were named appropriately.
# Since irl.experiments plots generates overlays for *all* found envs/methods, the main command above
# effectively covers Figure 3 as well if the ablation runs (Proposed_NoGate etc.) are in runs_suite.

echo "Generating Figure 4 (Gating Dynamics)..."
# Extract gating dynamics from a representative Proposed run.
# This requires a more specific call to irl.plot curves targeting a single run.
# Find a proposed run for a complex env (e.g. Ant or HalfCheetah).
PROPOSED_RUN=$(find "$RUNS_DIR" -maxdepth 1 -name "proposed__Ant-v5__seed*" | head -n 1)

if [ -z "$PROPOSED_RUN" ]; then
    echo "No Proposed run found for Ant-v5. Skipping Figure 4 generation."
else
    echo "Using run: $PROPOSED_RUN"
    python -m irl.plot curves \
        --runs "$PROPOSED_RUN" \
        --metric gate_rate_pct \
        --smooth 5 \
        --out "$RESULTS_DIR/fig4_gating_dynamics_ant.png" \
        --label "Gate Rate (%)"
fi

echo "Done. Plots saved to $RESULTS_DIR/plots and $RESULTS_DIR."
