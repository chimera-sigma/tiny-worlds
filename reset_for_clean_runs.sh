#!/bin/bash
# Reset for Clean Experimental Runs
# Archives old messy logs and prepares for canonical 20-seed experiments

set -e

echo "============================================"
echo "Resetting for Clean Experimental Runs"
echo "============================================"

# Create archive directory with timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
archiveDir="./experiments/results/archive_$timestamp"

echo -e "\nCreating archive directory: $archiveDir"
mkdir -p "$archiveDir"

# Archive old summary.jsonl if it exists
if [ -f "./experiments/results/summary.jsonl" ]; then
    echo "Archiving old summary.jsonl..."
    mv "./experiments/results/summary.jsonl" "$archiveDir/summary.jsonl"
    echo "  -> Moved to $archiveDir/summary.jsonl"
else
    echo "No summary.jsonl to archive (already clean)"
fi

# Archive old aggregated_results.csv if it exists
if [ -f "./experiments/results/aggregated_results.csv" ]; then
    echo "Archiving old aggregated_results.csv..."
    mv "./experiments/results/aggregated_results.csv" "$archiveDir/aggregated_results.csv"
    echo "  -> Moved to $archiveDir/aggregated_results.csv"
else
    echo "No aggregated_results.csv to archive (already clean)"
fi

# Archive old runs directory if it exists and has content
if [ -d "./experiments/results/runs" ] && [ "$(ls -A ./experiments/results/runs)" ]; then
    echo "Archiving old per-run JSON files..."
    mv "./experiments/results/runs" "$archiveDir/runs"
    echo "  -> Moved to $archiveDir/runs"
    # Recreate empty runs directory
    mkdir -p "./experiments/results/runs"
else
    echo "No old runs to archive"
    mkdir -p "./experiments/results/runs"
fi

# Archive old figures directory if it exists
if [ -d "./experiments/results/figures" ] && [ "$(ls -A ./experiments/results/figures)" ]; then
    echo "Archiving old figures..."
    mv "./experiments/results/figures" "$archiveDir/figures"
    echo "  -> Moved to $archiveDir/figures"
    # Recreate empty figures directory
    mkdir -p "./experiments/results/figures"
else
    echo "No old figures to archive"
    mkdir -p "./experiments/results/figures"
fi

echo -e "\n============================================"
echo "Reset Complete! Ready for Clean Runs"
echo "============================================"

echo -e "\nOld data archived to:"
echo "  $archiveDir"

echo -e "\nFresh state:"
if [ -f "./experiments/results/summary.jsonl" ]; then
    echo "  - summary.jsonl: EXISTS (unexpected!)"
else
    echo "  - summary.jsonl: CLEAN"
fi

if [ -f "./experiments/results/aggregated_results.csv" ]; then
    echo "  - aggregated_results.csv: EXISTS (unexpected!)"
else
    echo "  - aggregated_results.csv: CLEAN"
fi

if [ -d "./experiments/results/runs" ] && [ ! "$(ls -A ./experiments/results/runs)" ]; then
    echo "  - runs directory: EMPTY"
else
    echo "  - runs directory: READY"
fi

echo -e "\nNext steps:"
echo "  1. Run clean experiments:"
echo "     ./run_clean_experiments.sh"
echo "  2. Aggregate results:"
echo "     python scripts/aggregate_results.py"
echo "  3. Generate figures:"
echo "     python scripts/analyze_results.py"
echo "  4. Compile paper:"
echo "     cd paper && ./compile.sh"

echo -e "\n============================================"
