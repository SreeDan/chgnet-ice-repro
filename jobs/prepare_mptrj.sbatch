#!/bin/bash
#SBATCH --job-name=chgnet-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mptrj-prep-%j.out
#SBATCH --error=logs/mptrj-prep-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

# Set this to persistent scratch/shared storage before calling sbatch.
: "${CHGNET_SCRATCH_DIR:?Export CHGNET_SCRATCH_DIR to your allocated scratch directory before submitting}"
case "$CHGNET_SCRATCH_DIR" in
  /*) ;;
  *) echo "CHGNET_SCRATCH_DIR must be an absolute path" >&2; exit 2 ;;
esac
mkdir -p "$CHGNET_SCRATCH_DIR" inputs

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate chgnet-repro

echo "Starting MPtrj preparation"
date -u

python scripts/download_mptrj.py \
  --output "$CHGNET_SCRATCH_DIR/mptrj_source"
cp "$CHGNET_SCRATCH_DIR/mptrj_source.manifest.json" \
  docs/mptrj_source_manifest.json

python scripts/make_subset.py \
  --input "$CHGNET_SCRATCH_DIR/mptrj_source" \
  --output inputs/mptrj_subset.json \
  --count 24 \
  --seed 42 \
  --min-atoms 2 \
  --max-atoms 64

sha256sum inputs/mptrj_subset.json \
  > inputs/mptrj_subset.sha256

echo "Data preparation completed"
date -u
