#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --time=23:59:00
#SBATCH --job-name=dino_rn50
#SBATCH --array=0-4 # inclusive 
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/b268dc/b268dc10/logs/ed-actionpose/%x_%j_%a
#SBATCH --error=/home/atuin/b268dc/b268dc10/logs/ed-actionpose/%x_%j_%a

GPUS=1
CONFIG=configs/gesture_detection_crossval_rn50.py

# RUN FROM PROJECT ROOT


set -e 

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=https://proxy:80

readonly GROUP=$(id -gn)

# COPY CODE DIRECTORY TO COMPUTE NODE
readonly CODE_SOURCE=/home/hpc/$GROUP/$USER/work/code/SensoryGestureRecognition/baselines/mm
readonly TARGET_PATH=${TMPDIR}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

mkdir -p "${TARGET_PATH}"

echo "[$(date)] Copying code from ${CODE_SOURCE} to ${TARGET_PATH}"
cp -r ${CODE_SOURCE} ${TARGET_PATH}
cd ${TARGET_PATH}/mm
echo "[$(date)] Code successfully copied."

# COPY DATA TO COMPUTE NODE
readonly SOURCE_DATA=/home/atuin/$GROUP/$USER/data/sensoryart/crossval/fold${SLURM_ARRAY_TASK_ID}.tar
readonly TARGET_DATA=${TARGET_PATH}/mm/data 

mkdir -p "${TARGET_DATA}"
echo "[$(date)] Copying data from ${SOURCE_DATA} to ${TARGET_DATA}"
tar xf ${SOURCE_DATA} -C ${TARGET_DATA} --strip-components=1 # remove outer foldX directory to comply with mmdetection config expectations
echo "[$(date)] Data successfully copied."

source "/home/atuin/${GROUP}/${USER}/venvs/sensoryart/bin/activate"
# patch the missing info field in the annotation jsons
python - <<'PY'
import json, sys, time
paths = [
    "data/annotations/person_keypoints_val2017.json",
    "data/annotations/person_keypoints_test2017.json",
]
for p in paths:
    with open(p, "r") as f:
        d = json.load(f)
    if "info" not in d:
        d["info"] = {
            "description": "SensoryArt crossval",
            "version": "1.0",
            "year": 2025,
            "date_created": time.strftime("%Y-%m-%d")
        }
    if "licenses" not in d:
        d["licenses"] = []
    with open(p, "w") as f:
        json.dump(d, f)
    print(f"Patched {p}")
PY


# TRAINING
readonly WORK_DIR="${TARGET_PATH}/work_dir"
mkdir -p "${WORK_DIR}"

# Save console output to workdir
LOG_FILE="${WORK_DIR}/console_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging console output to: $LOG_FILE"


# run training, capture exit code (so set -e doesn't kill the script)
set +e
export PORT=$((29501+${SLURM_ARRAY_TASK_ID}))
./tools/dist_train_hpc.sh "${CONFIG}" "${GPUS}" --work-dir "${WORK_DIR}"
train_rc=$?
set -e
echo "dist_train_hpc.sh exited with code: ${train_rc}"



if (( train_rc != 0 )); then
  echo "Training failed (rc=${train_rc}). Skipping test and continuing to copy artifacts."
else
  # --- TESTING ---
  echo "[$(date)] Starting test..."

  set +e
  # Resolve checkpoint from last_checkpoint (it contains a path)
  if [[ -f "${WORK_DIR}/last_checkpoint" ]]; then
    read -r CKPT < "${WORK_DIR}/last_checkpoint"
    [[ "$CKPT" != /* ]] && CKPT="${WORK_DIR}/${CKPT}"
    echo "Resolved checkpoint from last_checkpoint: $CKPT"
  else
    echo "No last_checkpoint file in ${WORK_DIR}" 
    CKPT=""
  fi

  if [[ -n "$CKPT" && -f "$CKPT" ]]; then
    echo "Using checkpoint: ${CKPT}"
    TEST_OUT="${WORK_DIR}/preds/test"
    mkdir -p "$(dirname "$TEST_OUT")"

    # generate predictions
    python -u test.py "$CONFIG" "$CKPT" \
      --launcher none \
      --work-dir "$WORK_DIR" \
      --cfg-options test_evaluator.format_only=True test_evaluator.outfile_prefix="$TEST_OUT"
    echo "[$(date)] Test finished. Predictions under: $(dirname "$TEST_OUT")"

    # generate metrics
    python -u test.py "$CONFIG" "$CKPT" \
      --launcher none \
      --work-dir "$WORK_DIR" 
    echo "[$(date)] Test finished."


  else
    echo "Resolved checkpoint missing: $CKPT. Skipping test."
  fi
  set -e
fi

# COPY OUTPUT TO $WORK
TARGET_WORKDIR="$WORK/work_dirs/dino_rn50/${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TARGET_WORKDIR}"
cat "$0" > ${TARGET_WORKDIR}/slurm.sh # copy this file to workdir
cp -r ${WORK_DIR} ${TARGET_WORKDIR}
