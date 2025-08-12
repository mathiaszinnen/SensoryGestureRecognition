#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --job-name=sensoryart_crossval
#SBATCH --array=0 # 
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/b268dc/b268dc10/logs/ed-actionpose/%x_%j_%a
#SBATCH --error=/home/atuin/b268dc/b268dc10/logs/ed-actionpose/%x_%j_%a


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

GPUS=1
CONFIG=configs/gesture_detection_crossval.py

./tools/dist_train_hpc.sh ${CONFIG} ${GPUS} --work-dir ${WORK_DIR}

# Resolve checkpoint from last_checkpoint (it contains a path)
if [[ -f "${WORK_DIR}/last_checkpoint" ]]; then
  read -r CKPT < "${WORK_DIR}/last_checkpoint"   # read single line, strip newline
  [[ "$CKPT" != /* ]] && CKPT="${WORK_DIR}/${CKPT}"  # make absolute if relative
else
  echo "No last_checkpoint file in ${WORK_DIR}" >&2
  exit 1
fi

echo "Using checkpoint: ${CKPT}"

# Run test (single GPU)
TEST_OUT="${WORK_DIR}/preds"
python test.py $CONFIG $CKPT \
    --work-dir "$WORK_DIR" \
    --cfg-options test_evaluator.format_only=True test_evaluator.outfile_prefix=$OUT_FILE
echo "[$(date)] Test finished. Predictions: ${TEST_OUT}"

# COPY OUTPUT TO $WORK
TARGET_WORKDIR="$WORK/work_dirs/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TARGET_WORKDIR}"
cat "$0" > ${TARGET_WORKDIR}/slurm.sh # copy this file to workdir
cp -r ${WORK_DIR} ${TARGET_WORKDIR}

