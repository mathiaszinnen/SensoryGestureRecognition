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

# TRAINING
readonly WORK_DIR="${TARGET_PATH}/work_dir"
mkdir -p "${WORK_DIR}"

echo "CURRENT WORKING DIRECTORY CONTENTS: $(ls .)"

GPUS=1
CONFIG=configs/gesture_detection_crossval.py

./tools/dist_train_hpc.sh ${CONFIG} ${GPUS} --work-dir ${WORK_DIR}


CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


# COPY OUTPUT TO $WORK
TARGET_WORKDIR="$WORK/work_dirs/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TARGET_WORKDIR}"
cat "$0" > ${TARGET_WORKDIR}/slurm.sh # copy this file to workdir
cp -r ${WORK_DIR} ${TARGET_WORKDIR}
