#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --job-name=bo_sgld_inp

# Log compute-node infos into slurm log
echo '***********************************'
echo "Job placed on:" $HOSTNAME
echo '***********************************'
echo "GPU infos"
#lspci -v | grep VGA
nvidia-smi
echo '***********************************'
echo 'CPU infos'
lscpu | grep 'Model name'
lscpu | grep 'Architecture'
cat /proc/meminfo | grep 'MemTotal'
cat /proc/meminfo | grep 'MemFree'
echo '***********************************'

# Load python and cuda modules
echo '***********************************'
module load devel/python_intel/3.9
module load lib/cudnn/7.6.5-cuda-10.1
module list
echo '***********************************'

echo '***********************************'
echo 'Slurm environment:'
printenv | grep SLURM_JOB_NAME
printenv | grep SLURM_JOB_GPUS
printenv | grep SLURM_TASKS_PER_NODE
printenv | grep SLURM_JOB_ID
printenv | grep SLURM_JOB_USER
printenv | grep SLURM_SUBMIT_DIR
printenv | grep SLURM_NPROCS
printenv | grep SLURM_CPUS_ON_NODE
printenv | grep SLURM_JOB_PARTITION
printenv | grep SLURM_MEM_PER_NODE
echo '***********************************'

# Define your own parameters
SDS_DIR=/mnt/sds-hd/sd20i001
WORKSPACE_DIR=/gpfs/bwfor/work/ws/hd_cn265-aicm

USER=malte
PATH_TO_CODE=code/mfvi-dip-mia
# PATH_TO_CONFIG=code/mfvi-dip-mia/bo_configs/bo_mfvi
PATH_TO_LOGS=logs

TASK=denoising
BAYES=sgld
CONFIG=./bo_configs/bo_sgld.json

CONDA_ENV_NAME=mfvidip


# orchestrate the paths
# Use the standard slurm log and a separate job-log
# SDS_DATA=${SDS_DIR}/${USER}/${PATH_TO_DATA}
WORKSPACE_DATA=${WORKSPACE_DIR}/${SLURM_JOB_NAME}
SDS_CODE=${SDS_DIR}/${USER}/${PATH_TO_CODE}
SDS_CONFIG=${SDS_DIR}/${USER}/${PATH_TO_CONFIG}
SDS_LOG=${SDS_DIR}/${USER}/${PATH_TO_LOGS}/${SLURM_JOB_NAME}
SDS_LOG_FILE=${SDS_LOG}/${SLURM_JOB_ID}.log

# Define the path to the entry script
PATH_TO_START_SCRIPT=${SDS_CODE}/bayesian_optimization.py

#############################################

# echo 'data dir:' ${SDS_DATA}
echo 'code/working dir:' ${SDS_CODE}
echo 'configs dir:' ${SDS_CONFIG}
echo 'log dir:' ${SDS_LOG}

# create folders if necessary
mkdir -p ${SDS_LOG}

# Copy source data into the working directory
# needs to be tested, this should be faster.
#cp -r ${SRC_DIR} ${SRC_TARGET_DIR}

# Make sure conda is initialised
# activate a virtual environment
# could be crated on the login nodes via: "conda create -f env.yml"
conda init bash
source ~/.bash_profile
conda activate ${CONDA_ENV_NAME}

echo '***********************************'
echo 'Show Python version and available libraries'
# individual checks of some versions and modules
python --version
which python
which pip
#pip list
pip list | grep tensorflow
cd ${SDS_CODE}
pwd

# Call here your python scripts, write output to job-log
python ${PATH_TO_START_SCRIPT} --task ${TASK} --bayes ${BAYES} --config ${CONFIG} >> ${SDS_LOG_FILE}
echo 'python script executed'
