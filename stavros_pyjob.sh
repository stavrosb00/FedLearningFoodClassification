#!/bin/bash

#$ -cwd
#$ -q cuda.q
#$ -S /bin/bash
#$ -shell y
#$ -j y
#$ -V

## Export these environmental variables
## -v PVM_ROOT,FOOBAR=BAR

## to run I want 6 or more parallel processes
## under the PE pvm. the processes require
## 128M of memory
## -pe pvm 6- -l mem=128

## Send mail to these users
#$ -M stavroscallmaster@yahoo.gr
## Mail at beginning/end/on suspension
#$ -m es

## Change the following vars if you want
## -o stdout.log
## -e stderr.log

# Setup to use the currently idle GPU
gpu0util="$(nvidia-smi -q -i 0 | grep Utilization -A 1 | grep Gpu | tail -c 4 | head -c 1)"
gpu1util="$(nvidia-smi -q -i 1 | grep Utilization -A 1 | grep Gpu | tail -c 4 | head -c 1)"

if [ $gpu0util -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

if [ $gpu1util -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=1
fi

echo Date: `date`
echo User: `whoami`
echo Job : $0
echo Args: $@
echo pwd : `pwd`
echo "--------"

# Ensure conda is available
source /home/stavrosmpoul/miniconda3/etc/profile.d/conda.sh

echo conda : `which conda`

# Activate the conda environment
conda activate pyt_flwr
echo python: `which python`
echo "--------"

# Run the experiment: extension 50 rounds sta 150 xwris warm_up kai me conflict sto strategy LR scheduler
python main_ssfl.py partitioning=iid alpha=0.5 num_clients=5 num_classes=10 C_fraction=1 strategy=fedsimsiam optimizer=fedsimsiam num_workers=4 val_ratio=0 rad_ratio=0.02 local_epochs=5 num_rounds=50 warm_up_rounds=0 cos_decay=True batch_size=128 pretrained=False warm_start=True

exit

### echo conda : `which conda`
### conda : /home/stavrosmpoul/miniconda3/condabin/conda

### $USER/miniconda3/envs/pyt_flwr/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games
### stavrosmpoul/miniconda3/envs/pyt_flwr/bin/python
### # export PATH=$USER/Programs/miniconda3/envs/tensorflow/bin:$PATH <--- OFFICIAL
### /home/
### :/home/stavrosmpoul/miniconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games
#### export PATH=stavrosmpoul/miniconda3/envs/pyt_flwr/bin/python
#######PATH=stavrosmpoul/miniconda3/envs/pyt_flwr/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games
### python main_ssfl.py partitioning=iid num_clients=5 num_classes=10 C_fraction=1 strategy=fedsimsiam optimizer=fedsimsiam num_workers=4 val_ratio=0 rad_ratio=0.02 local_epochs=5 num_rounds=1 warm_up_rounds=10 cos_decay=True batch_size=128 pretrained=False