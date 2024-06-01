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
#$ -M smpoulio@ece.auth.gr
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

echo conda : `which conda`
# export PATH=stavrosmpoul/miniconda3/envs/pyt_flwr/bin:/home/stavrosmpoul/miniconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games
## export PATH=$USER/miniconda3/envs/pyt_flwr/bin:$PATH
conda activate pyt_flwr
echo python: `which python`
echo "--------"

# run the experiment
python main_ssfl.py partitioning=iid num_clients=10 num_classes=10 C_fraction=1 strategy=fedsimsiam optimizer=fedsimsiam num_workers=4 val_ratio=0 rad_ratio=0.02 local_epochs=5 num_rounds=200 warm_up_rounds=10 cos_decay=True batch_size=128 pretrained=False
exit