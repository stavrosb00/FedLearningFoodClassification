#! /bin/bash

#$ -cwd
#$ -q main
#$ -p 0
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

## request a single CUDA device
#$ -l cudadev=1

## Send mail to these users
## -M stavrousmpoul@mug6.ee.auth.gr
## Mail at beginning/end/on suspension
#$ -m bes

## Change the following vars
## -o file_for_output_log_here 
## -e file_for_error_log_here 

echo "Date: `date`"
echo "User: `whoami`"
echo "Job : $0"
echo "Args: $@"
echo "pwd : `pwd`"
echo "--------"
export PATH=~/Programs/miniconda3/envs/tf-gpu/bin:${PATH}  # NOTE - 'tf-gpu' is the conda env
echo "PATH=${PATH}"
echo "python: `which python`"
echo "--------"
echo " "

source activate pyt_pg
which python
echo " "

#python main.py strategy=scaffold optimizer=scaffold num_rounds=1 num_classes=4
python testScript.py $@

exit