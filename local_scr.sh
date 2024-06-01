#! /bin/bash
#scripts for desktop
echo "Running locally FL simulation experiment"

# python testScript.py
# dataset split-Visuals
# python -m main --multirun optimizer=momentum strategy=fedavg partitioning=dirichlet,iid num_clients=10,16,32 num_classes=4,10,30
# python -m main --multirun optimizer=momentum strategy=fedavg partitioning=iid balance=True num_clients=10,16,32 num_classes=4,10,30
# scaffold , dataframes k subdirs
# python -m main --multirun partitioning=iid,dirichlet num_rounds=5 num_clients=16 num_clients_per_round_fit=8 strategy=scaffold optimizer=scaffold
conda activate pyt_flwr
python -m main --multirun strategy=scaffold optimizer=scaffold partitioning=dirichlet num_rounds=100 num_classes=4,10 num_clients=10 C_fraction=0.5
python main.py strategy=scaffold optimizer=scaffold partitioning=dirichlet num_rounds=100 num_classes=10 num_clients=10 C_fraction=0.5
python main.py strategy=scaffold optimizer=scaffold partitioning=iid num_rounds=100 num_classes=10 num_clients=10 C_fraction=0.5
# python main.py strategy=fedavg optimizer=proximal partitioning=dirichlet num_rounds=100 num_classes=10 num_clients=10 C_fraction=0.5
# python main.py strategy=scaffold optimizer=scaffold partitioning=dirichlet num_rounds=100 num_classes=10 num_clients=10 C_fraction=0.5

# python main.py partitioning=dirichlet num_rounds=1 num_clients=16 C_fraction=0.5 num_clients_per_round_fit=8 strategy=scaffold optimizer=scaffold
# python main.py strategy=scaffold optimizer=scaffold num_rounds=1 num_classes=4

# python main.py partitioning=dirichlet num_clients=32 num_classes=4
# python main.py partitioning=dirichlet num_clients=16 num_classes=4


# SSL k linear eval 
# Downstream training - linear evaluation : python .\evaluation_linear.py num_rounds=50 batch_size=256 val_ratio=0 optimizer=adam num_workers=0
# SSFL 

python main_ssfl.py partitioning=dirichlet alpha=0.5 num_clients=10 num_classes=10 C_fraction=0.2 strategy=heterossfl optimizer=heterossfl num_workers=0 val_ratio=0 rad_ratio=0.02 local_epochs=5 num_rounds=2

# xwris cos decay gia early exps me c_fraction xamhlo
python main_ssfl.py partitioning=dirichlet alpha=0.5 num_clients=10 num_classes=10 C_fraction=0.2 strategy=heterossfl optimizer=heterossfl num_workers=0 val_ratio=0 rad_ratio=0.02 local_epochs=5 num_rounds=3 cos_decay=False

# stis outputs\2024-05-14\12-02-59
python main_ssfl.py partitioning=dirichlet alpha=0.5 num_clients=10 num_classes=10 C_fraction=1 strategy=heterossfl optimizer=heterossfl num_workers=0 val_ratio=0 rad_ratio=0.02 local_epochs=5 num_rounds=5 cos_decay=False
# bs=128 maybe
python main_ssfl.py partitioning=dirichlet alpha=0.5 num_clients=10 num_classes=10 C_fraction=1 strategy=heterossfl optimizer=heterossfl num_workers=0 val_ratio=0 rad_ratio=0.02 local_epochs=10 num_rounds=4 cos_decay=False batch_size=128

# Centr SimSiam 
python .\centralised_simsiam.py num_rounds=200 batch_size=128 val_ratio=0 optimizer=simsiam num_workers=2 optimizer.lr=0.03

## FedSimSiam 
# Long
python main_ssfl.py partitioning=iid alpha=0.5 num_clients=5 num_classes=10 C_fraction=1 strategy=fedsimsiam optimizer=fedsimsiam num_workers=4 val_ratio=0 rad_ratio=0.02 local_epochs=5 num_rounds=150 warm_up_rounds=10 cos_decay=True batch_size=128 pretrained=False

python main_ssfl.py partitioning=iid alpha=0.5 num_clients=5 num_classes=10 C_fraction=1 strategy=fedsimsiam optimizer=fedsimsiam num_workers=0 val_ratio=0 rad_ratio=0.02 local_epochs=2 num_rounds=10 cos_decay=True batch_size=128
#Debug me IID kai short rounds
python main_ssfl.py partitioning=iid alpha=0.5 num_clients=5 num_classes=10 C_fraction=0.2 strategy=fedsimsiam optimizer=fedsimsiam num_workers=0 val_ratio=0 rad_ratio=0.02 local_epochs=2 num_rounds=1 cos_decay=True batch_size=128

# Have been run
# 24/5 prwi 9-40-56 me meshmeri
python main_ssfl.py partitioning=iid alpha=0.5 num_clients=5 num_classes=10 C_fraction=1 strategy=fedsimsiam optimizer=fedsimsiam num_workers=4 val_ratio=0 rad_ratio=0.02 local_epochs=5 num_rounds=150 warm_up_rounds=10 cos_decay=True batch_size=128 pretrained=False
# 24/5 apo 22-04 to brady
python main_ssfl.py partitioning=iid alpha=0.5 num_clients=5 num_classes=10 C_fraction=1 strategy=fedsimsiam optimizer=fedsimsiam num_workers=4 val_ratio=0 rad_ratio=0.02 local_epochs=2 num_rounds=150 warm_up_rounds=10 cos_decay=True batch_size=128 pretrained=False
# 25/5 apo 11-33 prwi.
python main_ssfl.py partitioning=dirichlet alpha=0.5 num_clients=5 num_classes=10 C_fraction=1 strategy=fedsimsiam optimizer=fedsimsiam num_workers=4 val_ratio=0 rad_ratio=0.02 local_epochs=5 num_rounds=200 warm_up_rounds=10 cos_decay=True batch_size=128 pretrained=False
# 25/5 apo brady bghke kako 
# declare -a partitioning =("dirichlet" "iid")
# declare -a num_clients =(10 16 32)
# declare -a num_classes =(4 10 30)
# for i in "${partitioning[@]}"
# for j in "${num_clients[@]}"
# for k in "${num_classes[@]}"
# do 
#  echo "${i} ${j} ${k}"
# done 

# for file in ./data/*.mtx; do
# 	for i in 2 4 5 10 15 20; do
# 	  echo "V3-OMP with "$file" with "$i" workers:" >> ./output_parallel/v3_omp.txt
# 	  export OMP_NUM_THREADS=$i
# 		./v3_omp $file >> ./output_parallel/v3_omp.txt
# 	done
# done


# python -m main --multirun optimizer=vanilla,momentum,proximal strategy=fedavg,fednova var_local_epochs=False,True seed=4,5,6