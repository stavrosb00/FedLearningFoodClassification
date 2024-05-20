# FLsim
## FL simulation
conda create -n my_env_name python=3.10.13

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install flwr[simulation]==1.7.0

pip install hydra-core==1.3.2

conda install pandas

conda install matplotlib

conda install scikit-learn=1.3.0

conda install tqdm

conda install tensorboard 

pip install py-spy snakeviz  for profiling

[comment]: # (might need pip install tensorboard to avoid warnings w/o filtering them afterwards)
[comment]: # (might need conda install cudatoolkit)
[comment]: # (ray==2.6.3)
