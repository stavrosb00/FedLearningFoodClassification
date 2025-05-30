# [Thesis - Federated learning between devices for food image recognition](https://ikee.lib.auth.gr/record/356526/files/Bouliopoulos_Stavros_Vasileios_Thesis.pdf)
This thesis project deals with the investigation and analysis of the food image recognition problem in the context of Federated Learning system utilising supervised and self-supervised machine learning techniques. The key features of machine learning and Federated Learning are described with emphasis on the image categorization domain. In particular, the convolutional neural network ResNet-18 is chosen as a basis for comparison for food recognition. The document analyzes, among other aggregation algorithms, the implemented aggregation algorithms FedAvg, FedProx, SCAFFOLD of federated supervised learning and the implemented algorithm FedSimSiam of federated self-supervised learning. The selected trade-offs are then analyzed to simulate federated learning between devices using the widely recognized computational library Flower. This thesis concludes by presenting and evaluating executed Federated Learning simulations that compare and examine different parameterized cases using the Food10 dataset, which is a subset of Food101 food image set.

# Required Python Libraries
The following sequence of commands installs all required Python libraries and dependencies  inside a conda environment for this project:

```
conda create -n my_env_name python=3.10.13
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install flwr[simulation]==1.7.0
pip install hydra-core==1.3.2
conda install pandas
conda install matplotlib
conda install scikit-learn=1.3.0
conda install tqdm
conda install tensorboard
pip install py-spy snakeviz
```

<!-- Optional: Uncomment to install albumentations for augmentations -->
<!-- pip install albumentations  -->
