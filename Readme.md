## Setup Environment

Set up the environment using Anaconda.
```
conda create --name protac python=3.8
conda activate protac
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

pip install biopandas==0.4.1 biopython==1.79 dill==0.3.6 e3nn==0.5.1 easydict==1.10 lmdb==1.4.1 networkx==3.1 numpy==1.23.5 pandas==2.0.1 prody==2.4.1 pytorch-lightning==2.0.2  rmsd==1.5.1 scikit-learn==1.2.2 scipy==1.10.1 tqdm==4.65.0 rdkit-pypi==2022.9.5 networkx==3.1 autograd==1.6.2 pyyaml==6.0 easydict==1.10 kmeans-pytorch timm==0.9.5tensorboard==2.13.0 tensorboard==2.13.0 tensorboardx==2.6 numba==0.57.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==2.3.1 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

git clone https://github.com/subhadarship/kmeans_pytorch
cd kmeans_pytorch
pip install --editable .

```

To preprocess the structures you also needs to install MSMS(https://ccsb.scripps.edu/msms/downloads/) and replace the bin path in data config and `dataset/preprocess.py`

## Train
```
python train_PP.py --mode train --data default
```
