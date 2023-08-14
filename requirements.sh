apt update && apt install -y libsm6 libxext6 libxrender-dev
pip install pandas tensorboardX scikit-image tensorboard commentjson h5py matplotlib lmdb einops numpy==1.20.3 -i https://pypi.mirrors.ustc.edu.cn/simple
### optical flow
# pip install optuna
pip install pyiqa -i https://pypi.mirrors.ustc.edu.cn/simple
pip3 install --upgrade protobuf==3.20.1
python /data/booker/LN_base/Code_redrain/USDerain_v2/train_mprnet/pytorch-gradual-warmup-lr/setup.py install
pip install commentjson