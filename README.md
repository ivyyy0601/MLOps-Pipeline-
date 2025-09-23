# MLOps-Pipeline-

conda create -n youtube python=3.11 -y

conda activate youtube

pip install -r requirements.txt


## DVC

dvc init

dvc repro

（（在弄model_building:
# 仍需在 conda 环境里运行你的代码
conda activate youtube

# 用 brew 安装 OpenMP
brew install libomp

#（通常不需要，但万一还提示找不到）
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH"
验证一下
python -c "import lightgbm; from lightgbm import LGBMClassifier; print('LightGBM OK:', lightgbm.__version__)"
看到 LightGBM OK: 就说明依赖齐了。然后再跑：
））

dvc dag



## AWS

aws configure