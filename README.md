# MLOps-Pipeline-

conda create -n youtube python=3.11 -y

conda activate youtube

pip install -r requirements.txt


## DVC

dvc init

dvc repro

（（在弄model_building:
1. # 仍需在 conda 环境里运行你的代码
conda activate youtube

2. # 用 brew 安装 OpenMP
brew install libomp

3. #（通常不需要，但万一还提示找不到）
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH"
验证一下
python -c "import lightgbm; from lightgbm import LGBMClassifier; print('LightGBM OK:', lightgbm.__version__)"
看到 LightGBM OK: 就说明依赖齐了。然后再跑：
））
#然后就是run你的mlflow了


dvc dag



## AWS

aws configure

#去google cloud 弄一个api key
#AIzaSyA7kKnxD4jF9cKOFhtMM0y4WgoJF5Tl108

chrome://extensions/



cicd
# Description: About the deployment
1. Build docker image of the source code
2. Push your docker image to ECR
3. Launch your EC2
4. Pull your image from ECR in EC2
5. Launch your docker image in EC2

# Policy:
1. AmazonEC2ContainerRegistryFullAccess
2. AmazonEC2FullAccess

## 3. Create ECR repo to store/save docker image
- Save the URI: 294892597101.dkr.ecr.us-east-1.amazonaws.com/mlops-pipeline


## how to get youtube api key from gcp:

https://www.youtube.com/watch?v=i_FdiQMwKiw



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 294892597101.dkr.ecr.us-east-1.amazonaws.com/mlops-pipeline

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one
	(就是去github把语句给弄下来 )


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  294892597101.dkr.ecr.us-east-1.amazonaws.com
    ECR_REPOSITORY_NAME = mlops-pipeline
