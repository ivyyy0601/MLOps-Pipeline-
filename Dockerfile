FROM python:3.11-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]

#Dockerfile 就是写给 Docker 的“配方”：告诉它“怎么把我的代码和依赖打包成一个镜像（image）”。
#有了镜像，就能在任何装了 Docker 的机器上以**容器（container）**方式一键运行，省去“装环境、装依赖、配版本”的麻烦。