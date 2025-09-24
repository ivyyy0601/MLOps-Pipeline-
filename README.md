# 🎬 MLOps-Pipeline: YouTube Sentiment Analysis

End-to-end **YouTube comment sentiment analysis MLOps** project:  
Data → Preprocessing/EDA → Baseline Model → MLflow (AWS) → DVC Reproducible Pipeline → Model Registry → Flask API → Chrome Extension → Docker & GitHub Actions CI/CD.

---

## 1️⃣ Data Collection
Collect raw data via **APIs / databases / CSV** (this project uses CSV).
```bash
dvc repro   # first run triggers data_ingestion
```

---

## 2️⃣ Data Preprocessing & EDA [use the reddit data first]
- **Cleaning & Processing**: handle missing/duplicate values, regex cleaning, lowercasing  
- **Stopwords & Text**: custom stopword list, keep negation/transition words, lemmatization  
- **Features**: BoW/TF‑IDF, control n‑gram and max_features  
- **Stats**: word/char/punctuation counts  
- **EDA**: class balance, text length distribution, stopword stats, Top‑N n‑grams, **word cloud & frequency analysis**

**Outputs**
- `data/interim/train_processed.csv`  
- `data/interim/test_processed.csv`

---

## 3️⃣ Baseline Model
First make it work with a simple baseline: **TF‑IDF + Logistic Regression / Random Forest**. Use metrics as a reference for later improvements.

> Steps 1–3 were validated in `notebooks/1_Preprocessing_&_EDA.ipynb`.

---

## 4️⃣ MLflow Server Setup on AWS
**Purpose**: Centralized experiment tracking (logs, parameters, metrics, model versioning).

**Components**
- **IAM User** (e.g., `ml_server`) with S3 access  
![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%202.png)
- **S3 Bucket** (e.g., `mlbucket922`) to store MLflow **artifacts** (models/logs/files)  
![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%204.png)
- **EC2 Instance** (e.g., `ml-machine`) to run the **MLflow server**
![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%205.png)

### ⚙️ AWS Setup Details (in order)
**1) Connect to EC2**
```bash
sudo apt update
sudo apt install python3-pip
sudo apt install pipenv
sudo apt install virtualenv
```

**2) Project Environment**
```bash
mkdir mlflow && cd mlflow
pipenv install mlflow awscli boto3
pipenv shell
aws configure
```

**3) Run MLflow Server**
```bash
mlflow server \
    --host 0.0.0.0 \
    --default-artifact-root s3://mlbucket922
# replace mlbucket922 with your S3 bucket name
```

**4) Access MLflow UI**
- Add **inbound port 5000** in the EC2 **Security Group**  
- Open: `http://<EC2-Public-DNS>:5000/`  
  (example: `http://ec2-xx-xxx-xxx-xx.compute-1.amazonaws.com:5000/` — use your own)
![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%209.png)

> The four steps above mirror the internal doc sequence. Follow them to bring MLflow online.

---

## 5️⃣ Improve Baseline Model (Experiment Plan)
In NLP, a typical baseline is **BoW/TF‑IDF + LogReg/SVM/Naive Bayes**. Improvement path:
- **BoW / TF‑IDF**: fundamental text vectorization
- **Max Features**: cap vocabulary size (e.g., 5k) to control complexity and speed/generalization
- **Imbalanced Data**: SMOTE, class weights, undersampling
- **Multiple Models + Tuning**: LogReg / RF / XGBoost / SVM; Grid/Random/Optuna
- **Stacking**: use base model predictions as features for a meta‑model

**Notebooks (already in repo)**
- Baseline: `2_experiment_1_baseline_model.ipynb`  
- BoW vs TF‑IDF: `3_experiment_2_bow_tfidf.ipynb`  
- TF‑IDF + n‑gram + max_features: `4_experiment_3_tfidf_(1,3)_max_features.ipynb`  
- Imbalanced: `5_experiment_4_handling_imbalanced_data.ipynb`  
- XGBoost + HPT: `6_experiment_5_xgboost_with_hpt.ipynb`  
- LightGBM + HPT: `7_experiment_6_lightgbm_detailed_hpt.ipynb`  
- Stacking: `8_stacking.ipynb`

code in `notebooks/`.

> All experiments are tracked in **MLflow**.

---

## 6️⃣ Make It Reproducible with DVC
`dvc.yaml` splits the pipeline into **4+1** stages (the last one is registration):
- **data_ingestion** → read/split raw data (controlled by `data_ingestion.test_size`)  → output: data/raw 
- **data_preprocessing** → cleaning + vectorization (controlled by `ngram_range / max_features / vectorizer`)  → output: data/interim 
- **model_building** → training (controlled by LightGBM/XGBoost/LogReg params, etc.)→ output: data/interim
- **Model Evaluation:** `src/model/model_evaluation.py`  
  Logs **Accuracy, F1, Classification Report, Confusion Matrix** to **MLflow**--->output:lgbm_model.pkl  #save trained LightGBM model and TF-IDF

**Run**
```bash
conda create -n youtube python=3.11 -y
conda activate youtube
pip install -r requirements.txt

dvc init
dvc repro
```


---

## 7️⃣ Register the Model (MLflow Model Registry)  Add Model to Model Registry (MLflow)
**Purpose**: Store the best-performing model into the **MLflow Model Registry**, generating model versions (v1/v2/…), enabling easier deployment and rollback. 

**Script & Dependencies**
- `src/model/register_model.py`  
- **Input**: `experiment_info.json` (tells the script where to load the best artifacts)  
- **Output**: `yt_chrome_plugin_model` ( records best model.）
![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2029.png)
 

---

# 8️⃣ Implement Chrome Plugin + Flask API

### Flask API (`flask_api/app.py`)
- Uses `matplotlib.use('Agg')` → supports rendering without GUI.  
- **Core Endpoints**:
  - `POST /predict`: input `{"comments":[...]}` → returns sentiment predictions  
  - `POST /predict_with_timestamps`: predictions with `timestamp` included  
  - `POST /generate_chart`: input `sentiment_counts` → returns pie chart PNG  
  - `POST /generate_wordcloud`: input `comments` → returns word cloud PNG  

### Model Loading
1. **Preferred**: Load dynamically from MLflow Registry  
   ```python
   mlflow.pyfunc.load_model("models:/<name>/<ver>")
   ```
2. **Fallback**: Load from local `.pkl` file (template already in code comments).

> ⚡ The project already includes complete **preprocessing + vectorization alignment** logic to ensure the input DataFrame schema matches training.

### Run Locally
```bash
python flask_api/app.py  # default http://0.0.0.0:5000
```

### Chrome Plugin (`yt-chrome-plugin-frontend/`)
1. Open Chrome → go to `chrome://extensions/`  
2. Enable **Developer Mode**  
3. Click **Load unpacked** → select `yt-chrome-plugin-frontend/`  
4. The extension will call the Flask API and display predictions as labels, ratios, and visualizations (pie chart/word cloud).
[image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2031.png)

### Development & Debugging
- Use **Postman** to test API endpoints.  
![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2030.png)
- For **real YouTube data** → request a **YouTube Data API Key** from Google Cloud.  
  ⚠️ Do **NOT** commit real API Keys to the repo (sample key in docs is placeholder only).

----view my favorite vloger
![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2034.png)

---

---

# 9️⃣ CI/CD (Docker + GitHub Actions + AWS)

Goal: Automate build → push to **ECR** → pull & run on **EC2**.

### Workflow
1. **Build** docker image  
2. **Push** to ECR  
3. **Launch** EC2  
4. EC2 **pulls** image  
5. EC2 **runs** container  

Required AWS Policies:
- `AmazonEC2ContainerRegistryFullAccess`  
- `AmazonEC2FullAccess`  

### Example ECR Repository URI
```
294892597101.dkr.ecr.us-east-1.amazonaws.com/mlops-pipeline
```

---

### Self-hosted Runner & Secrets

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2035.png)

#### Install Docker on EC2
```bash
sudo apt-get update -y
sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

#### Configure EC2 as Self-hosted Runner
Go to GitHub → `Settings → Actions → Runners` and follow the setup guide.

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2039.png)

---

### Configure GitHub Secrets
In repo **Settings → Secrets and variables → Actions**, add:

- `AWS_ACCESS_KEY_ID`  
- `AWS_SECRET_ACCESS_KEY`  
- `AWS_REGION` (e.g., `us-east-1`)  
- `AWS_ECR_LOGIN_URI` (e.g., `294892597101.dkr.ecr.us-east-1.amazonaws.com`)  
- `ECR_REPOSITORY_NAME` (e.g., `mlops-pipeline`)  


![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2040.png)
---
