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

## 2️⃣ Data Preprocessing & EDA
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
```bash
dvc repro   # triggers model_building & model_evaluation
```
**Artifacts**
- `lgbm_model.pkl`  
- `tfidf_vectorizer.pkl`  
- `experiment_info.json`

> Steps 1–3 were validated in `notebooks/1_Preprocessing_&_EDA.ipynb`.

---

## 4️⃣ MLflow Server Setup on AWS
**Purpose**: Centralized experiment tracking (logs, parameters, metrics, model versioning).

**Components**
- **IAM User** (e.g., `ml_server`) with S3 access  
- **S3 Bucket** (e.g., `mlbucket922`) to store MLflow **artifacts** (models/logs/files)  
- **EC2 Instance** (e.g., `ml-machine`) to run the **MLflow server**

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

> All experiments are tracked in **MLflow**.

---

## 6️⃣ Make It Reproducible with DVC
`dvc.yaml` splits the pipeline into **4+1** stages (the last one is registration):
- **data_ingestion** → read/split raw data (controlled by `data_ingestion.test_size`)
- **data_preprocessing** → cleaning + vectorization (controlled by `ngram_range / max_features / vectorizer`)
- **model_building** → training (controlled by LightGBM/XGBoost/LogReg params, etc.)
- **model_evaluation** → unified evaluation, outputs `experiment_info.json`
- **model_registration** → (see next) write into MLflow Model Registry

**Commands**
```bash
dvc repro          # reproduce the whole pipeline
dvc exp run        # run an experiment with parameter changes
dvc params diff    # show parameter differences
```

---

## 7️⃣ Register the Model (MLflow Model Registry)
**Goal**: Register the **approved best model** into MLflow Registry with a canonical name & versioning for easy deployment/rollback.

**Script & Dependencies**
- `src/model/register_model.py`  
- **Input**: `experiment_info.json` (tells the script where to load the best artifacts)  
- **Output**: `model_registration.json` (records model name & version)

**Run**
```bash
python src/model/register_model.py
# or trigger via the model_registration stage in dvc.yaml
```
**Result**: In MLflow, you’ll see versions (e.g., `yt_chrome_plugin_model` v1/v2/…).

---

## 8️⃣ Serving with Flask API
**Location**: `flask_api/app.py`  
**Highlights**: CORS enabled; supports loading from **MLflow Model Registry** (or fallback to local `pkl`).

**Key Endpoints**
- **POST /predict**  
  Request: `{"comments": ["text1", "text2", ...]}`  
  Response: `[{"comment": "...", "sentiment": 1|0|-1}, ...]`
- **POST /predict_with_timestamps**  
  Request: `{"comments":[{"text":"...","timestamp":"01:23"}, ...]}`  
  Response: predictions with timestamps for video overlay/markers
- **POST /generate_chart**  
  Request: `{"sentiment_counts":{"1":10,"0":5,"-1":2}}`  
  Response: pie chart PNG (Matplotlib)
- **POST /generate_wordcloud**  
  Request: `{"comments":["...","..."]}`  
  Response: word cloud PNG (WordCloud)

**Run locally**
```bash
python flask_api/app.py  # default 0.0.0.0:5000
```

**Example (cURL)**
```bash
curl -X POST http://localhost:5000/predict   -H "Content-Type: application/json"   -d '{"comments":["this is great","not good at all"]}'
```

---

## 9️⃣ Frontend: Chrome Extension (`yt-chrome-plugin-frontend/`)
**Suggested Structure**
```
yt-chrome-plugin-frontend/
├─ manifest.json
├─ popup.html
├─ popup.js
├─ background.js         # optional: long-running tasks / context menus
└─ icons/                # 16/48/128 px
```

**Workflow**
1. `popup.js` reads comments from the page or via **YouTube Data API** (configure API key, or fetch via backend).  
2. POST the comment array to Flask `/predict` or `/predict_with_timestamps`.  
3. Render sentiment labels and ratios; optionally fetch `/generate_chart` and `/generate_wordcloud` images and display.

**Load for development**
- Open `chrome://extensions/` → enable **Developer mode** → **Load unpacked** → select `yt-chrome-plugin-frontend/`  
- In the extension settings, set **API Base URL** (e.g., `http://<your-domain-or-ec2>:5000`)

**Security tip**: Store your API key in `chrome.storage` or the backend; **don’t hardcode it in Git**.

---

## 🔟 Dockerization & Deployment (ECR + EC2)
**Build image**
```bash
docker build -t mlops-pipeline:latest .
```

**Run (locally or on EC2)**
```bash
docker run -d -p 8080:5000   -e MLFLOW_TRACKING_URI="http://<mlflow-ec2>:5000"   --name yt-sentiment mlops-pipeline:latest
```

**Push to ECR (example)**
```bash
aws ecr get-login-password --region us-east-1 |   docker login --username AWS --password-stdin <acct>.dkr.ecr.us-east-1.amazonaws.com

docker tag mlops-pipeline:latest <acct>.dkr.ecr.us-east-1.amazonaws.com/mlops-pipeline:latest
docker push <acct>.dkr.ecr.us-east-1.amazonaws.com/mlops-pipeline:latest
```

---

## 1️⃣1️⃣ GitHub Actions: CI/CD (with Self-Hosted Runner)
**Required Secrets**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (e.g., `us-east-1`)
- `AWS_ECR_LOGIN_URI` (e.g., `<acct>.dkr.ecr.us-east-1.amazonaws.com`)
- `ECR_REPOSITORY_NAME` (e.g., `mlops-pipeline`)

**Recommended Stages**
- **CI**: Checkout → Lint → Unit tests  
- **CD (build & push to ECR)**: Login ECR → Build → Push  
- **Deploy (on self-hosted runner/EC2)**: Pull → `docker run -d -p 8080:5000 ...`

---

### Notes
- For macOS + LightGBM/OpenMP errors: `brew install libomp` and export `DYLD_LIBRARY_PATH` accordingly.  
- Keep secrets out of the repo; never commit real API keys or credentials.  
- `experiment_info.json` links training/evaluation outputs; `model_registration.json` records the production model name/version.
