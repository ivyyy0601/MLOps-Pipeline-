# 使用 Python、AWS、Docker 的 MLOps 管道 – YouTube 观众情绪

1. **Data collection**
    
    → 数据收集：通过 API、数据库、CSV 文件等方式获取原始数据。
    
2. **Data preprocessing and EDA**
    
    → 数据预处理和探索性数据分析 (Exploratory Data Analysis)：清洗数据、处理缺失值/异常值、特征工程，以及对数据进行可视化和统计分析。
    
3. **Building Baseline Model**
    
    → 建立基线模型：先用最简单的模型（如 Logistic Regression、Random Forest、TF-IDF + 分类器）跑一个初始版本，作为后续优化的参考标准。
    
4. **Setup MLflow server on AWS**
    
    → 在 AWS 上搭建 **MLflow 服务器**：用于管理机器学习实验，包括模型训练日志、参数、指标、模型版本管理等。
    
5. **improve Baseline Model**
    
    → 在 NLP 任务里，基线一般是用 Bag-of-Words (BoW) 或 TF-IDF 特征 + 逻辑回归 / SVM / Naive Bayes 做分类。接下来的步骤就是在这个基线基础上改进。
    
    - **BoW, TF-IDF**
        
        → 最基础的文本特征提取方法，把文本转成词频矩阵或加权矩阵。
        
    - **Max Features**
        
        → 限制特征数量（比如取前 5,000 个词），减少维度，提高训练速度和泛化能力。
        
    - **Handling Imbalanced Data**
        
        → 处理类别不平衡问题，比如：
        
        - 使用 **SMOTE** 做过采样
        - 使用 **class weights** 调整损失函数
        - 下采样多数类
    - **Hyperparameter Tuning with Multiple ML Models**
        
        → 调不同模型（Logistic Regression, Random Forest, XGBoost, SVM），并通过网格搜索 (Grid Search) 或随机搜索 (Random Search) 来找到最优参数。
        
    - **Stacking Model**
        
        → 模型集成 (Ensemble Learning) 技术之一，把多个模型的预测结果作为新特征，再训练一个“元模型” (meta-model) 来提高性能。
        
    
6. **Build ML pipeline using DVC**

→ 用 **DVC (Data Version Control)** 搭建机器学习流水线，用于数据版本管理、特征处理、训练流程自动化。

1. **Add model to model registry**
    
    → 把训练好的模型加入 **模型注册库 (Model Registry)**，便于版本管理、回滚、部署。常见工具：MLflow、Sagemaker、Kubeflow。
    
2. **Implement Chrome plugin**
    
    → 开发一个 **Chrome 插件**，把模型的预测功能嵌入到浏览器插件中，作为用户端的交互界面。
    
3. **CI/CD workflow**
    
    → 建立持续集成 / 持续部署 (CI/CD) 工作流，例如用 GitHub Actions、Jenkins 或 GitLab CI，把代码更新、模型训练、测试和上线自动化。
    
4. **Dockerization**
    
    → 用**Docker**把模型和依赖环境打包，保证在不同环境里都能一致运行。
    
5. **Deployment – AWS**
    
    → 把模型部署到**AWS**（比如 EC2、SageMaker、ECS、Lambda），让它对外提供 API 或服务。
    
6. **GitHub**
    
    → 用 GitHub 管理项目代码，做版本控制、协作和开源。
    

1. **Data collection**
    1. 处理直接去掉nul值和duplicate数值
    2. 多余的首尾空格在 **分词(tokenization)** 或 **向量化(比如 TF-IDF, Word2Vec)** 时会造成 “假特征”。举例：
        
        `"apple "` 和 `"apple"` 在没清洗之前会被当成两个不同的 token。
        
        这样会让词表变得混乱，增加噪声，影响模型效果。
        
    3. 
    - **统一格式**（小写、去掉多余符号）
    - **清理噪音**（URL、换行符、空格）
    - **保证一致性**（让同一个词只对应一种形式）
    
    d. EDA:
    
    可视化一些基本的东西
    
    ### **`KDE 图 / Boxplot / Scatterplot`**
    
    - **`KDE 图**：看不同类别下词数的密度分布，是否有偏移。`
    - **`箱线图 (Boxplot)**：找极端值（outliers）。`
    - **`散点图 (Scatterplot)**：直观展示类别 vs. 词数的关系。`
    
    `👉 意义：`
    
    - `检查类别间是否有明显差异。`
    - `判断要不要对 **长尾数据** 做截断/过滤。`
        - `比如：发现负面评论字数特别长，可能要考虑加 max_features 或长度归一化。`
    
    e. 停用词：
    
    ![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image.png)
    

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%201.png)

f. **特征工程**

- **`捕捉更长的上下文**：比 bigram 更能体现语义，比如：`
    - `bigram: "climate change"（主题大方向）`
        - `trigram: "climate change policy"（更具体的议题）`
    - **`识别典型短语**：可以看出用户评论里最常见的完整表达。`
    - **`特征工程**：在情感分析、主题建模中，trigram 特征比单词或 bigram能提供更细致的上下文信息。`

**文本特征工程 (Text Feature Engineering)：**

### `1. **基础特征提取**`

- **`word_count**（词数）：一句话里单词多少，可以反映评论的长度和信息量。`
    - `短评论（如 "ok", "bad"）和长评论（如一整段解释）通常在语义和情感上表现不同。`
- **`num_chars**（字符数）：与 word_count 类似，但能更细粒度地体现评论的复杂度。`
- **`num_punctuation_chars**（标点数）：标点（如 "!!!", "???"）常常和情绪强度相关，尤其是在社交媒体评论里。`

---

### `2. **语言复杂度与风格分析**`

- **`num_stop_words**（停用词个数）：停用词（如 *the, and, is*）通常不承载太多语义，但停用词比例可以区分评论风格。`
    - `比如学术性文本会包含更多停用词，而情绪化的短评可能停用词更少。`
- **`lemmatization**（词形还原）：把 *running, runs → run*，减少同义形式的冗余，使模型更容易理解。`

---

### `3. **情感与分类任务的辅助特征**`

- `这些特征可以作为机器学习模型的输入：`
    - **`情感分析 (Sentiment Analysis)**：标点、停用词比例、词数长短，可能帮助区分正负面评论。`
    - **`垃圾评论检测 (Spam Detection)**：spam 评论往往有特定模式，比如超长文本或大量重复符号。`
    - **`主题建模 (Topic Modeling)**：特征能辅助文本聚类或分类。`

---

### `4. **可视化与探索性分析**`

- `通过这些统计，可以画出：`
    - `评论长度的分布；`
    - `不同情感类别下的平均词数差异；`
    - `标点使用习惯和情绪的关系。`
        
        `这些不仅能帮你理解数据本身，还能发现潜在的噪音或异常（比如有些评论过长、包含无意义字符等）。`
        
        1. **`取数与初筛`**
        - `读取 reddit.csv（37249 行，2 列）。`
        - `去缺失（clean_comment 有 100 个 NaN，全为中性类）、去重复、去空白行。`
        1. **`文本清洗（ETL-Transform）`**
        - `统一小写、去首尾空格，替换换行符 \n。`
        - `检测 URL（可选择性删除）。`
        - `仅保留英文与常见标点；统计字符频率。`
        - `自定义停用词表（保留否定词 not/no/however/but/yet），移除其余停用词。`
        - `词形还原（WordNetLemmatizer）。`
        1. **`特征工程（数值化特征）`**
        - `word_count（词数）、num_chars（字符数）、num_stop_words（停用词数）、num_punctuation_chars（标点数）。`
        - `目的：把“文本风格/长度/情绪强度”转换为可用于建模的数值特征。`
        1. **`EDA（探索性分析与可视化）`**
        - `类别占比：正面 42.86%、中性 34.71%、负面 22.42%。`
        - `词数分布（KDE/箱线图/中位数柱状图）：正面与负面更“啰嗦”，中性更短。`
        - `停用词数量分布与 Top-25 停用词。`
        - `N-gram：Top-25 **bigrams** 与 **trigrams**（常见短语，捕捉上下文搭配）。`
        - `词云：整体、以及按情感类别分别绘制。`
        - `Top-N 词频（整体 & 按情感堆叠柱状图）：展示同一词在三类情感中的出现结构。`
        1. **`用途与价值`**
        - `清洗与特征让文本更“干净一致”，便于后续 **向量化/建模**；`
        - `EDA 揭示差异（如长度、停用词比例、典型短语），指导 **特征选择** 与 **模型假设**（例如把长度类特征与 TF-IDF/词向量一起喂给模型）。`

4. **Setup MLflow server on AWS**

→ 在 AWS 上搭建 **MLflow 服务器**：用于管理机器学习实验，包括模型训练日志、参数、指标、模型版本管理等。

1. **IAM 用户**
- **ml_server**
    - 这是你创建的 IAM 用户，给它分配了访问 S3 的权限。
    - 它有一组 **Access Key ID** 和 **Secret Access Key**，需要在 EC2 上配置，供 MLflow 使用。

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%202.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%203.png)

1. **S3 存储桶**
- **mlbucket922**
    - 这是你在 S3 上创建的 bucket（存储桶），用来存放 MLflow 的 artifact（模型、日志、文件等）。

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%204.png)

1. **EC2 实例**
- **i-0b7f2bf64c9769970 (ml-machine)**
    - 这是你创建的 **EC2 虚拟机实例**，相当于一台云服务器。
- **ml_server**
    - 这是你创建的 IAM 用户，给它分配了访问 S3 的权限。
    - 它有一组 **Access Key ID** 和 **Secret Access Key**，需要在 EC2 上配置，供 MLflow 使用。

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%205.png)

1. open ec2
    
       
    
    1. `更新包列表`
    
    ```bash
    sudo apt update
    
    ```
    
    1. `安装 pip（Python 3 的包管理器）`
    
    ```bash
    sudo apt install python3-pip
    
    ```
    
    1. `安装 pipenv`
    
    ```bash
    sudo apt install pipenv
    
    ```
    
    1. `安装 virtualenv`
    
    ```bash
    sudo apt install virtualenv
    
    ```
    
    1. `创建并进入项目目录`
    
    ```bash
    mkdir mlflow
    cd mlflow
    
    ```
    
    1. `用 pipenv 安装需要的包`
    
    ```bash
    pipenv install mlflow
    pipenv install awscli
    pipenv install boto3
    
    ```
    
    1. `进入 pipenv 环境`
    
    ```bash
    pipenv shell
    
    aws configure
    
    ```
    
2. 

### . **MLflow 启动命令**

在 EC2 里运行：

```bash
mlflow server \
    --host 0.0.0.0 \
    --default-artifact-root s3://mlbucket922 \
mlflow server \
    --host 0.0.0.0 \
    --default-artifact-root s3://mlbucket922 \
```

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%206.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%207.png)

[https://www.notion.so](https://www.notion.so)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%208.png)

[ec2-34-238-171-48.compute-1.amazonaws.com](http://ec2-34-238-171-48.compute-1.amazonaws.com/)

http://ec2-34-238-171-48.compute-1.amazonaws.com/

[http://ec2-34-238-171-48.compute-1.amazonaws.com:5000/](http://ec2-34-238-171-48.compute-1.amazonaws.com:5000/)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%209.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2010.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2011.png)

1. **安装并连上 MLflow Server**
- 安装 `mlflow`，并把追踪地址指向你的 EC2：
    
    `mlflow.set_tracking_uri("http://ec2-54-175-41-29.compute-1.amazonaws.com:5000/")`
    
- 运行任何 `mlflow.start_run()` 内的代码，参数、指标、工件都会被发到那个服务端。
- 你创建了实验 **“RF Baseline”**（若不存在会自动创建）。控制台提示里的链接能直接打开到 Run/Experiment 页面。
1. **S3 做工件（artifacts）存储**
- 从日志：`artifact_location='s3://mlflow-test-25/...'` 可见，模型文件、图像、数据等工件都存进了 S3（你的 bucket）。
- 你把 `confusion_matrix.png` 和 `dataset.csv` 都作为工件上传了；模型保存在 `random_forest_model/` 目录下。
1. **数据预处理（相当于小型 ETL）**
- 读入 reddit 数据 → `dropna` / 去重 / 去空白样本。
- 统一小写、去首尾空格、清理换行、保留常见标点、去掉非英文字元。
- **停用词**：保留了“not, no, but, however, yet”等否定/转折词（对情感很关键）。
- **Lemmatization**：用 `WordNetLemmatizer` 做词形还原。
    
    → 输出列仍叫 `clean_comment`（干净文本），这一步就是“为建模做干净输入”。
    
1. **特征化 + 模型**
- 用 **Bag of Words（CountVectorizer）**，`max_features=10000` → 变成 36,793 × 10,000 的稀疏特征矩阵。
- 划分训练/测试，训练 **RandomForestClassifier**（`n_estimators=200, max_depth=15`）。
- 记录 **accuracy**、**classification_report** 的各类指标，并产出混淆矩阵图。
1. **MLflow 记录了什么**
- **Params**：矢量化方式、max_features、RF 的 n_estimators、max_depth 等。
- **Metrics**：accuracy、以及每个类别（-1/0/1）的 precision/recall/f1。
- **Artifacts**：`confusion_matrix.png`、`dataset.csv`、以及 **已保存的模型**（`random_forest_model`）。

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2012.png)

这是在把 **MLflow 客户端**连到你的服务端，并创建/切换到一个实验。

具体含义：

1. `mlflow.set_tracking_uri("http://ec2-34-238-171-48.compute-1.amazonaws.com:5000/")`
    - 告诉本地/笔记本里的 MLflow 客户端：以后所有 **runs 的参数、指标、工件信息** 都发到这个 **MLflow Tracking Server**（你在 EC2 上开的 5000 端口）。
2. `mlflow.set_experiment("RF Baseline")`
    - 选择名为 “RF Baseline” 的 **Experiment** 作为当前实验空间；
    - 日志里提示 “does not exist. Creating a new experiment.” 表示该实验不存在，于是**在服务端新建**了一个，并返回：
        - `experiment_id`：实验的唯一 ID（后面所有 runs 都归到这个实验名下）。
        - `artifact_location='s3://mlbucket922/…'`：该实验的**工件根目录**（模型、图像、csv 等会存到这个 S3 路径下）。

以后你在 `with mlflow.start_run(): ...` 中的 `log_param / log_metric / log_artifact / log_model`：

- **元数据**（参数、指标、tags、runs 结构）保存在 MLflow 服务器的后端存储；
- **工件**（模型文件、图片、数据）按上面的 `artifact_location` 存进 **S3** 对应前缀。
    
    这样你就能在 MLflow UI（5000 端口）里看到该实验下所有 run 的记录与工件链接。
    

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2013.png)

：你这次是在**做特征工程对比实验**，把“只用 BoW（CountVectorizer）”升级为**BoW vs TF-IDF** + **不同 n-gram** 组合的系统比较，并把每次结果都记到同一个 MLflow 实验里。

## 和“刚刚”的区别

| 点 | 之前（RF Baseline） | 现在（Exp 2 - BoW vs TfIdf） |
| --- | --- | --- |
| 向量化 | **CountVectorizer**（BoW）一种，`max_features=10000`，默认 `ngram_range=(1,1)` | **可切换**：BoW **或** TF-IDF；并循环 **(1,1)、(1,2)、(1,3)**；`max_features=5000` |
| 实验结构 | 单次 run | 一次脚本触发 **6 个 run**（BoW×3 + TF-IDF×3），便于横向对比 |
| 记录 | 参数/指标/混淆矩阵/模型 | 同上，但每个 run 会带上向量器名、n-gram 作为 tag/param，UI 对比更清晰 |
| 目标 | 跑通基线 | 比较**哪种向量化更好**（以及哪种 n-gram 设置更合适） |

## “Which vectorization?”——两种向量化的本质

- **BoW（CountVectorizer）**
    
    统计词或 n-gram 的**出现次数**。
    
    优点：简单、稳定、速度快。
    
    缺点：高频停用词权重可能过大（你前面已做停用词处理，影响减弱）。
    
- **TF-IDF（TfidfVectorizer）**
    
    在词频基础上乘以 **IDF**，**降低全局常见词**、**提高区分度高的词**的权重。
    
    常在文本分类/信息检索里表现更好，尤其是“词很常见但没啥区分度”的场景。
    
- **n-gram**
    - (1,1)：仅单词；
    - (1,2)：单词+二元组（能引入短语模式，如 “not good”）；
    - (1,3)：再加三元组，表达力更强但更稀疏、维度更高。

## 结论怎么选？

- 打开 MLflow UI → 实验 **“Exp 2 - BoW vs TfIdf”**，比较各 run 的**指标**。
    
    数据**类别不平衡**（你前面报告里 -1 召回很低），仅看 accuracy 会被“多数类”掩盖。
    
    - 建议优先看 **macro F1**、**1 类的 recall/F1**。
- 经验上：**TF-IDF + (1,2)** 往往在文本分类更稳；但要以你这次的指标为准。
- 如果 TF-IDF 的宏平均 F1 更好（特别是 -1 类提升），就选它作为后续默认向量化。

### 🔹 第一个实验（Baseline RF）

- **数据来源**：直接读原始的 `reddit.csv`，然后做了清洗（去掉 NaN、去掉空字符串、去掉重复、正则清理、停用词处理、词形还原）。
- **保存**：最后用
    
    ```python
    df.to_csv('reddit_preprocessing.csv', index=False)
    
    ```
    
    得到 `36793` 行。
    
- **训练数据大小**：约 **36,793** 行。

---

### 🔹 第二个实验（BoW vs TF-IDF）

- **数据来源**：不是重新清洗原始 `reddit.csv`，而是直接读 **`reddit_preprocessing.csv`**。
    
    这时因为额外 `dropna` + `drop_duplicates`，以及存储过程中格式差异，结果少了一部分行。
    
- **你截图显示**：`df.shape = (36662, 2)`，比第一次少了大概 `131` 行。
- **训练数据大小**：约 **36,662** 行。

---

### ⚠️ 问题

所以这两个实验训练的模型，虽然逻辑类似，但 **数据集样本数不同**，严格对比并不公平。

---

### ✅ 建议做法（保证可比性）

如果你想要一个「干净的对比：BoW vs TF-IDF」：

1. **统一数据源**：始终从最原始的 `reddit.csv` 开始清洗，然后保存一份最终稳定的数据集，例如：
    
    ```python
    df_clean = clean_pipeline(raw_df)   # 统一的预处理函数
    df_clean.to_csv("reddit_final.csv", index=False)
    
    ```
    
2. **固定随机种子**：在 `train_test_split` 里已经加了 `random_state=42`，这部分是固定的，可以保证切分一致。
3. **后续实验**：不管是 Baseline RF 还是 BoW vs TF-IDF，全部使用 **同一份 `reddit_final.csv`**，这样对比结果才有意义。

```python

# Step 1: Function to run the experiment
def run_experiment(vectorizer_type, ngram_range, vectorizer_max_features, vectorizer_name):
    # Step 2: Vectorization
    if vectorizer_type == "BoW":
        vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=vectorizer_max_features)
    else:
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=vectorizer_max_features)

    X_train, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['category'], test_size=0.2, random_state=42, stratify=df['category'])

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Step 4: Define and train a Random Forest model
    with mlflow.start_run() as run:
        # Set tags for the experiment and run
        mlflow.set_tag("mlflow.runName", f"{vectorizer_name}_{ngram_range}_RandomForest")
        mlflow.set_tag("experiment_type", "feature_engineering")
        mlflow.set_tag("model_type", "RandomForestClassifier")

        # Add a description
        mlflow.set_tag("description", f"RandomForest with {vectorizer_name}, ngram_range={ngram_range}, max_features={vectorizer_max_features}")

        # Log vectorizer parameters
        mlflow.log_param("vectorizer_type", vectorizer_type)
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("vectorizer_max_features", vectorizer_max_features)

        # Log Random Forest parameters
        n_estimators = 200
        max_depth = 15

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Step 5: Make predictions and log metrics
        y_pred = model.predict(X_test)

        # Log accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Log classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # Log confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: {vectorizer_name}, {ngram_range}")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # Log the model
        mlflow.sklearn.log_model(model, f"random_forest_model_{vectorizer_name}_{ngram_range}")

# Step 6: Run experiments for BoW and TF-IDF with different n-grams
ngram_ranges = [(1, 1), (1, 2), (1, 3)]  # unigrams, bigrams, trigrams
max_features = 5000  # Example max feature size

for ngram_range in ngram_ranges:
    # BoW Experiments
    run_experiment("BoW", ngram_range, max_features, vectorizer_name="BoW")

    # TF-IDF Experiments
    run_experiment("TF-IDF", ngram_range, max_features, vectorizer_name="TF-IDF")

```

## 1) 核心目标

对同一份文本数据，比较两种**向量化方式**与不同 **n-gram** 设置在同一模型（RandomForest）下的效果，并把每次试验记录到 **MLflow**（参数、指标、混淆矩阵图、模型文件）。

比较的配置：

- 向量化：**BoW**（`CountVectorizer`） vs **TF-IDF**（`TfidfVectorizer`）
- n-gram：`(1,1)` (unigram), `(1,2)` (uni+bi), `(1,3)` (uni+bi+tri)
- 特征数上限：`max_features=5000`
- 模型：`RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)`

## 2) 函数 `run_experiment(...)` 详解

```python
def run_experiment(vectorizer_type, ngram_range, vectorizer_max_features, vectorizer_name):

```

### (a) 文本向量化配置

```python
if vectorizer_type == "BoW":
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=vectorizer_max_features)
else:
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=vectorizer_max_features)

```

- `ngram_range` 控制 n-gram 片段范围；越大能捕捉更多局部短语，但稀疏度变高。
- `max_features` 控制保留的高频特征上限；越大信息越多，内存和时间也越大。

### (b) 数据切分（避免信息泄漏）

```python
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_comment'], df['category'],
    test_size=0.2, random_state=42, stratify=df['category']
)
X_train = vectorizer.fit_transform(X_train)   # 只在训练集 fit
X_test = vectorizer.transform(X_test)         # 测试集仅 transform

```

- `stratify` 保证类别分布在 train/test 一致。
- 只在 **训练集** 上 `fit` 向量器，避免数据泄漏（做得对 ✅）。

### (c) 启动一次 MLflow 运行并记录元数据

```python
with mlflow.start_run() as run:
    mlflow.set_tag("mlflow.runName", f"{vectorizer_name}_{ngram_range}_RandomForest")
    mlflow.set_tag("experiment_type", "feature_engineering")
    mlflow.set_tag("model_type", "RandomForestClassifier")

```

- `runName` 会直接显示在 MLflow UI，便于区分每个配置。

### (d) 记录参数（params）

```python
mlflow.log_param("vectorizer_type", vectorizer_type)
mlflow.log_param("ngram_range", ngram_range)
mlflow.log_param("vectorizer_max_features", vectorizer_max_features)
mlflow.log_param("n_estimators", 200)
mlflow.log_param("max_depth", 15)

```

### (e) 训练 + 预测

```python
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

```

### (f) 记录指标（metrics）

```python
accuracy = accuracy_score(y_test, y_pred)
mlflow.log_metric("accuracy", accuracy)

classification_rep = classification_report(y_test, y_pred, output_dict=True)
for label, metrics in classification_rep.items():
    if isinstance(metrics, dict):
        for metric, value in metrics.items():
            mlflow.log_metric(f"{label}_{metric}", value)

```

- 除了总体 `accuracy`，还把每个类别的 `precision/recall/f1-score/support` 都打点到 MLflow（如 `1_precision`, `1_f1-score` 等）。

### (g) 记录混淆矩阵图（artifact）

```python
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix: {vectorizer_name}, {ngram_range}")
plt.savefig("confusion_matrix.png")
mlflow.log_artifact("confusion_matrix.png")
plt.close()

```

- 生成图片文件并作为 **artifact** 上传到 MLflow（如果 MLflow 设置了 S3 artifact store，这张图也会进你的 S3 桶里）。

### (h) 记录模型（artifact）

```python
mlflow.sklearn.log_model(model, f"random_forest_model_{vectorizer_name}_{ngram_range}")

```

- 在 MLflow UI 的每个 run 里可以点击下载模型；也可以用 `mlflow.sklearn.load_model()` 回载。

---

## 3) 批量试验入口

```python
ngram_ranges = [(1, 1), (1, 2), (1, 3)]
max_features = 5000

for ngram_range in ngram_ranges:
    run_experiment("BoW", ngram_range, max_features, vectorizer_name="BoW")
    run_experiment("TF-IDF", ngram_range, max_features, vectorizer_name="TF-IDF")

```

- 一共会产生 **6 个 run**：BoW ×3 + TF-IDF ×3。
- 每个 run 都会在 MLflow 里有独立的参数、指标、图和模型文件，方便横向对比。

## 2️⃣ 不同向量化方法 → 特征表达能力不同

### BoW (Bag of Words)

- 只统计词频（某个词出现几次）。
- 简单直观，但 **忽略词语顺序**，区分能力有限。
- 常常把高频词当成重要词，但在情感分析里，像 *not* 这种低频词可能更关键。

### TF-IDF (Term Frequency – Inverse Document Frequency)

- 在 BoW 基础上加了「逆文档频率」，降低全局高频词（如 *the*, *and*），提升那些在少数文档里才出现的词（如 *excellent*, *horrible*）。
- 更适合捕捉情感/主题，往往比单纯 BoW 表现更好。

### N-grams (n=2,3 …)

- 允许模型考虑短语而不是单个词。
- 例如 *“not good”*：
    - 在 BoW/TF-IDF unigram 下：`not` 和 `good` 分开，模型可能学不到“否定情感”。
    - 在 bigram 下：`not good` 会变成一个特征 → 更强区分力。

---

## 3️⃣ 不同任务对向量化敏感度很高

- 如果任务简单（如垃圾邮件分类），BoW 就够。
- 如果任务里有很多否定、转折（情感分析），TF-IDF + n-gram 往往明显更好。
- 如果任务涉及长上下文或语义理解（如问答系统），BoW/TF-IDF 都会失效，需要用 **词向量 (Word2Vec, GloVe)** 或 **Transformer embedding (BERT, GPT embeddings)**。

---

## 4️⃣ 模型效果差异，可能完全来自向量化

在你刚跑的实验里，模型部分其实固定（RandomForest 参数差不多），真正能拉开差距的就是 **向量化方法**：

- BoW vs TF-IDF → 结果差别可能达到 5–15% 的准确率。
- n-gram 范围不同 → 能捕捉短语依赖关系，F1 提升显著。
- max_features 太小可能丢信息，太大又可能过拟合。

---

## 5️⃣ 实验对比的意义

- **可解释性**：能看清楚哪种向量化更适合你的数据集。
- **可复现性**：在 MLflow 里记录下来，后续别人可以重现实验。
- **工程落地**：实际部署时，通常会选 **效果最好但开销最低** 的方法（比如 TF-IDF bigram 可能比 BERT embedding 轻量很多）。

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2014.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2015.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2016.png)

1.对比accuracy：bow（1，3）

2.precision / recal

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2017.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2018.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2019.png)

## 和前两个实验的关系

- 实验1：BoW、`max_features=10000`、而且是 **先向量化再切分**（有数据泄漏） → 参考价值偏低。
- 实验2：系统对比 **BoW vs TF-IDF**，`max_features=5000`，三种 n-gram(1,1)/(1,2)/(1,3)，而且是 **先切分再 fit 向量器** → 结果可靠，可横向比较“方法+ngram”。
- 实验3（这次）：固定 **TF-IDF + (1,3)**，只改变 **`max_features` ∈ {1k…10k}** → 纵向比较“**保留的特征数**”对效果的影响。

## 1) `max_features` 为啥影响

它就是**词表上限**：只保留出现最“重要/常见”的前 N 个特征（BoW/TF-IDF 会按词频或信息量选）。

- **太小**
    - 信息丢失：很多有辨识度的词/短语进不了词表 → 模型看不到关键信号。
    - 现象：训练快但准确率/F1 上不去，少数类尤其差。
- **太大**
    - 维度暴涨、**稀疏**更严重，训练慢、内存高。
    - **过拟合**风险升高：随机森林/树模型在超高维稀疏空间容易学到噪声分裂。
    - 边际收益递减：加到一定规模后，新增的长尾特征大多是噪声。
- **折中点**
    - 往往存在一个“平台区”：性能基本到顶，再增大只换来更高成本。
    - 你在 Exp-3 做的就是扫 `max_features` 找这个平台位置。

> 小招：配合 min_df/max_df 过滤极稀有或极常见（如 “the”）的词，能更快更干净。
> 

## 2) n-gram（(1,1)/(1,2)/(1,3)）为啥影响

它决定**特征粒度**：只看单词（unigram），还是把相邻词拼成短语（bigram、trigram）。

- **带来上下文/短语信息**
    - “good” vs “not good”
        - Unigram 只看见 `not` 和 `good` 两个独立词，容易误判为正向。
        - Bigram 捕捉到 `not good`，直接编码否定关系，判别更准。
    - 话题短语（如 `prime minister`, `supreme court`）常能显著提升可分性。
- **同时增加维度与稀疏度**
    - n 越大，**组合爆炸** → 特征数与稀疏性暴涨。
    - 数据不够时，很多 n-gram 只出现几次，噪声/过拟合风险上升。
- **经验**
    - (1,2) 往往是性价比最高：兼顾词与短语。
    - (1,3) 可能更好，但需要**更多样本**和**更高的 `max_features`** 才不至于“稀薄”

```python
# Step 1: Function to run the experiment
def run_experiment_tfidf_max_features(max_features):
    ngram_range = (1, 3)  # Trigram setting

    # Step 2: Vectorization using TF-IDF with varying max_features
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)

    X_train, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['category'], test_size=0.2, random_state=42, stratify=df['category'])

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Step 4: Define and train a Random Forest model
    with mlflow.start_run() as run:
        # Set tags for the experiment and run
        mlflow.set_tag("mlflow.runName", f"TFIDF_Trigrams_max_features_{max_features}")
        mlflow.set_tag("experiment_type", "feature_engineering")
        mlflow.set_tag("model_type", "RandomForestClassifier")

        # Add a description
        mlflow.set_tag("description", f"RandomForest with TF-IDF Trigrams, max_features={max_features}")

        # Log vectorizer parameters
        mlflow.log_param("vectorizer_type", "TF-IDF")
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("vectorizer_max_features", max_features)

        # Log Random Forest parameters
        n_estimators = 200
        max_depth = 15

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Step 5: Make predictions and log metrics
        y_pred = model.predict(X_test)

        # Log accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Log classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # Log confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: TF-IDF Trigrams, max_features={max_features}")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # Log the model
        mlflow.sklearn.log_model(model, f"random_forest_model_tfidf_trigrams_{max_features}")

# Step 6: Test various max_features values
max_features_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

for max_features in max_features_values:
    run_experiment_tfidf_max_features(max_features)
    
    
    设了10个参数所以就有10行
```

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2020.png)

### **Step 1: n-gram 设定**

```python
ngram_range = (1, 3)  # Trigram setting

```

- 表示用 **1-gram, 2-gram, 3-gram** 一起做特征。
- e.g. `"I love pizza"` → `["I", "love", "pizza", "I love", "love pizza", "I love pizza"]`

---

### **Step 2: 向量化 (TF-IDF)**

```python
vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)

```

- `TfidfVectorizer`：把文本转成 TF-IDF 权重矩阵。
- `ngram_range=(1,3)`：包含 unigram、bigram、trigram。
- `max_features`：只保留前 N 个最重要的特征。

---

### **Step 3: 数据划分**

```python
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_comment'], df['category'],
    test_size=0.2, random_state=42, stratify=df['category']
)

```

- 数据分成 80% 训练集，20% 测试集。
- `stratify=y` 保证类别分布一致。

然后把文本转成 TF-IDF 向量：

```python
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

```

---

### **Step 4: 模型训练 + MLflow Tracking**

```python
with mlflow.start_run() as run:

```

- 开启一个 MLflow 运行，每次实验都会单独存档。

记录元信息：

```python
mlflow.set_tag("mlflow.runName", f"TFIDF_Trigrams_max_features_{max_features}")
mlflow.set_tag("experiment_type", "feature_engineering")
mlflow.set_tag("model_type", "RandomForestClassifier")
mlflow.set_tag("description", f"RandomForest with TF-IDF Trigrams, max_features={max_features}")

```

- 方便你在 MLflow UI 里查阅。

记录参数：

```python
mlflow.log_param("vectorizer_type", "TF-IDF")
mlflow.log_param("ngram_range", ngram_range)
mlflow.log_param("vectorizer_max_features", max_features)
mlflow.log_param("n_estimators", 200)
mlflow.log_param("max_depth", 15)

```

- 把特征选择和模型的关键参数都存下来。

训练模型：

```python
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)

```

---

### **Step 5: 模型评估 + 记录结果**

预测：

```python
y_pred = model.predict(X_test)

```

评估：

```python
accuracy = accuracy_score(y_test, y_pred)
mlflow.log_metric("accuracy", accuracy)

```

- 记录 Accuracy。

分类报告：

```python
classification_rep = classification_report(y_test, y_pred, output_dict=True)
for label, metrics in classification_rep.items():
    if isinstance(metrics, dict):
        for metric, value in metrics.items():
            mlflow.log_metric(f"{label}_{metric}", value)

```

- 记录 Precision、Recall、F1 等指标（对每个类别单独保存）。

混淆矩阵：

```python
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.savefig("confusion_matrix.png")
mlflow.log_artifact("confusion_matrix.png")

```

- 画混淆矩阵并保存到 MLflow。

保存模型：

```python
mlflow.sklearn.log_model(model, f"random_forest_model_tfidf_trigrams_{max_features}")

```

- 把训练好的模型存档到 MLflow。

---

### **Step 6: 循环不同 max_features**

```python
max_features_values = [1000,2000,...,10000]

for max_features in max_features_values:
    run_experiment_tfidf_max_features(max_features)

```

- 自动跑 10 次实验：从 1000 特征 → 10000 特征。
- 每次都会在 MLflow UI 生成一条 run，方便横向对比。

---

## 🎯 总结作用

1. **核心变量：`max_features`** → 控制特征数量。
2. **保持随机森林参数不变** → 让你清楚看到 TF-IDF 词表大小对性能的影响。
3. **MLflow 记录全流程** → 方便实验可复现、可对比。

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2021.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2022.png)

第四个实验

这段代码是在 **同一份预处理后的文本数据** 上，用 **相同特征（TF-IDF 三元gram、10k 词表）+ 相同模型（RF）**，对比 **五种处理类别不均衡的策略**，并把每次训练的参数、指标、混淆矩阵、模型统统记录到 **MLflow**，方便你在 Web UI 里横向比较，选择最合适的方法。

- **Exp 2**：向量化方式 & n-gram 谁更好？
- **Exp 3**：TF-IDF 下词表多大最好？
- **Exp 4（现在）**：在固定特征与模型下，**哪种不均衡处理**能让小类表现（macro-F1、少数类 recall）最好。
1. 目标不同
- **Baseline（最早那次）**：做一个随机森林基线；而且你先把全量文本 BoW 向量化再切分（有轻微数据泄露风险）。
- **Exp 2 – BoW vs TF-IDF**：同一份预处理后的 CSV，上比较 **向量化方式**（BoW vs TF-IDF）+ **n-gram**(1,1)/(1,2)/(1,3)。
- **Exp 3 – TF-IDF max_features**：固定 **TF-IDF + 三元组(1,3)**，只比较 **词表大小 max_features**（1000→10000）。
- **Exp 4 – Imbalanced**（当前）：固定 **TF-IDF + 三元组(1,3) + max_features=10000 + 同一随机森林**，只比较 **不均衡处理策略**：
    - `class_weight='balanced'`
    - 训练集上 **SMOTE** 过采样
    - **ADASYN** 过采样
    - **RandomUnderSampler** 下采样
    - **SMOTEENN**（SMOTE+ENN 清噪）

### 1. 现实中的数据通常不均衡

比如你的 reddit 数据，某些类别的样本数量可能特别多，而某些类别只有很少。

- 如果直接训练，模型会偏向于“多数类”。
- 在分类报告里你会看到：accuracy 很高，但小类别的 recall/F1 非常差。

---

### 2. 不同策略会影响模型“是否关注少数类”

常见手段：

- **class_weight="balanced"**：给少数类更高的权重，让模型在损失函数上更重视它。
- **SMOTE / ADASYN**：人为“造”一些少数类样本（过采样），缓解数量差距。
- **RandomUnderSampler**：删掉部分多数类样本，缩小差距。
- **SMOTEENN**：结合过采样+清理噪声，平衡同时避免生成太多垃圾样本。

不同方法的效果可能完全不一样：

- 有的提升 recall，但 precision 下降 → 预测小类更多，但也容易误报。
- 有的提升 macro-F1，总体更平衡。
- 有的（如简单下采样）可能 accuracy 降低，但公平性更好。

---

### 3. 对比的意义

- **找到 trade-off**：到底要 accuracy 高，还是要每个类都公平（macro-F1 高）。
- **指导后续实验**：如果某类任务很依赖 recall（例如欺诈检测、医疗预警），就要优先挑 recall 最好的策略。
- **避免“假高分”**：如果只看 accuracy，多数类撑起来的高分其实毫无意义。

---

### 4. 放在你实验的语境里

前两个实验（BoW/TF-IDF、max_features）解决的是“**文本怎么表示**”；

而这个实验解决的是“**类别不平衡怎么处理**”。

这样你才能确认：模型性能差异不是因为 **特征表示的问题**，而是因为 **数据分布的问题**。

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2023.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2024.png)

第五个实验：

1. 安装 & 配置
- `pip install mlflow boto3 awscli optuna xgboost imbalanced-learn` 装实验追踪、超参搜索、模型与不均衡处理工具。
- `aws configure` 配置 S3 凭证（⚠️ 你把 AK/SK 粘出来了，强烈建议立刻在 AWS 上**旋转密钥**并改用环境变量，而不是明文写/交互式输入）。
- `mlflow.set_tracking_uri(...)` 把 MLflow 的追踪指向你远端的 tracking server。
1. 准备数据与文本特征
- 读 `reddit_preprocessing.csv`；把标签从 `{-1,0,1}` 改成 `{2,0,1}`；train/test 切分（`stratify` 保证各类比例稳定）。
- 用 **TF-IDF** 做 (1,3) 的 n-gram，`max_features=10000`，只在训练集上 `fit`，再 transform 测试集（避免数据泄漏）。
- 用 **SMOTE** 只在训练集上做过采样，平衡类别。
1. 定义通用的 MLflow 记录函数
- 训练传入的模型→预测→记录 `accuracy` + `classification_report` 各类别的 precision/recall/f1 到 MLflow，并把模型以 `mlflow.sklearn.log_model` 形式保存。
1. 用 **Optuna** 给 XGBoost 做超参搜索
- 目标函数里对每个 trial 采样 `n_estimators / learning_rate / max_depth`，训练 `XGBClassifier`，返回在**固定测试集**上的 `accuracy`。
- `study.optimize(..., n_trials=30)` 开始试验；你日志里能看到每个 trial 的得分，例如 Trial 4 得到 ~0.772 的 accuracy，是目前最佳。
- 结束后只用 **最优超参** 构建一个 XGBoost 模型，并通过上面的 `log_mlflow` 记录到 MLflow（Run name：`XGBoost_SMOTE_TFIDF_Trigrams`）。
1. 你看到的报错/中断
- `KeyboardInterrupt` 出现在 Trial 13，说明第 13 次试验训练耗时较长，你手动中断了（或会话超时）。这不会影响前面已完成的 trial，它们已经记录在内（但如果 MLflow UI 里没看到新 run，可能是 tracking URI 指向的不是你现在开的服务器，或浏览的是别的 Experiment）。

用 TF-IDF(1–3gram, 10k特征) + SMOTE 作为固定特征工程，对 **XGBoost** 的关键超参做 **Optuna** 搜索；把**最优模型**以及每个模型的指标上传到 **MLflow** 以便对比。

- 前面三组（Exp2/3/4）是在**同一类模型（RF）**下，分别比较“向量化方式 / 特征维度 / 不均衡处理”。
- 最新这组（**Exp5**）把关注点转到**模型与超参**（XGBoost + Optuna），特征工程和不均衡处理固定为 TF-IDF(1,3,10000) + SMOTE，仅记录**最佳**结果到 MLflow。

sorry 不跑了

第六个

**在固定的文本表示（TF-IDF，1-3gram，max_features=1000）+ SMOTE 的前提下，用 Optuna 给 LightGBM 做超参数搜索，并把每个 trial 作为一次 MLflow run 记录下来**。目标是看看 **LightGBM+调参** 能否比你前面那些固定模型/固定参数（RandomForest、TF-IDF 参数网格、各类不均衡处理、以及第5个实验的 XGBoost+Optuna）更好。

# 代码在做什么（逐步）

1. **标签重映射**：把 {-1,0,1} → {2,0,1}，避免负数标签（有些库不喜欢负标签）。
2. **向量化**：`TfidfVectorizer(ngram_range=(1,3), max_features=1000)` 把文本转成稀疏特征。
3. **SMOTE 上采样**：用少数类合成样本来平衡类别。
4. **切分数据**：将平衡后的数据按 8/2 切分成训练/测试集。
5. **Optuna 调参 LightGBM**：搜索如 `n_estimators、learning_rate、max_depth、num_leaves、min_child_samples、colsample_bytree、subsample、reg_alpha、reg_lambda` 等；
    
    每个 trial：
    
    - 生成一个 LGBMClassifier；
    - 训练并在测试集上算 accuracy；
    - **用 MLflow 记录**参数、准确率、分类报告、并保存模型。
6. **完成后**：用 Optuna 的可视化看 **参数重要性** 和 **优化历史**，并把 **best_params** 再训练一次，作为“最佳模型”记录到 MLflow。

# 和前面的实验有什么区别？

- **对象不同**：
    - 前面 1~3：对比“向量化方式/超参数”（BoW vs TF-IDF、n-gram、max_features）。
    - 第4：对比**不均衡处理策略**（class_weight、SMOTE/ADASYN/Under/SMOTEENN）。
    - 第5：先锁定“向量化+SMOTE”，对 **XGBoost** 做 Optuna 调参。
    - **第6（当前）**：同样锁定“向量化+SMOTE”，但把算法换成 **LightGBM** 并调参，对比 **XGBoost vs LightGBM** 哪个在你这个文本任务上更强、用什么参数最好。
- **搜索空间不同**：XGBoost 和 LightGBM 的关键超参不一样（比如 LGBM 的 `num_leaves、min_child_samples` 等）。
- **实验目的**：前面是在找**特征表示/数据处理**的最佳组合；第5/6是在同一特征与采样策略下，比较**不同梯度提升框架+调参**的上限表现。

低七个

### 🔹前一个实验（第六个）

- **目标**：用 Optuna 超参数搜索，找到最优的 LightGBM 参数。
- **重点**：
    - 只用 **LightGBM** 作为模型。
    - 通过 **Optuna** 不断试不同参数组合（n_estimators, learning_rate, num_leaves, etc.）。
    - 用 **SMOTE** 做类别平衡。
    - 用 **MLflow** 记录每次实验（参数、指标、模型）。
    - 结果是 → 找到一个最优的 LightGBM 配置。

换句话说，这个实验是在「调参 + 单模型优化」。

---

### 🔹你现在的实验（Stacking）

- **目标**：用集成学习（Stacking）提升模型表现。
- **重点**：
    - 使用 **多个基模型 (base learners)**：LightGBM + Logistic Regression。
    - 使用 **一个元学习器 (meta learner)**：KNN。
    - 训练方式是：
        - Base models 先学习特征。
        - Meta learner（KNN）再基于 base models 的预测结果来做最终预测。
    - 这里没有用 SMOTE、没有用 Optuna、也没有记录到 MLflow。
    - 结果是 → 用组合模型来试试能不能比单一 LightGBM 效果更好。

## 🧪 实验 1：文本向量化对比（BoW vs TF-IDF，不同 n-gram）

- **做了什么**
    - 用 BoW（词袋模型）和 TF-IDF 两种方式表示文本。
    - 尝试了 unigram、bigram、trigram 三种 n-gram 设置。
    - 用 RandomForest 做分类。
- **目的**
    - 对比文本向量化方法和 n-gram 范围对模型性能的影响。
- **区别**
    - 这是最基础的实验，核心关注 **特征工程（文本表示方法）**。

---

## 🧪 实验 2：固定 n-gram，对比 max_features（特征数量）

- **做了什么**
    - 使用 TF-IDF + trigram。
    - 改变 `max_features`（1000–10000），限制词汇表大小。
- **目的**
    - 观察高维特征 vs 限制维度时模型表现的差异。
- **区别**
    - 还是 RandomForest，但这次关注的是 **特征数量对性能的影响**。

---

## 🧪 实验 3：处理类别不均衡（Class weights、Oversampling、SMOTE、ADASYN 等）

- **做了什么**
    - 在 TF-IDF 基础上，用不同的不均衡处理策略。
    - 再用 RandomForest 训练。
- **目的**
    - 解决 “多数类压制少数类” 的问题，避免模型只预测大类。
- **区别**
    - 前两个实验默认数据分布不变，这个实验开始关注 **数据层面的平衡**。

---

## 🧪 实验 4：在 MLflow 中记录不均衡实验

- **做了什么**
    - 把实验 3 的各种不均衡策略（class weight、oversampling、SMOTE+ENN 等）运行结果写入 MLflow。
- **目的**
    - 不只是跑实验，还要把结果系统化管理和对比。
- **区别**
    - 和实验 3 的技术内容类似，但增加了 **实验追踪和可视化管理**。

---

## 🧪 实验 5：XGBoost + Optuna 超参数搜索

- **做了什么**
    - 使用 TF-IDF + SMOTE 处理过的数据。
    - 用 Optuna 自动搜索 XGBoost 的超参数（n_estimators、learning_rate、max_depth）。
    - 只记录最优结果到 MLflow。
- **目的**
    - 找到 XGBoost 在当前数据上的最佳配置。
- **区别**
    - 前面是 “固定参数 + 对比方法”，这次开始做 **自动化超参数调优**。

---

## 🧪 实验 6：LightGBM + Optuna 超参数搜索

- **做了什么**
    - 思路和实验 5 类似，但模型换成 LightGBM。
    - 搜索范围更大（n_estimators、num_leaves、colsample_bytree、subsample、正则化项等）。
- **目的**
    - 验证 LightGBM 是否比 XGBoost 在这个任务上表现更好。
- **区别**
    - 和实验 5 的区别主要是 **模型框架不同**（LightGBM vs XGBoost）。

---

## 🧪 实验 7：Stacking 集成学习

- **做了什么**
    - 用多个模型（LightGBM、Logistic Regression）作为 base learners。
    - 用一个 meta learner（KNN）整合前面模型的预测结果。
- **目的**
    - 集成多个模型，利用不同模型的互补性提升表现。
- **区别**
    - 前 1–6 都是 “单模型”，这里是 **模型集成**，难度和复杂度更高。

---

## 📊 总结（七个实验脉络）

1. **实验 1–2**：探索特征工程 → 文本表示（BoW vs TF-IDF，n-gram，特征数）。
2. **实验 3–4**：解决数据问题 → 类别不均衡处理 + 实验记录。
3. **实验 5–6**：模型层面对比 → XGBoost vs LightGBM + 超参优化。
4. **实验 7**：进一步提升 → 模型集成（Stacking）。

👉 这七个实验构成了一个 **完整的机器学习流程**：

文本表示 → 数据均衡 → 模型选择与调优 → 集成提升 → 实验管理。

**第六Build ML pipeline using DVC**

### ① Data Preprocessing（数据预处理）

- 清理文本（去停用词、特殊符号、lowercase）
- 标签重映射（[-1,0,1] → [2,0,1]）
- TF-IDF/BoW 向量化（ngram, max_features）
    
    👉 对应你前几个实验里 **文本表示、max_features 对比**。
    

---

### ② Model Building（模型构建）

- 建立基线模型（LogReg / RF / NB / SVM）
- 引入更强的模型（XGBoost / LightGBM）
- 用 Optuna 调优
    
    👉 对应实验 **3–6**：基线 → 改进 → 调优 → 集成。
    

---

### ③ Model Evaluation with MLflow（模型评估）

- 在 test set 上预测
- 记录 accuracy、F1、classification report、混淆矩阵
- 每个实验 run 都 log 到 MLflow
    
    👉 对应你前面截图里在 MLflow 中可对比不同参数/模型结果。
    

---

### ④ Model Register with MLflow（模型注册）

- 把表现最好的模型（比如 LightGBM Optuna 最优参数）用 MLflow Registry 存档
- 赋予版本号（v1, v2, …），方便部署和回滚
    
    👉 这一步是实验完成 → 上线前的 **成果固化**。
    

**组件化 ML pipeline（用 DVC + MLflow 搭起来）**，目的是让你的整个机器学习项目像一个“软件工程项目”一样，**可复现、可追踪、可扩展**。

和你前面做的七个实验相比：

- 前面七个实验更多是“探索性”（试不同的向量化、采样方法、算法、调参）。
- 现在做 pipeline，是把这些步骤**工程化**，让别人（或者未来的你自己）能一键复现，并且持续改进。

---

### 为什么要弄这些组件？

1. **Data Ingestion Component**
    - 负责把原始数据读进来（比如从 CSV/数据库/接口）。
    - 好处：数据源如果变了，只要改 ingestion，不用整个 pipeline 改。
2. **Data Preprocessing Component**
    - 负责清洗、分词、向量化（TF-IDF、BoW）、特征工程。
    - 好处：保证每次训练的数据处理一致，避免“手工跑一次处理不一样”。
3. **Model Building Component**
    - 定义并训练模型（Logistic Regression、LightGBM、Stacking 等）。
    - 好处：模型换了，pipeline 也能自动更新并追踪。
4. **Model Evaluation Component with MLflow**
    - 统一评估指标（accuracy、F1、混淆矩阵）。
    - 用 MLflow 自动记录每次实验的结果。
    - 好处：可以比较哪次模型更好，而不是手工记笔记。
5. **Model Register Component with MLflow**
    - 把最优模型存到 MLflow 的 Model Registry。
    - 好处：方便部署和版本控制（比如你可以回滚到旧模型）。
6. **DVC 管理 pipeline**
    - DVC（Data Version Control）可以像 Git 一样管理数据、模型和中间结果。
    - 好处：团队协作时，保证每个人跑出来的结果一致，方便 CI/CD。

### 前面 7 个实验

- **核心目标**：找到“效果好的模型”。
- **特点**：
    1. 是 **探索性实验** → 比较文本表示方法（BoW / TF-IDF）、不同 n-gram、max_features。
    2. 研究 **类别不均衡的处理策略**（SMOTE、undersampling、class weight 等）。
    3. 比较 **不同算法**（Random Forest、SVM、LightGBM、XGBoost 等），再用 Optuna 调参。
    4. 尝试 **集成方法**（Stacking、Boosting）提升性能。
    5. 每次实验都用 MLflow 记录指标，主要是“实验管理”。

👉 本质：这是 **建模阶段的探索**，你在回答“哪个模型和配置更好？”

---

### 现在的 DVC + MLflow Pipeline

- **核心目标**：让实验 **工程化、可复现、可部署**。
- **特点**：
    1. **组件化** → 分成 Data Ingestion、Preprocessing、Model Building、Evaluation、Register。
    2. **可复现** → 不管你还是别人，下个月再跑一遍，结果完全一样。
    3. **可追踪** → DVC 跟踪数据/模型变化，MLflow 跟踪实验结果。
    4. **可部署** → 最优模型直接进入 MLflow Model Registry，方便上线/回滚。
    5. **可扩展** → 如果以后换数据、加新模型，不需要推倒重来，只要更新对应组件。

👉 本质：这是 **MLOps 阶段的落地**，你在回答“怎么让好的模型能稳定复现、上线和迭代？”

---

### 简单比喻

- **7 个实验** = 在厨房里尝不同的菜谱，找到哪道菜最好吃。
- **DVC + MLflow pipeline** = 把这道菜写成餐厅的 SOP（标准化流程），保证每次端出来的菜都一样，还能随时换新菜单。

### 1. **DVC (Data Version Control)**

- **是什么**：
    
    类似于 **Git**，但专门用来管理 **数据和模型文件**。
    
- **为什么要用**：
    - Git 可以管代码，但几百 MB/GB 的数据、模型不好管。
    - DVC 可以像 Git 一样打 tag、做版本回溯（比如：回到上个月的数据、模型版本）。
    - 方便团队协作：别人一键 `dvc pull` 就能获得同样的数据/模型。
- **场景**：
    - 记录数据集的变化（v1 → v2 → v3）。
    - 跟踪训练得到的模型（model_v1.pkl, model_v2.pkl）。
    - 在 pipeline 中定义数据 → 预处理 → 训练 → 评估 → 注册的 **步骤依赖**。

---

### 2. **YAML 文件**

- **是什么**：一种配置文件格式（像 JSON，但更简洁）。
- **为什么要用**：
    
    在 ML pipeline 里，用 YAML 文件来定义 **步骤、依赖、输入输出**。
    
- **例子**：DVC 的 `dvc.yaml` 文件可能长这样：
    
    ```yaml
    stages:
      preprocess:
        cmd: python src/preprocess.py
        deps:
          - data/raw/reddit.csv
          - src/preprocess.py
        outs:
          - data/processed/clean_reddit.csv
    
      train:
        cmd: python src/train.py
        deps:
          - data/processed/clean_reddit.csv
          - src/train.py
        outs:
          - models/random_forest.pkl
    
    ```
    
    👉 含义：
    
    - 如果原始数据或 `preprocess.py` 改了，DVC 会自动重跑 **preprocess** 和 **train**。
    - 如果只有训练代码改了，就只重跑 **train**。
    - 相当于一个 **自动化的实验流水线**。

---

### 3. **PKL 文件**

- **是什么**：Python 的序列化文件（Pickle 格式）。
- **为什么要用**：
    - 训练好的模型（比如 RandomForestClassifier）不能直接保存为 `.csv`，所以用 Pickle 保存成 `.pkl` 文件。
    - 下次预测时直接 `joblib.load("model.pkl")` 或 `pickle.load()` 就能加载模型，而不用重新训练。
- **场景**：
    - 模型上线部署。
    - 保存/加载特征工程对象（如 `TfidfVectorizer.pkl`）。

---

### 4. **为什么后面还要弄 MLflow？**

- DVC 解决的是：**数据 & 训练过程的可复现**。
- 但实验还有更复杂的需求：
    - 不同参数、模型效果要记录（Accuracy, F1, 混淆矩阵）。
    - 要有一个地方管理所有实验的结果。
    - 要能把最优模型 **注册/部署**，方便生产环境调用。
- **MLflow 就是做实验管理 + 模型管理的工具**：
    - `mlflow.log_param()` → 记录参数。
    - `mlflow.log_metric()` → 记录结果。
    - `mlflow.sklearn.log_model()` → 保存模型，并推到 **Model Registry**。

👉 结合起来：

- **DVC** = 数据/过程的 Git
- **MLflow** = 实验 & 模型的日志簿/仓库
- **PKL** = 模型存档格式
- **YAML** = 定义 pipeline 的配置脚本

---

✅ 总结一句：

前面的 7 个实验只是“找到好模型”，

DVC + YAML + PKL + MLflow = 把这个模型做成 **标准化工程流水线**，保证任何人、任何时候都能复现和上线。

### 三者的关系

- **DVC**：是整个项目的“流程管理 + 版本控制”工具。
- **YAML**：是 DVC 用来定义流程的配置文件，规定数据流和产物。
- **PKL**：是流程的产物之一，保存的就是训练好的模型。

👉 可以这样理解：

- **DVC = 管理者**
- **YAML = 管理者手里的流程图**
- **PKL = 流程产出的成品（模型文件）**

建好mlflow了

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2025.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2026.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2027.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2028.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2029.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2030.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2031.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2032.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2033.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2034.png)

[https://www.notion.so](https://www.notion.so)

AIzaSyA7kKnxD4jF9cKOFhtMM0y4WgoJF5Tl108

cicd

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2035.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2036.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2037.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2038.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2039.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2040.png)

```markdown
# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access(you)
	0. iam(有access key)
	1. EC2 access : It is virtual machine !
(然后就会进入mahine）
	2. ECR: Elastic Container registry to save your docker image in aws
	
	294892597101.dkr.ecr.us-east-1.amazonaws.com/mlops-pipeline

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
    - Save the URI: 
    294892597101.dkr.ecr.us-east-1.amazonaws.com/mlops-pipeline

	
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

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app

```

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2041.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2042.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2043.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2044.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2045.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2046.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2047.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2048.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2049.png)

52.71.42.112

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2050.png)

![image.png](%E4%BD%BF%E7%94%A8%20Python%E3%80%81AWS%E3%80%81Docker%20%E7%9A%84%20MLOps%20%E7%AE%A1%E9%81%93%20%E2%80%93%20YouTube%20%E8%A7%82%E4%BC%97%E6%83%85%E7%BB%AA%20276307fdbf47807e8e4ac3141fa0821c/image%2051.png)