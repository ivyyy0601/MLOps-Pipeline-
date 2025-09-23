import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


def _vectorize_as_df(texts):
    # 预处理
    pre = [preprocess_comment(t) for t in texts]

    # 向量化 -> DataFrame（带列名）
    Xs = vectorizer.transform(pre)
    try:
        feats = vectorizer.get_feature_names_out()
    except AttributeError:
        feats = vectorizer.get_feature_names()
    import pandas as pd
    X = pd.DataFrame(Xs.toarray(), columns=feats)

    # 读取模型记录的输入 schema，保证列集合 & 顺序一致（缺的补0，多的丢）
    expected = [fld.name for fld in model.metadata.get_input_schema().inputs]
    X = X.reindex(columns=expected, fill_value=0)

    # 保险起见，转成 float
    return X.astype("float64")


# Load the model and vectorizer from the model registry and local storage 从模型注册表和本地存储加载模型和向量化器
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    # Set MLflow tracking URI to your server 设置MLflow跟踪URI为你的服务器
    mlflow.set_tracking_uri("http://ec2-52-71-42-112.compute-1.amazonaws.com:5000/")  # Replace with your MLflow tracking URI 替换为你的MLflow跟踪URI
    client = MlflowClient() #创建MLflow客户端
    model_uri = f"models:/{model_name}/{model_version}" #创建模型URI
    model = mlflow.pyfunc.load_model(model_uri) #加载模型
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file) #加载向量化器
   
    return model, vectorizer #返回模型和向量化器


 #Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "2", "./tfidf_vectorizer.pkl")  # Update paths and versions as needed 根据需要更新路径和版本

#r如果你没有开启MLflow Registry，这里需要改成你的本地路径
# def load_model(model_path, vectorizer_path): 加载模型和向量化器
#     """Load the trained model."""
#     try:
#         with open(model_path, 'rb') as file:
#             model = pickle.load(file) #加载模型
        
#         with open(vectorizer_path, 'rb') as file:
#             vectorizer = pickle.load(file)        
      
#         return model, vectorizer #返回模型和向量化器
#     except Exception as e:
#         raise #抛出异常

# # Initialize the model and vectorizer
# model, vectorizer = load_model("./lgbm_model.pkl", "./tfidf_vectorizer.pkl")  

@app.route('/')
def home():
    return "Welcome to our flask api"



@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    items = data.get('comments')
    if not items:
        return jsonify({"error": "No comments provided"}), 400
    try:
        texts = [it['text'] for it in items]
        stamps = [it['timestamp'] for it in items]
        X = _vectorize_as_df(texts)

        print("[DEBUG] type(X)=", type(X), "shape=", X.shape)

        preds = np.asarray(model.predict(X)).tolist()
        return jsonify([
            {"comment": t, "sentiment": p, "timestamp": ts}
            for t, p, ts in zip(texts, preds, stamps)
        ])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    if not comments:
        return jsonify({"error": "No comments provided"}), 400
    try:
        X = _vectorize_as_df(comments)

        # 调试日志（确认不是 numpy）
        print("[DEBUG] type(X)=", type(X), "shape=", X.shape)
        print("[DEBUG] first 5 cols=", list(X.columns[:5]))

        preds = model.predict(X)
        preds = np.asarray(preds).tolist()
        return jsonify([{"comment": c, "sentiment": p} for c, p in zip(comments, preds)])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500



@app.route('/generate_chart', methods=['POST'])
def generate_chart(): #生成图表
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400 #如果sentiment_counts为空，返回错误

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative'] #标签
        sizes = [
            int(sentiment_counts.get('1', 0)), #正数
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0)) #负数
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero") #如果情感计数总和为0，抛出错误
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red #蓝色、灰色、红色

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle. #确保饼图是圆形的

        # Save the chart to a BytesIO object
        img_io = io.BytesIO() #创建一个字节IO对象
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close() #关闭图表

        # Return the image as a response
        return send_file(img_io, mimetype='image/png') #返回图片
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}") #记录错误
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500 #返回错误


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments') #获取评论

        if not comments:
            return jsonify({"error": "No comments provided"}), 400 #如果comments为空，返回错误

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments] #预处理评论

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments) #将所有评论合并为一个字符串

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black', #背景颜色
            colormap='Blues',
            stopwords=set(stopwords.words('english')), #停用词
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO() #创建一个字节IO对象
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')        
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}") #记录错误
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500 #返回错误

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)