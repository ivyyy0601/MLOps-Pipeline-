import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# Logging configuration 定义日志器（logger），名字是 data_ingestion，级别是 DEBUG（所有信息都会被记录）
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)
#设置两个 handler，一个往 控制台打印日志，一个写入 文件。

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#日志格式：时间 - 模块名 - 等级 - 消息。
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)
#把 handler 加到 logger 上。

def load_params(params_path: str) -> dict: #读取参数   从 params.yaml 文件加载配置
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise
#从 params.yaml 文件加载配置（例如 test_size）。

def load_data(data_url: str) -> pd.DataFrame: #. 加载数据
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url) #从给定的 URL 加载数据。
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame: #预处理数据
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    try:
        # Removing missing values 删除缺失值
        df.dropna(inplace=True)
        # Removing duplicates 删除重复值
        df.drop_duplicates(inplace=True)
        # Removing rows with empty strings 删除 包含空字符串 的行
        df = df[df['clean_comment'].str.strip() != '']
        
        logger.debug('Data preprocessing completed: Missing values, duplicates, and empty strings removed.') #预处理完成
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist.""" #保存数据
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        
        # Create the data/raw directory if it does not exist #创建 data/raw 目录
        os.makedirs(raw_data_path, exist_ok=True)
        
        # Save the train and test data #保存训练和测试数据
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        
        logger.debug('Train and test data saved to %s', raw_data_path) #训练和测试数据保存到 raw_data_path
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():#下载数据 → 清洗 → 划分训练集/测试集 → 保存。
    try:
        # Load parameters from the params.yaml in the root directory    从 params.yaml 里取出 test_size 参数
        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size = params['data_ingestion']['test_size']
        
        # Load data from the specified URL  其实不是youtube数据是rediit数据。。。
        df = load_data(data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
        
        # Preprocess the data
        final_df = preprocess_data(df)
        
        # Split the data into training and testing sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        
        # Save the split datasets and create the raw folder if it doesn't exist
        save_data(train_data, test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))
        
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
