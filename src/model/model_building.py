import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

# logging configuration 日志配置
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:  #读取参数
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


def load_data(file_path: str) -> pd.DataFrame:  #加载数据
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values 填充任何NaN值
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:  #应用TF-IDF
    """Apply TF-IDF with ngrams to the data."""  #应用TF-IDF
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        # Perform TF-IDF transformation 执行TF-IDF变换
        X_train_tfidf = vectorizer.fit_transform(X_train)

        logger.debug(f"TF-IDF transformation complete. Train shape: {X_train_tfidf.shape}")

        # Save the vectorizer in the root directory 保存向量化器
        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug('TF-IDF applied with trigrams and data transformed')
        return X_train_tfidf, y_train
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise


def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:  #训练LightGBM模型
    """Train a LightGBM model."""
    try:
        best_model = lgb.LGBMClassifier( #在notebook里面的时候刚刚也在MLflow找到这个超参数的优化结果。
            objective='multiclass',
            num_class=3,
            metric="multi_logloss",
            is_unbalance=True,
            class_weight="balanced",
            reg_alpha=0.1,  # L1 regularization 正则化
            reg_lambda=0.1,  # L2 regularization 正则化
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        best_model.fit(X_train, y_train)
        logger.debug('LightGBM model training completed')
        return best_model
    except Exception as e:
        logger.error('Error during LightGBM model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:  #保存模型
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def get_root_directory() -> str:    #获取根目录
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():  #主函数
    try:
        # Get root directory and resolve the path for params.yaml 获取根目录
        root_dir = get_root_directory()

        # Load parameters from the root directory 从根目录加载参数
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])

        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        # Load the preprocessed training data from the interim directory 从 interim 目录加载预处理后的训练数据
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

        # Apply TF-IDF feature engineering on training data 应用TF-IDF特征工程
        X_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)

        # Train the LightGBM model using hyperparameters from params.yaml 使用params.yaml中的超参数训练LightGBM模型
        best_model = train_lgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators)

        # Save the trained model in the root directory 保存训练好的模型
        save_model(best_model, os.path.join(root_dir, 'lgbm_model.pkl'))

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
