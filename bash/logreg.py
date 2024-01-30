import argparse
import logging
import numpy as np
import pandas as pd
import sys

from typing import Tuple

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Создание логгера.
_LOG = logging.getLogger("logreg.log")
_LOG.setLevel(logging.DEBUG)
# Определние формата логирования.
ch = logging.FileHandler("logreg.log")
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%d-%m-%Y %I:%M:%S"
)
# Добавление формата в созданный логгер.
ch.setFormatter(formatter)
_LOG.addHandler(ch)


def download_data(test_size:float) -> Tuple[pd.DataFrame, pd.DataFrame, np.array, np.array]:
    
    wine_data = datasets.load_wine()
    
    X = pd.DataFrame(wine_data['data'], columns = wine_data['feature_names'])
    y = wine_data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: np.array, C:float, penalty:str) -> LogisticRegression:
    
    model = LogisticRegression(C=C, penalty=penalty, solver='liblinear')
    model.fit(X_train, y_train)

    return model
    

def predict(model: LogisticRegression, X_test: pd.DataFrame, y_test: np.array) -> float:
    
    y_pred = model.predict(X_test)
    result = f1_score(y_test, y_pred, average='macro')
    
    return result


if __name__=="__main__":

    # Чтение аргументов из командной строки.
    parser = argparse.ArgumentParser(description='Train logistic regression')
    parser.add_argument('test_size', type=float, help='Data split test size')
    parser.add_argument('C', type=float, help='Logistic regression C parametr')
    parser.add_argument('penalty', type=str, help='Logistic regression penalty parameter')
    args = parser.parse_args()

    # Обучение модели.
    X_train, X_test, y_train, y_test =  download_data(args.test_size)
    model = train_model(X_train, y_train, args.C, args.penalty)
    result = predict(model, X_test, y_test)
    
    # Логирование результатов.
    _LOG.info('C: %.2f, penalty: %s, test size: %.1f, result: %.2f', args.C, args.penalty, args.test_size, result)