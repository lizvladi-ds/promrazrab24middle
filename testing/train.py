import pickle
from datetime import date
from typing import Any, Dict, Tuple

import boto3
import numpy as np
import pandas as pd
import sqlalchemy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

import testing.config as config

S3_CONFIG = config.S3_CONFIG
PG_CONFIG = config.PG_CONFIG


def download_data():
    """
    Чтение всех данных из таблицы wine_data.
    """
    # Подключение к БД (SQLAlchemy)
    engine = sqlalchemy.create_engine(
        url=
        f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:"
        f"{PG_CONFIG['port']}/{PG_CONFIG['database']}"
    )
    data = pd.read_sql_query("SELECT * FROM wine_data", engine.connect())
    engine.dispose()
    return data


def preprcocess_and_split_data(
        data: pd.DataFrame) -> Tuple[np.array, np.array, pd.Series, pd.Series]:
    """
    Деление данных на train/test 0.2 и нормализация данных.
    """
    target = data["quality"].copy()
    del data["quality"]
    features = data

    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=target)

    norm = Normalizer()
    X_train_norm = norm.fit_transform(X_train)
    X_test_norm = norm.transform(X_test)

    return X_train_norm, X_test_norm, y_train, y_test


def train_model(X_train: np.array, X_test: np.array, y_train: pd.Series,
                y_test: pd.Series) -> Tuple[Any, Dict[str, Any]]:
    """
    Обучение модели и запись результатов в словарь метрик.
    """
    # Обучение модели
    clf = LogisticRegression(solver="liblinear")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Сбор метрик
    metrics = {
        "precision": precision_score(y_test, y_pred, average="micro"),
        "recall": recall_score(y_test, y_pred, average="micro"),
        "f1_score": f1_score(y_test, y_pred, average="micro")
    }

    # Добавление метрик в словарь
    timestamp = date.today().strftime("%d%m%Y")
    model_name = f"logreg_{timestamp}"
    metrics["model_name"] = model_name

    return clf, metrics


def save_metrics(metrics):
    """
    Сохранение метрик в таблицу wine_results.
    """
    # Подключение к БД (SQLAlchemy)
    engine = sqlalchemy.create_engine(
        url=
        f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:"
        f"{PG_CONFIG['port']}/{PG_CONFIG['database']}"
    )
    # Вызов описания таблицы
    table = sqlalchemy.Table("wine_results",
                             sqlalchemy.MetaData(),
                             autoload_with=engine)
    # Подготовка insert statement
    stmt = sqlalchemy.insert(table).values(**metrics)
    # Исполнение скрипта
    connection = engine.connect()
    connection.execute(stmt)
    connection.commit()
    connection.close()
    engine.dispose()


def save_results(clf, model_name):
    """
    Сохранение модели на S3.
    """
    s3_resource = boto3.resource(
        service_name="s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=S3_CONFIG["aws_access_key_id"],
        aws_secret_access_key=S3_CONFIG["aws_secret_access_key"],
    )
    pickle_byte_obj = pickle.dumps(clf)
    s3_resource.Object(bucket_name=S3_CONFIG["bucket"],
                       key=model_name).put(Body=pickle_byte_obj)
