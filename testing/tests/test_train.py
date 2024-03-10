import random
import string
import sys
from datetime import date
from typing import Optional

import boto3
import pandas as pd
import pytest
import sqlalchemy
from sklearn.ensemble import RandomForestClassifier

sys.path.append('..')
import config
import train

PG_CONFIG = config.PG_CONFIG
S3_CONFIG = config.S3_CONFIG
BUCKET = "test-bucket-lizvladi"


@pytest.mark.download
def test_download_data_1():
    data = train.download_data()
    assert type(data) is pd.DataFrame

    
@pytest.mark.download
def test_download_data_2():
    data = train.download_data()
    assert data.shape[0] != 0

    
@pytest.mark.skip(reason="no way of currently testing this")
def test_train_model():
    clf, metrics = train_model(X_train, X_test, y_train, y_test)
    assert (type(clf) is RandomForestClassifier) and (
        type(metrics) is dict)

    
@pytest.mark.s3
def test_s3_connection_1():
    # Подключиьтся к S3
    s3_resource = boto3.resource(
        service_name="s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=S3_CONFIG["aws_access_key_id"],
        aws_secret_access_key=S3_CONFIG["aws_secret_access_key"],
    )
    # Проверить, что подключение работает (возвращает статус 200)
    response_code = s3_resource.meta.client.head_bucket(
        Bucket=S3_CONFIG["bucket"])['ResponseMetadata']['HTTPStatusCode']
    
    assert response_code == 200


@pytest.fixture()
def s3_context():
    # Подключиьтся к S3
    s3_resource = boto3.resource(
        service_name="s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=S3_CONFIG["aws_access_key_id"],
        aws_secret_access_key=S3_CONFIG["aws_secret_access_key"],
    )
    return s3_resource


@pytest.mark.s3
def test_s3_connection_2(s3_context):
    response_code = s3_context.meta.client.head_bucket(
        Bucket=S3_CONFIG["bucket"])['ResponseMetadata']['HTTPStatusCode']
    assert response_code == 200


@pytest.mark.dependency(on=['test_s3_connection'])
def test_save_results_1():
    # Создать тестовый классификатор
    clf = RandomForestClassifier()
    # Создать ему тестовое имя
    timestamp = date.today().strftime("%d%m%Y")
    model_name = f"test_{timestamp}"
    # Сохранить результаты обучения на S3
    train.save_results(clf=clf, model_name=model_name)

    s3_resource = boto3.resource(
        service_name="s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=S3_CONFIG["aws_access_key_id"],
        aws_secret_access_key=S3_CONFIG["aws_secret_access_key"],
    )

    try:
        s3_resource.Object(bucket_name=BUCKET, key=model_name).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            pytest.fail(f"The object does not exist")
        else:
            pytest.fail(f"Something went wrong: {e}")


@pytest.fixture()
def save_context():
    # Создать тестовый классификатор
    clf = RandomForestClassifier()
    # Создать ему тестовое имя
    timestamp = date.today().strftime("%d%m%Y")
    model_name = f"test_{timestamp}"
    # Сохранить результаты обучения на S3
    train.save_results(clf, model_name)

    # Передать имя модели
    yield model_name


@pytest.mark.dependency(on=['test_s3_connection'])
def test_save_results_2(save_context, s3_context):
    # Прочитать данные из фикстур
    model_name = save_context
    s3_resource = s3_context

    # Попробовать прочитать модель с S3 
    try:
        s3_resource.Object(bucket_name=BUCKET, key=model_name).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            pytest.fail(f"The object does not exist")
        else:
            pytest.fail(f"Something went wrong: {e}")


def test_save_metrics_1():
    # Создать словарь метрик с рандомными именами моделей
    metrics = {
        "model_name": ''.join(random.choices(string.ascii_uppercase, k=3)),
        "precision": 0.90,
        "recall": 0.8,
        "f1_score": 0.88
    }
    # Сохранить метрики в PG
    train.save_metrics(metrics)

    # Подключиться к PG
    engine = sqlalchemy.create_engine(
        "postgresql://postgres:postgres@127.0.0.1:5432/postgres")
    # Прочитать имена моделей в таблице результатов
    models = pd.read_sql("SELECT DISTINCT(model_name) from wine_results", con=engine)
    models = models["model_name"].values
    engine.dispose()

    # Убедиться, что имена моделей из словаря выше есть в PG
    assert metrics["model_name"] in models


@pytest.fixture()
def models(request):
     # Подключиться к PG
    engine = sqlalchemy.create_engine(
        "postgresql://postgres:postgres@127.0.0.1:5432/postgres")
    # Прочитать имена моделей в таблице результатов
    models = pd.read_sql("SELECT DISTINCT(model_name) from wine_results", con=engine)
    models = models["model_name"].values
    
    # Закрыть соединение с PG
    def fin():
        engine.dispose()

    request.addfinalizer(fin)

    return models


def test_save_metrics_2(models):
    metrics = {
        "model_name": ''.join(random.choices(string.ascii_uppercase, k=3)),
        "precision": 0.90,
        "recall": 0.8,
        "f1_score": 0.88
    }
    train.save_metrics(metrics)
    
    assert metrics["model_name"] in models
    
    
@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
def test_eval(test_input, expected):
    assert eval(test_input) == expected