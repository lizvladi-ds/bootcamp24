import io
import json
import logging
import numpy as np
import pandas as pd
import pickle
import re

from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from typing import Any, Dict, List, Literal

import nltk
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

DEFAULT_ARGS = {
    "owner": "Elizaveta Gavrilova",
    "email": "example@gmail.com",
    "email_on_failure": False,
    "email_on_retry": False,
    "retry": 3,
    "retry_delay": timedelta(minutes=1)
}

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = "lizvladi-mlops"
FEATURES = "text"
TARGET = "tags"

models = dict(
    zip(["logreg", "nb"], [
        LogisticRegression(max_iter=1000),
        MultinomialNB(),
    ]))

dag = DAG(dag_id="bootcamp_dag",
          schedule_interval="0 1 * * *",
          start_date=days_ago(2),
          catchup=False,
          tags=["mlops"],
          default_args=DEFAULT_ARGS)


def init() -> Dict[str, Any]:
    """
    Step0: Pipeline initialisation.
    """
    info = {}
    info["start_tiemstamp"] = datetime.now().strftime("%Y%m%d %H:%M")
    info["dataset_end"] = datetime.now().strftime("%Y-%m-%d")
    # Импользуем данные с сегодня на 2 года назад.
    info["dataset_start"] = (datetime.now() -
                           timedelta(365 * 2)).strftime("%Y-%m-%d")
    return info


def get_data_from_postgres(**kwargs) -> Dict[str, Any]:
    """
    Step1: Read data from news PG table and save to S3.
    """
    ti = kwargs["ti"]
    info = ti.xcom_pull(task_ids="init")
    info["data_download_start"] = datetime.now().strftime("%Y%m%d %H:%M")

    # Использовать созданный ранее PG connection
    pg_hook = PostgresHook("pg_connection")
    con = pg_hook.get_conn()

    # Прочитать данные из таблицы news за заданный период
    start = info["dataset_start"]
    end = info["dataset_end"]
    data = pd.read_sql_query(
        f"SELECT * FROM news WHERE date BETWEEN '{start}' and '{end}'", con)

    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connection")

    # Сохранить файл в формате pkl на S3
    pickle_byte_obj = pickle.dumps(data)
    s3_hook.load_bytes(pickle_byte_obj, "bootcamp/datasets/news.pkl", 
                          bucket_name=BUCKET, replace=True)

    _LOG.info("Data download finished.")

    info["data_download_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    return info


def preprocess_text(text: str) -> str:
    """
    Preprocess text.
    
    :param text: text for preprocessing
    :return: lower-case text without digits and special symbols.
    """
    text = text.lower()

    text = re.sub(r'\d+', '', text)  # убрать цифры
    text = re.sub(r'[^\w\s]', '', text)  # убрать специальные символы

    tokens = nltk.word_tokenize(text, language="russian")

    return ' '.join(tokens)


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove russian stopwords.
    
    :param tokens: list of tokens.
    :return: list of tokens without stopwords.
    """
    stop_words = set(stopwords.words("russian"))
    clean_tokens = [token for token in tokens if token not in stop_words]

    return clean_tokens


def prepare_data(**kwargs) -> Dict[str, Any]:
    """
    Step 2: Prepare data for training.
    """

    ti = kwargs["ti"]
    info = ti.xcom_pull(task_ids="get_data")
    info["data_preparation_start"] = datetime.now().strftime("%Y%m%d %H:%M")

    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(key="bootcamp/datasets/news.pkl", bucket_name=BUCKET)
    data = pd.read_pickle(file)

    # Сделать препроцессинг
    # Разделить на train и test
    train, test = train_test_split(data, test_size=0.2, stratify=data[TARGET])

    # Подготовить TFidf
    tfidfvectorizer = TfidfVectorizer(analyzer="word",
                                      lowercase=False,
                                      stop_words=list(
                                          stopwords.words("russian")),
                                      preprocessor=preprocess_text)

    # Обучить TFidf на train, применить к train и test
    X_train = tfidfvectorizer.fit_transform(train[FEATURES])
    X_test = tfidfvectorizer.transform(test[FEATURES])

    y_train = train[TARGET]
    y_test = test[TARGET]

    # Сохранить готовые данные на S3
    for name, data in zip(["X_train", "X_test", "y_train", "y_test"],
                          [X_train, X_test, y_train, y_test]):
        pickle_byte_obj = pickle.dumps(data)
        s3_hook.load_bytes(pickle_byte_obj, f"bootcamp/datasets/{name}.pkl", 
                          bucket_name=BUCKET, replace=True)

    _LOG.info("Data preparation finished.")
    info["data_preparation_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    return info


def train_model(**kwargs) -> Dict[str, Any]:
    """
    Step 3: Train logistic regression.
    """

    ti = kwargs["ti"]
    info = ti.xcom_pull(task_ids="prepare_data")
    m_name = kwargs["model_name"]

    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connection")
    # Загрузить готовые данные с S3
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(key=f"bootcamp/datasets/{name}.pkl",
                                     bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    # Обучить модель
    model = models[m_name]
    info[f"{m_name}_train_start"] = datetime.now().strftime("%Y%m%d %H:%M")
    model.fit(data["X_train"], data["y_train"])
    y_pred = model.predict(data["X_test"])
    y_test = data["y_test"]
    info[f"{m_name}_train_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    # Посчитать метрики
    info["metrics"] = classification_report(y_test, y_pred, output_dict=True)
    return info


def save_results(**kwargs) -> None:
    """
    Step 3: Save results to S3.
    """

    ti = kwargs["ti"]
    info = ti.xcom_pull(task_ids=["train_logreg", "train_nb"])
    
    result = {}
    for metric in info:
        result.update(metric)

    date = datetime.now().strftime("%Y_%m_%d_%H")
    s3_hook = S3Hook("s3_connection")
    json_byte_object = json.dumps(result).encode()
    s3_hook.load_bytes(json_byte_object, f"bootcamp/results/{date}.json", 
                          bucket_name=BUCKET, replace=True)


task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)

task_get_data = PythonOperator(task_id="get_data",
                               python_callable=get_data_from_postgres,
                               dag=dag,
                               provide_context=True)

task_prepare_data = PythonOperator(task_id="prepare_data",
                                   python_callable=prepare_data,
                                   dag=dag,
                                   provide_context=True)

task_train_models = [
    PythonOperator(task_id=f"train_{model_name}",
                   python_callable=train_model,
                   dag=dag,
                   provide_context=True,
                   op_kwargs={"model_name": model_name})
    for model_name in models.keys()
]

task_save_results = PythonOperator(task_id="save_results",
                                   python_callable=save_results,
                                   dag=dag,
                                   provide_context=True)

task_init >> task_get_data >> task_prepare_data >> task_train_models >> task_save_results