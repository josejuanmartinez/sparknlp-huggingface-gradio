import json

import sparknlp
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql import functions as F

from hf.HFNerDLModel import HFNerDLModel


def start_pyspark():
    """
    Starts a PySpark session, needed for Spark NLP.
    :return: SparkSession object
    """
    spark_session = sparknlp.start()

    print("Spark NLP version", sparknlp.version())
    print("Apache Spark version:", spark_session.version)

    return spark_session


def fit_pipeline(spark_session):
    """
    Returns a PySpark MLLib Pipeline, with Spark NLP basic components DocumentAssembler+Tokenizer+Embeddings) and
    an NER model, downloaded from Hugging Face Models Hub in a downstream fashion.

    :param spark_session: SparkSession, called with the result of start_pyspark
    :return: a fit PySpark Pipeline
    """

    # Spark NLP: Basic components
    documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    tokenizer = Tokenizer() \
        .setInputCols("document") \
        .setOutputCol("token")

    glove_embeddings = WordEmbeddingsModel.pretrained() \
        .setInputCols("document", "token") \
        .setOutputCol("embeddings")

    # Hugging Face: Here is where the Hugging Face downstream task is carried out
    nerdl_model = HFNerDLModel() \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner") \
        .fromPretrained("jjmcarrascosa/ner_ncbi_glove_100d_en", "./models")

    # A mixed SparkNLP+Hugging Face PySpark pipeline
    nlp_pipeline = Pipeline(stages=[
        documentAssembler,
        tokenizer,
        glove_embeddings,
        nerdl_model
    ])

    return nlp_pipeline.fit(spark_session.createDataFrame([['']]).toDF("text"))


def transform_pipeline(session, pipe, text):
    """
    Returns the result of applying a transform operation using the pipeline, to an input text.

    :param session: A Spark Session
    :param pipe: A fit pipeline with Spark NLP and a model from HF hub
    :param text: The input text to be used for Named Entity Recognition
    :return: a json with information about tokens and entities detected.
    """
    result = pipe.transform(session.createDataFrame([[text]]).toDF("text"))
    text_tokens = result.select(F.explode(F.arrays_zip('token.result', 'ner.result')).alias("cols")) \
        .select(F.expr("cols['0']").alias("token"),
                F.expr("cols['1']").alias("ner")).collect()

    # text_json = [{'token': x.token, 'ner': x.ner} for x in text_tokens]
    text_list = []
    for x in text_tokens:
        text_list.extend([(x.token, x.ner), (" ", None)])

    return text_list


if __name__ == '__main__':
    # 1. Start PySpark
    spark_session = start_pyspark()
    # 2. Create pipeline
    spark_pipeline = fit_pipeline(spark_session)
    # 3. Predict with input text
    spark_prediction = transform_pipeline(spark_session, spark_pipeline, "The patient was vomiting, had a poor appetite"
                                                                         " and was diagnosed a month ago with "
                                                                         "gestational diabetes mellitus")
    # 4. Return json with NER
    print(spark_prediction)
