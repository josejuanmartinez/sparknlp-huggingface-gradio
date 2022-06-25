import sparknlp
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from sparknlp.base import LightPipeline

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

def fit_pipeline(spark):
    """
    Returns a PySpark MLLib Pipeline, with Spark NLP basic components DocumentAssembler+Tokenizer+Embeddings) and
    an NER model, downloaded from Hugging Face Models Hub in a downstream fashion.

    :param spark: SparkSession, called with the result of start_pyspark
    :return: a fit PySpark Pipeline
    """

    # Spark NLP: Basic components
    documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    tokenizer = Tokenizer() \
        .setInputCols("document") \
        .setOutputCol("token")

    glove_embeddings = WordEmbeddingsModel.pretrained()\
        .setInputCols("sentence", "token")\
        .setOutputCol("embeddings")

    # Hugging Face: Here is where the Hugging Face downstream task is carried out
    nerdl_model = HFNerDLModel().fromPretrained("ner_ncbi_glove_100d", "./models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

    # A mixed SparkNLP+Hugging Face PySpark pipeline
    nlp_pipeline = Pipeline(stages=[
        documentAssembler,
        tokenizer,
        glove_embeddings,
        nerdl_model
    ])

    return nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text"))


def transform_pipeline(pipe, text):
    """
    Returns the result of applying a transform operation using the pipeline, to an input text.
    We will use LightPipeline to serialize and speed up the inference, since we are not able to leverage Spark NLP
    parallel capabilities in a single-node machine.

    :param pipe: A fit pipeline with Spark NLP and a model from HF hub
    :param text: The input text to be used for Named Entity Recognition
    :return: a json with information about tokens and entities detected.
    """
    lp_model = LightPipeline(pipe)
    return lp_model.annotate(text)


if __name__ == '__main__':
    # 1. Start PySpark
    spark = start_pyspark()
    # 2. Create pipeline
    pipeline = fit_pipeline(spark)
    # 3. Predict with input text
    prediction = transform_pipeline(pipeline, "The patient was vomiting the night before, had a poor appetite and"
                                              "explains she was diagnosed a month ago with gestational diabetes "
                                              "mellitus")
    # 4. Return json with NER
    print(prediction)
    model = LightPipeline(pipeline)
