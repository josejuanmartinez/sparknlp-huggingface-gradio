---
license: apache-2.0
language: 
  - en
tags:
- clinical
- medical
- healthcare
- ner
- spark-nlp
datasets:
- ncbi
---

## Introduction
![spark nlp](https://drive.google.com/uc?export=download&id=1oO3vynfGEmuIURIQPKU6edyZ-ifOJDkC)
This model is aimed to showcase how easily you can integrate Spark NLP, one of the most used open source libraries, including the most use for the healthcare domain - Spark NLP for Healthcare - with the Hugging Face Hub. 

Spark NLP has an Open Source version of the library, which is open for the community to contributions or forks. However, the great potential of the library lies on the usage of [Spark](https://spark.apache.org/docs/latest/api/python/) and [Spark MLLib](https://spark.apache.org/mllib/), which offer a unique way to create pipelines with propietary or custom annotators, and be run at scale in Spark Clusters. Custom annotators help, for example, to carry out custom operations, as retrieving models from Hugging Face Hub, as we are going to showase.

To do this, we have included a basic Spark NLP Pipeline, built on the top of [PySpark](https://spark.apache.org/docs/latest/api/python/) and [Spark MLLib](https://spark.apache.org/mllib/), having the following components:
- A [DocumentAssembler](https://nlp.johnsnowlabs.com/docs/en/annotators#documentassembler)
- A [Tokenizer](https://nlp.johnsnowlabs.com/docs/en/annotators#tokenizer)
- An English Glove Embeddings (100 dimensions), using Spark NLP [WordEmbeddings](https://nlp.johnsnowlabs.com/docs/en/annotators#wordembeddings) annotator

The main task we will resolve is Named Entity Recognition. To do that, an NER model, trained with Spark NLP, has been uploaded to this repo. In the `How to use` section we will explain how to integrate Spark NLP with a custom annotator which loads a pretrained model, published in Hugging Face.

## Dataset
The NER model was trained using `ncib` dataset, available [here](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data)

## Spark MLLib and Spark NLP: Pipelines
![spark mllib pipelines](https://miro.medium.com/max/1199/0*UmAn9E2Ak0duhb_b)
Spark NLP leverages Spark MLlib pipelines, which means every operation is carried out by an annotator, which enriches a dataset and sends it to the next annotator in a series of "stages". There are some annotators available for generic Data Science operations in [Spark MLLib](https://spark.apache.org/mllib/), but Spark NLP is the specific library to carry out Natural Language Processing tasks.

## Motivation 
![spark nlp for healthcare](https://www.johnsnowlabs.com/wp-content/uploads/2020/05/healthcare_banner.svg)
Spark NLP is a natively scalable, production-ready library, which runs on the top of Spark, leveraging all the parallelization which Spark clusters bring. This means that, any pipeline supported by Spark NLP, can be run at any scale, including millions of documents, just adjusting your cluster configurations in an almost transparent way.

Also, the creators of Spark NLP, [John Snow Labs](https://johnsnowlabs.com/) are the creators of Spark NLP For Healthcare, the most used NLP library in the Healthcare domain, achieving SOTA results in tasks as Named entity Recognition, Assertion Status, Relation Extraction, Entity Linking, Data Augmentation, etc. This library is empowered with both in-house Deep Learning architectures and the latest transformers, to provide with a variety of ways to train your own models, fine tune or import from other libraries.

## How to use
Spark NLP has its own repository, available [here](https://nlp.johnsnowlabs.com/models), used internally to load pretrained models using the `pretrained` method. However, as mentioned before, Spark NLP works with Spark MLLib pipelines, that provides a class interface to create your own annotators.

Creating an annotator to load a model from Hugging Face Hub is as simple as this:

```python
from pyspark.ml.param.shared import (
    HasOutputCol,
    HasInputCols
)
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from sparknlp.annotator import NerDLModel
from sparknlp.base import *

from sparknlp.common import AnnotatorProperties

from huggingface_hub import snapshot_download


class HFNerDLModel(
    Transformer, AnnotatorProperties,
    DefaultParamsReadable, DefaultParamsWritable
):
    @keyword_only
    def __init__(self):
        super(HFNerDLModel, self).__init__()
        self.model = None
        self.dir = None

    # camelCase is not pythonic, but SparkNLP uses SCALA naming conventions since it's a Scala library
    # so I keep using camelCase
    def setInputCols(self, value):
        """
            Spark NLP works reading from columns in a dataframe and putting the output in a new column.
        This function sets the name of the columns this annotator will be retrieving information from.

        :param value: array of strings with the name of the columns to be used as input
        :return: void
        """
        return self._set(inputCols=value)

    def setOutputCol(self, value):
        """
            Spark NLP works reading from columns in a dataframe and putting the output in a new column.
        This function sets the name of the column this annotator will be storing to...

        :param value: Name of the output column
        :return: void
        """
        return self._set(outputCol=value)

    def fromPretrained(self, model, cache_dir):
        """
            Function that implements the loading from Hugging Face of a pretrained save model.
        Uses https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub

        :param model: Name of the model in HF
        :param cache_dir: name of the folder where the model will be downloaded
        :return: void
        """
        self.dir = snapshot_download(repo_id=model, cache_dir=cache_dir)

        self.model = NerDLModel().load(self.dir)\
            .setInputCols(self.getInputCols())\
            .setOutputCol(self.getOutputCol())

        return self

    def _transform(self, dataset):
        """
            Called from PySpark when applying `transform` to a pipeline.

        :param dataset: Dataset received from previous components in the PySpark pipeline
        :return: another dataset, now enriched, with NER information
        """
        return self.model.transform(dataset)
```

With that, we have a class that implements a custom Annotator in a Spark MLLIb pipeline. We can add it to a pipeline, all along with the rest of the Spark NLP models required by this custom NER annotator:

```python
    # Spark NLP: Basic components
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    dl_tokenizer = Tokenizer() \
        .setInputCols("document") \
        .setOutputCol("token")

    glove_embeddings = WordEmbeddingsModel.pretrained() \
        .setInputCols("document", "token") \
        .setOutputCol("embeddings")

    # Hugging Face: Here is where the Hugging Face downstream task is carried out
    ner_model = HFNerDLModel() \
        .setInputCols(("document", "token", "embeddings")) \
        .setOutputCol("ner") \
        .fromPretrained("jjmcarrascosa/ner_ncbi_glove_100d_en", "./models")

    # A mixed SparkNLP+Hugging Face PySpark pipeline
    nlp_pipeline = Pipeline(stages=[
        document_assembler,
        dl_tokenizer,
        glove_embeddings,
        ner_model
    ])
```

And just calling fit() and transform() on a dataset, you are done!

```python
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
```

## Gradio demo
A Gradio demo has been created so that you check all the previously described information. Check it [here]() 

## Training process
We have used Spark NLP NerDLModel `Approach` (a specific Annotator in Spark NLP API to train models), to train the NER model on the ncib dataset. The code for training and the hyperparameters used are the following:

```python
nerTagger = NerDLApproach()\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setLabelColumn("label")\
    .setOutputCol("ner")\
    .setMaxEpochs(15)\
    .setLr(0.003)\
    .setDropout(0.3)\
    .setBatchSize(128)\
    .setRandomSeed(42)
```


## Evaluation results
These are the metrics provided by the library after the last stage of the training process:
```
              precision    recall  f1-score   support

   B-Disease       0.86      0.85      0.85       960
   I-Disease       0.80      0.89      0.84      1087

    accuracy                           0.85      2047
   macro avg       0.83      0.87      0.84      2047
weighted avg       0.84      0.88      0.85      2047
```
