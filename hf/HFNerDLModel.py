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
