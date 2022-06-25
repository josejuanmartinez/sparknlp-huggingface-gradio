import gradio as gr
from main import *

title = "Interactive demo: Spark NLP + Hugging Face Hub on Gradio"
description = "This demo aims to showcase how easily you can create Spark NLP pipelines consuming models from Hugging" \
              " Face Hub."


def predict(input_text):
    # 1. Start PySpark
    spark_session = start_pyspark()
    # 2. Create pipeline
    spark_pipeline = fit_pipeline(spark_session)
    # 3. Predict with input text
    spark_model = LightPipeline(spark_pipeline)
    return spark_model.annotate(input_text)


iface = gr.Interface(fn=predict,
                     inputs=[gr.inputs.Textbox(label="input")],
                     outputs='text',
                     title=title,
                     description=description,
                     examples=["The patient was vomiting the night before, had a poor appetite",
                               "The patient was diagnosed a month ago with gestational diabetes "
                               "mellitus"],
                     enable_queue=True)

iface.launch()
