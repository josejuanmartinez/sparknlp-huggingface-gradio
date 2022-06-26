import gradio as gr
from main import *

title = "Interactive demo: Spark NLP + Hugging Face Hub on Gradio"
description = "This demo aims to showcase how easily you can create Spark NLP pipelines consuming models from Hugging" \
              " Face Hub."


def predict(input_text):
    # 1. Start PySpark
    spark = start_pyspark()
    # 2. Create pipeline
    pipeline = fit_pipeline(spark)
    # 3. Predict with input text
    prediction = transform_pipeline(spark, pipeline, input_text)
    # 4. Return json with NER
    return prediction


iface = gr.Interface(fn=predict,
                     inputs=[gr.inputs.Textbox(label="input")],
                     outputs='highlight',
                     title=title,
                     description=description,
                     examples=["The patient was vomiting",
                               "He had a bad appetite",
                               "She was diagnosed a month ago with gestational diabetes mellitus"],
                     enable_queue=True)

iface.launch()
