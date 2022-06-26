import gradio as gr
from main import *

spark = None
pipeline = None

EXAMPLES = [
    "The patient was diagnosed a month ago with gestational diabetes mellitus",
    "He was vomiting the day before",
    "This is a sentence with no entities to be recognized"
    ]


def get_spark_session():
    """
        Starts a PySpark session, required for Spark NLP
    :return: Spark Session
    """
    return start_pyspark()


def get_fit_pipeline(session):
    """
        Creates, preloads and returns a Spark MLLib Pipeline with an NER model stored in Hugging Face Hub
    :param session: Spark Session
    :return: Spark MLLib pipeline
    """
    return fit_pipeline(session)


def set_example_1():
    return EXAMPLES[0]


def set_example_2():
    return EXAMPLES[1]


def set_example_3():
    return EXAMPLES[2]


def predict(input_text):
    """
        Calls to `transform` on a `fit pipeline` to carry out NER on medical texts

    :param input_text: Text to analyze
    :return: A list of tuples (token, entity)
    """
    global spark, pipeline

    try:
        return transform_pipeline(spark, pipeline, input_text)
    except Exception as e:
        print(e)
        spark = get_spark_session()
        pipeline = get_fit_pipeline(spark)
        return transform_pipeline(spark, pipeline, input_text)


demo = gr.Blocks()

with demo:
    with gr.Row():
        gr.Markdown(
        """
            # Spark NLP on Hugging Face: an interactive Gradio demo
            
            This Gradio demo showcases how to run Spark NLP Models, deployed in 
            [Hugging Face Hub]((https://huggingface.co/jjmcarrascosa/ner_ncbi_glove_100d_en)), inside a Spark MLLib
            pipeline, including [Spark NLP](https://www.johnsnowlabs.com/spark-nlp/) and carrying out a 
            [downstream](https://huggingface.co/docs/huggingface_hub/how-to-downstream) download from Hugging Face 
            for an NER model.
            
            The NER model detects the entity `DISEASE`, in a _BIO_ notation. It was trained using the `ncib` dataset, 
            available [here](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data)
            
            The model card cand be found [here](https://huggingface.co/jjmcarrascosa/ner_ncbi_glove_100d_en)
            
            It's been trained on a light word embeddings model (glove with 100 dimensions) to boost inference time (versus
            accuracy).  
            
            ## Inference time:
            - 1st time: ~40 sec
            - 2nd and further: ~6 sec
        """)

        gr.Image(value='./img/hf_sparknlp.png')

    with gr.Row():
        inp = gr.Textbox(placeholder="Write a sentence to analyze Diseases", value=EXAMPLES[1])
        out = gr.HighlightedText()

    btn = gr.Button("Predict", variant='primary')
    btn.click(fn=predict, inputs=inp, outputs=out)

    with gr.Row():
        predefined_btn_1 = gr.Button("Example 1", variant='secondary')
        predefined_btn_2 = gr.Button("Example 2", variant='secondary')
        predefined_btn_3 = gr.Button("Example 3", variant='secondary')

        predefined_btn_1.click(fn=set_example_1, inputs=None, outputs=inp)
        predefined_btn_2.click(fn=set_example_2, inputs=None, outputs=inp)
        predefined_btn_3.click(fn=set_example_3, inputs=None, outputs=inp)


demo.launch()
