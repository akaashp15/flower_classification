import gradio as gr
from flowers_model_run import flower_classification
from flowers_model_run import img_height
from flowers_model_run import img_width
   

demo = gr.Interface(fn = flower_classification, inputs= gr.Image(shape=(img_height, img_width)), outputs="text")

demo.launch()