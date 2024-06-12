import os
import requests
import logging
import pickle
import gradio as gr
from dotenv import load_dotenv


load_dotenv()
logging.root.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

APP_HOST_IP = "0.0.0.0"
APP_PORT = 8888 #int(os.getenv('APP_PORT'))
RAG_URL = f"http://{APP_HOST_IP}:8892/get_answer"
# CENSOR_URL = f"http://{APP_HOST_IP}:8891/filt"
logging.info(f"APP_HOST_IP: {APP_HOST_IP};\tAPP_PORT: {APP_PORT};\tRAG_URL: {RAG_URL}")

theme = gr.themes.Base(
    primary_hue="green",
    # secondary_hue="white",
)


with gr.Blocks(title="Retrieval-Augmented Generation with Censor-Filter", theme=theme) as demo:


    def ask_question(question, confidience):
        if len(question.strip()) == 0:
            gr.Warning('Строка не должна быть пустая!')
            return None

        logging.info(f"Inputs: {question}; {confidience}.")
        
        package = {
            "query": question,
            "threshold_confidience": confidience,
        }

        response = requests.post(RAG_URL, json=package)

        if response.status_code != 200:
            gr.Warning(f'Response status: {response.status_code}')
            return None, None

        logging.info(f"Response text: {response.text}")

        return response.text


    gr.Markdown(
                '<p style="font-size: 2.5em; text-align: center; margin-bottom: 1rem"><span style="font-family:Source Sans Pro; color:black"> Retrieval-Augmented Generation with Censor-Filter </span></p>'
            )

    with gr.Row():

        with gr.Column():
            msg_txt = gr.Textbox(
                label="Задайте вопрос:", 
                placeholder="какие цветы лучше всего растут зимой?",
                interactive=True,
                show_copy_button=True,
            )
            ask_quest_btn = gr.Button(
                value='Спросить', 
                variant="primary"
            )
            confidience_slider = gr.Slider(
                label="Порог Цензора, значение ниже которого определяется как \"атака\"",
                minimum=0.0,
                maximum=1.0,
                value=0.2,
                step=0.01,
                interactive=True,
            )

        with gr.Column():
            answer_text_area = gr.TextArea(
                label="Ответ:",
                interactive=False,
                lines=30,
                show_copy_button=True,
            )

        msg_txt.submit(ask_question, [msg_txt, confidience_slider], [answer_text_area])
        ask_quest_btn.click(ask_question, [msg_txt, confidience_slider], [answer_text_area])

demo.launch(
    server_name=APP_HOST_IP,
    server_port=APP_PORT,
    share=True,
    show_error=True,
)