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


GRADIO_APP_PORT = int(os.getenv('GRADIO_APP_PORT'))
RAG_APP_PORT = int(os.getenv('RAG_APP_PORT'))
CENSOR_APP_PORT = int(os.getenv('CENSOR_APP_PORT'))
RAG_URL = f"http://0.0.0.0:{RAG_APP_PORT}/get_answer"
CENSOR_URL = f"http://0.0.0.0:{CENSOR_APP_PORT}/filt"
logging.info(f"\tGRADIO_APP_PORT: {GRADIO_APP_PORT};\tRAG_URL: {RAG_URL};\tCENSOR_URL: {CENSOR_URL}")

theme = gr.themes.Base(
    primary_hue="green",
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

        logging.info(f"Response: {response['answer']}")

        return response['answer']


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
    server_name="0.0.0.0",
    server_port=GRADIO_APP_PORT,
    share=True,
    show_error=True,
)