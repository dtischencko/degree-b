import os
from typing import Optional

import logging
import asyncio
import uvicorn
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

# from rag import RAGHandler


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


# rag_handler = RAGHandler()
app = FastAPI()


class QuestionRequest(BaseModel):
    query: str
    threshold_confidience: float


@app.post("/get_answer/", response_class=PlainTextResponse)
async def retrieve_augmented_generate(q_request: QuestionRequest):
    
    # filt message
    response = requests.post(
        url="http://0.0.0.0:8891/filt/",  # fixme
        json={
            'query': q_request.query,
            'threshold_confidience': q_request.threshold_confidience,
        }
    )

    results = None

    if response.status_code != 200:
        logging.warning(f"Response code from CENSOR is {response.status_code}!")

    elif len(response.text) == 0:
        logging.info("RAG blocked!")

    else:
        # rag
        logging.info("RAG started!")
        results = response.text
        # results = rag_handler.do_rag(
        #     query=q_request.query,
        #     threshold_confidience=0.8,
        # )

    return results


@app.get("/hello", response_class=HTMLResponse)
async def hello_world():
    html_content = """
    <html>
        <head>
            <title>Some HTML in here</title>
        </head>
        <body>
            <h1>Hello, World!</h1>
        </body>
    </html>
    """
    return html_content


if __name__ == '__main__':
    uvicorn.run(
        'app:app',
        host="0.0.0.0",
        port=8892,
        log_level="info"
    )
