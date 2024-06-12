import os
from typing import Optional

import logging
import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from censor import Censor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


censor_handler = Censor()
app = FastAPI()


class QuestionRequest(BaseModel):
    query: str
    threshold_confidience: float


@app.post("/filt/", response_class=PlainTextResponse)
async def question_filtered(q_request: QuestionRequest):
    
    results = censor_handler.do_filt(
        query=q_request.query,
        threshold=q_request.threshold_confidience,
    )
    logging.info(results)

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
        port=8891,
        log_level="info"
    )
