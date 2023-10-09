import argparse
import json
import os
import zipfile
from typing import Optional

from gptcache import cache, Cache, Config
from gptcache.adapter import bigdl_llm_serving, openai
from gptcache.adapter.api import get, put
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.onnx import OnnxModelEvaluation
from gptcache.embedding import Onnx as EmbeddingOnnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.processor.pre import last_content
from gptcache.processor.pre import get_last_content_or_prompt
from gptcache.utils import import_fastapi, import_pydantic, import_starlette

import_fastapi()
import_pydantic()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
import uvicorn
from pydantic import BaseModel
import time

app = FastAPI()
openai_cache: Optional[Cache] = None

embedding_onnx = EmbeddingOnnx()
class WrapEvaluation(SearchDistanceEvaluation):
    def evaluation(self, src_dict, cache_dict, **kwargs):
        return super().evaluation(src_dict, cache_dict, **kwargs)

    def range(self):
        return super().range()

sqlite_file = "sqlite.db"
faiss_file = "faiss.index"
has_data = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)

cache_base = CacheBase("sqlite")
vector_base = VectorBase("faiss", dimension=embedding_onnx.dimension)
data_manager = get_data_manager(cache_base, vector_base, max_size=100000)
cache.init(
    pre_embedding_func=get_last_content_or_prompt,
    embedding_func=embedding_onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=WrapEvaluation(),
    config=Config(similarity_threshold=0.95),
)

os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
cache.set_bigdl_llm_serving()


@app.get("/")
async def hello():
    return "hello gptcache server"

class CacheData(BaseModel):
    prompt: str
    answer: Optional[str] = ""

@app.post("/put")
async def put_cache(cache_data: CacheData) -> str:
    put(cache_data.prompt, cache_data.answer)
    return "successfully update the cache"


@app.post("/get")
async def get_cache(cache_data: CacheData) -> CacheData:
    result = get(cache_data.prompt)
    return CacheData(prompt=cache_data.prompt, answer=result)


@app.post("/flush")
async def get_cache() -> str:
    cache.flush()
    return "successfully flush the cache"

@app.api_route(
    "/v1/chat/completions",
    methods=["POST", "OPTIONS"],
)
async def chat(request: Request):

    import_starlette()
    from starlette.responses import StreamingResponse, JSONResponse

    succ_count = 0
    fail_count = 0

    params = await request.json()

    print("messages:", params.get("messages"))
    try:
        start_time = time.perf_counter()
        completion = bigdl_llm_serving.ChatCompletion.create(
            **params
        )

        res_text = bigdl_llm_serving.get_message_from_openai_answer(completion)
        consume_time = time.perf_counter() - start_time
        print("chat time consuming: {:.3f}s".format(consume_time))
        print(res_text)
        res = res_text

        return JSONResponse(content=res)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"bigdl llm serving error: {e}")

@app.api_route(
    "/v1/completions",
    methods=["POST", "OPTIONS"],
)
async def completions(request: Request):
    import_starlette()
    from starlette.responses import StreamingResponse, JSONResponse

    params = await request.json()

    print("prompt:", params.get("prompt"))
    try:
        start_time = time.perf_counter()
        completion = bigdl_llm_serving.Completion.create(
            **params
        )
        consume_time = time.perf_counter() - start_time
        print("completions time consuming: {:.3f}s".format(consume_time))
        print(completion["choices"][0]["text"])
        res = completion["choices"][0]["text"]
        return JSONResponse(content=res)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"bigdl llm serving error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--host", default="localhost", help="the hostname to listen on"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=8000, help="the port to listen on"
    )

    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()