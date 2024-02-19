"""The definition of the Llama Index chain server."""
import base64
import os
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List
import tempfile
import importlib
from inspect import getmembers, isclass

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pymilvus.exceptions import MilvusException, MilvusUnavailableException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create the FastAPI server
app = FastAPI()
# prestage the embedding model
# _ = app.chains.get_embedding_model()
# # set the global service context for Llama Index
# app.chains.set_service_context()

CHAINS_DIR = "src/logic/"
CHAIN_MODE = "chains"

class Prompt(BaseModel):
    """Definition of the Prompt API data type."""

    question: str = Field(description="The input query/prompt to the pipeline.")
    context: str = Field(description="Additional context for the question (optional)")
    use_knowledge_base: bool = Field(description="Whether to use a knowledge base", default=True)
    num_tokens: int = Field(description="The maximum number of tokens in the response.", default=50)
    temperature: float = Field(description="The temperature number of inference.", default=1.0)


class DocumentSearch(BaseModel):
    """Definition of the DocumentSearch API data type."""

    content: str = Field(description="The content or keywords to search for within documents.")
    num_docs: int = Field(description="The maximum number of documents to return in the response.", default=4)


@app.on_event("startup")
def import_example() -> None:
    """
    Import the example class from the specified example file.
    The example directory is expected to have a python file where the example class is defined.
    """

    for root, dirs, files in os.walk(CHAINS_DIR):
        for file in files:
            if not file.endswith(".py"):
                continue

            # Import the specified file dynamically
            spec = importlib.util.spec_from_file_location(name=CHAIN_MODE, location=os.path.join(root, file))
            logger.info(f"spec: {spec}")
            module = importlib.util.module_from_spec(spec)
            logger.info(f"spec: {module}")
            spec.loader.exec_module(module)
            logger.info(f"spec: {module}")
            # Scan each class in the file to find one with the 3 implemented methods: ingest_docs, rag_chain and llm_chain
            for name, _ in getmembers(module, isclass):
                try:
                    logger.info(f"class name: {name}")
                    cls = getattr(module, name)
                    logger.info(f"cls: {cls}")
                    if set(["ingest_docs", "llm_chain", "rag_chain"]).issubset(set(dir(cls))):
                        if name == "BaseExample":
                            continue
                        logger.info("loading chain...")
                        chains = cls()
                        app.chains = cls
                        return
                except Exception as ex:
                    logger.exception(ex)
                    raise ValueError(f"Class {name} is not implemented and could not be instantiated.")

    raise NotImplementedError(f"Could not find a valid chain class in {CHAINS_DIR}")


@app.get("/health")
async def health() -> JSONResponse:
      return JSONResponse(
        content={"status": "OK"}, status_code=200
    )


@app.post("/uploadDocument")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a document to the vector store."""
    if not file.filename:
        return JSONResponse(content={"message": "No files provided"}, status_code=200)

    try:
        chain = app.chains()
        upload_folder = tempfile.mkdtemp()
        upload_file = os.path.basename(file.filename)
        if not upload_file:
            raise RuntimeError("Error parsing uploaded filename.")
        file_path = os.path.join(upload_folder, upload_file)
        uploads_dir = Path(upload_folder)
        uploads_dir.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        chain.ingest_docs(file_path, upload_file)

        return JSONResponse(
            content={"message": "File uploaded successfully"}, status_code=200
        )

    except Exception as e:
        logger.error("Error from /uploadDocument endpoint. Ingestion of file: " + file.filename + " failed with error: " + str(e))
        return JSONResponse(
            content={"message": str(e)}, status_code=500
        )


@app.post("/generate")
async def generate_answer(prompt: Prompt) -> StreamingResponse:
    """Generate and stream the response to the provided prompt."""

    try:
        chain = app.chains()
        if prompt.use_knowledge_base:
            logger.info("Knowledge base is enabled. Using rag chain for response generation.")
            generator = chain.rag_chain_streaming(prompt.question, prompt.num_tokens, prompt.temperature)
            return StreamingResponse(generator, media_type="text/event-stream")

        generator = chain.llm_chain_streaming(prompt.context, prompt.question, prompt.num_tokens, prompt.temperature)
        return StreamingResponse(generator, media_type="text/event-stream")

    except (MilvusException, MilvusUnavailableException) as e:
        logger.error(f"Error from Milvus database in /generate endpoint. Please ensure you have ingested some documents. Error details: {e}")
        return StreamingResponse(iter(["Error from milvus server. Please ensure you have ingested some documents. Please check chain-server logs for more details."]), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error from /generate endpoint. Error details: {e}")
        return StreamingResponse(iter(["Error from chain server. Please check chain-server logs for more details."]), media_type="text/event-stream")


@app.post("/documentSearch")
async def document_search(data: DocumentSearch) -> List[Dict[str, Any]]:
    """Search for the most relevant documents for the given search parameters."""

    try:
        chain = app.chains()
        if hasattr(chain, "document_search") and callable(chain.document_search):
            return chain.document_search(data.content, data.num_docs)

        raise NotImplementedError("Example class has not implemented the document_search method.")

    except Exception as e:
        logger.error(f"Error from /documentSearch endpoint. Error details: {e}")
        return []

@app.get("/documentDatabaseReset")
async def reset_document_database() -> JSONResponse:
    """Reset vector document database"""
    try:
        chain = app.chains()
        if hasattr(chain, "document_database_reset") and callable(chain.document_database_reset):
            chain.document_database_reset()
            return JSONResponse(
                content={"message": "Document database reset complete successfully"}, status_code=200
            )
        raise NotImplementedError("Example class has not implemented the document_search method.")
    except (MilvusException, MilvusUnavailableException) as e:
        logger.error(f"Error from Milvus database in /documentDatabaseReset endpoint. Error details: {e}")
        return JSONResponse(
            content={"message": f"Error from Milvus database in /documentDatabaseReset endpoint. Error details: {e}"}, status_code=500
        )
    except Exception as e:
        logger.error("Error from /documentDatabaseReset endpoint. Failed with error: " + str(e))
        return JSONResponse(
            content={"message": str(e)}, status_code=500
        )