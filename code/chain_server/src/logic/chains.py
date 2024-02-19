"""LLM Chains for executing Retrival Augmented Generation."""
import base64
import os
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Generator, List, Dict, Any

from llama_index import (
    Prompt,
    download_loader
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import StreamingResponse, Response

if TYPE_CHECKING:
    from llama_index.types import TokenGen

from src.common.utils import (
    get_config,
    set_service_context,
    get_llm,
    get_doc_retriever,
    LimitRetrievedNodesLength,
    is_base64_encoded,
    get_vector_index,
    get_embedding_model,
    get_milvus_connection,
    get_milvus_collection
)

from src.common.base import BaseExample

# prestage the embedding model
_ = get_embedding_model()
set_service_context()

logger = logging.getLogger(__name__)

class Chain(BaseExample):

    def llm_chain(
        self, context: str, question: str, num_tokens: int, temperature: float
    ) -> Generator[str, None, None]:
        """Execute a simple LLM chain using the components defined above."""
        set_service_context()
        prompt = get_config().prompts.chat_template.format(
            context_str=context, query_str=question
        )
        response = get_llm().complete(prompt, max_tokens=num_tokens, temperature=temperature)

        for i in range(0, len(response.text), 20):
            yield response.text[i:i + 20]

    def llm_chain_streaming(
        self, context: str, question: str, num_tokens: int, temperature: float
    ) -> Generator[str, None, None]:
        """Execute a simple LLM chain using the components defined above."""
        set_service_context()
        prompt = get_config().prompts.chat_template.format(
            context_str=context, query_str=question
        )
        response = get_llm().stream_complete(prompt, max_tokens=num_tokens, temperature=temperature)
        gen_response = (resp.delta for resp in response)
        return gen_response

    def rag_chain(self, prompt: str, num_tokens: int, temperature) -> "TokenGen":
        """Execute a Retrieval Augmented Generation chain using the components defined above."""
        set_service_context()
        llm = get_llm()
        llm.llm.max_tokens = num_tokens  # type: ignore
        llm.llm.temperature = temperature  # type: ignore
        retriever = get_doc_retriever(num_nodes=4)
        qa_template = Prompt(get_config().prompts.rag_template)
        query_engine = RetrieverQueryEngine.from_args(
            retriever,
            text_qa_template=qa_template,
            node_postprocessors=[LimitRetrievedNodesLength()],
            streaming=False,
        )
        response = query_engine.query(prompt)

        # Properly handle an empty response
        if isinstance(response, Response):
            for i in range(0, len(response.response), 20):
                yield response.response[i:i + 20]
        return Response([]).response  # type: ignore

    def rag_chain_streaming(self, prompt: str, num_tokens: int, temperature: float) -> "TokenGen":
        """Execute a Retrieval Augmented Generation chain using the components defined above."""
        set_service_context()
        llm = get_llm()
        llm.llm.max_tokens = num_tokens  # type: ignore
        llm.llm.temperature = temperature  # type: ignore
        retriever = get_doc_retriever(num_nodes=4)
        qa_template = Prompt(get_config().prompts.rag_template)
        query_engine = RetrieverQueryEngine.from_args(
            retriever,
            text_qa_template=qa_template,
            node_postprocessors=[LimitRetrievedNodesLength()],
            streaming=True,
        )
        response = query_engine.query(prompt)

        # Properly handle an empty response
        if isinstance(response, StreamingResponse):
            return response.response_gen
        return StreamingResponse([]).response_gen  # type: ignore

    def ingest_docs(self, data_dir: str, filename: str) -> None:
        """Ingest documents to the VectorDB."""
        unstruct_reader = download_loader("UnstructuredReader")
        loader = unstruct_reader()
        documents = loader.load_data(file=Path(data_dir), split_documents=False)

        encoded_filename = filename[:-4]
        if not is_base64_encoded(encoded_filename):
            encoded_filename = base64.b64encode(encoded_filename.encode("utf-8")).decode(
                "utf-8"
            )

        for document in documents:
            document.metadata = {"filename": encoded_filename}

        index = get_vector_index()

        node_parser = SimpleNodeParser.from_defaults()
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        index.insert_nodes(nodes)

    def document_search(self, content: str, num_docs: int) -> List[Dict[str, Any]]:
        """Search for the most relevant documents for the given search parameters."""

        try:
            retriever = get_doc_retriever(num_nodes=num_docs)
            nodes = retriever.retrieve(content)
            output = []
            for node in nodes:
                file_name = nodes[0].metadata["filename"]
                decoded_filename = base64.b64decode(file_name.encode("utf-8")).decode("utf-8")
                entry = {"score": node.score, "source": decoded_filename, "content": node.text}
                output.append(entry)

            return output

        except Exception as e:
            logger.error(f"Error from /documentSearch endpoint. Error details: {e}")
            return []
    
    def document_database_reset(self):
        """Resets document database"""

        collection = get_milvus_collection()

        # Delete entities
        expr = "id!=''"
        collection.delete(expr)

        #collection.drop()