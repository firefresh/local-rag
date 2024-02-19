"""The definition of the application configuration."""
from src.common.configuration_wizard import ConfigWizard, configclass, configfield


@configclass
class VectorStoreConfig(ConfigWizard):
    """Configuration class for the Vector Store connection.

    :cvar name: Name of vector store
    :cvar url: URL of Vector Store
    """

    name: str = configfield(
        "name",
        default="milvus", # supports pgvector, milvus
        help_txt="The name of vector store",
    )
    collection_name: str = configfield(
        "collection_name",
        default="documents_data", # supports pgvector, milvus
        help_txt="The collection name of vector store",
    )
    collection_dimentions: str = configfield(
        "collection_dimentions",
        default="1024", # supports pgvector, milvus
        help_txt="The collection dimentions of vector store",
    )
    url: str = configfield(
        "url",
        default="http://milvus:19530", # for pgvector `pgvector:5432`
        help_txt="The host of the machine running Vector Store DB",
    )
    nlist: int = configfield(
        "nlist",
        default=64, # IVF Flat milvus
        help_txt="Number of cluster units",
    )
    nprobe: int = configfield(
        "nprobe",
        default=16, # IVF Flat milvus
        help_txt="Number of units to query",
    )


@configclass
class LLMConfig(ConfigWizard):
    """Configuration class for the llm connection.

    :cvar server_url: The location of the llm server hosting the model.
    :cvar model_name: The name of the hosted model.
    """

    server_url: str = configfield(
        "server_url",
        default="localhost:8001",
        help_txt="The location of the Triton server hosting the llm model.",
    )
    model_name: str = configfield(
        "model_name",
        default="ensemble",
        help_txt="The name of the hosted model.",
    )
    model_engine: str = configfield(
        "model_engine",
        default="openai",
        help_txt="The server type of the hosted model. Allowed values are openai",
    )
    model_temperature: float = configfield(
        "model_temperature",
        default=1.0,
        help_txt="The temperature of the hosted model.",
    )


@configclass
class TextSplitterConfig(ConfigWizard):
    """Configuration class for the Text Splitter.

    :cvar chunk_size: Chunk size for text splitter. Tokens per chunk in token-based splitters.
    :cvar chunk_overlap: Text overlap in text splitter.
    """

    chunk_size: int = configfield(
        "chunk_size",
        default=510,
        help_txt="Chunk size for text splitting.",
    )
    chunk_overlap: int = configfield(
        "chunk_overlap",
        default=200,
        help_txt="Overlapping text length for splitting.",
    )


@configclass
class EmbeddingConfig(ConfigWizard):
    """Configuration class for the Embeddings.

    :cvar model_name: The name of the huggingface embedding model.
    """

    model_name: str = configfield(
        "model_name",
        default="intfloat/e5-large-v2",
        help_txt="The name of huggingface embedding model.",
    )
    model_engine: str = configfield(
        "model_engine",
        default="huggingface",
        help_txt="The server type of the hosted model. Allowed values are hugginface",
    )
    engine_divice: str = configfield(
        "engine_divice",
        default="cpu",
        help_txt="The divice on the host where the embedding model will run.",
    )
    dimensions: int = configfield(
        "dimensions",
        default=1024,
        help_txt="The required dimensions of the embedding model. Currently utilized for vector DB indexing.",
    )


@configclass
class PromptsConfig(ConfigWizard):
    """Configuration class for the Prompts.

    :cvar chat_template: Prompt template for chat.
    :cvar rag_template: Prompt template for rag.
    """

    chat_template: str = configfield(
        "chat_template",
        default=(
            "<s>[INST] <<SYS>>"
            "You are a helpful, respectful and honest assistant."
            "Always answer as helpfully as possible, while being safe."
            "Please ensure that your responses are positive in nature."
            "<</SYS>>"
            "[/INST] {context_str} </s><s>[INST] {query_str} [/INST]"
        ),
        help_txt="Prompt template for chat.",
    )
    rag_template: str = configfield(
        "rag_template",
        default=(
            "<s>[INST] <<SYS>>"
            "Use the following context to answer the user's question. If you don't know the answer,"
            "just say that you don't know, don't try to make up an answer."
            "<</SYS>>"
            "<s>[INST] Context: {context_str} Question: {query_str} Only return the helpful"
            " answer below and nothing else. Helpful answer:[/INST]"
        ),
        help_txt="Prompt template for rag.",
    )


@configclass
class AppConfig(ConfigWizard):
    """Configuration class for the application.

    :cvar vector_store: The configuration of the vector db connection.
    :type vector_store: VectorStoreConfig
    :cvar llm: The configuration of the backend llm server.
    :type llm: LLMConfig
    :cvar text_splitter: The configuration for text splitter
    :type text_splitter: TextSplitterConfig
    :cvar embeddings: The configuration for huggingface embeddings
    :type embeddings: EmbeddingConfig
    :cvar prompts: The Prompts template for RAG and Chat
    :type prompts: PromptsConfig
    """

    vector_store: VectorStoreConfig = configfield(
        "vector_store",
        env=False,
        help_txt="The configuration of the vector db connection.",
        default=VectorStoreConfig(),
    )
    llm: LLMConfig = configfield(
        "llm",
        env=False,
        help_txt="The configuration for the server hosting the Large Language Models.",
        default=LLMConfig(),
    )
    text_splitter: TextSplitterConfig = configfield(
        "text_splitter",
        env=False,
        help_txt="The configuration for text splitter.",
        default=TextSplitterConfig(),
    )
    embeddings: EmbeddingConfig = configfield(
        "embeddings",
        env=False,
        help_txt="The configuration of embedding model.",
        default=EmbeddingConfig(),
    )
    prompts: PromptsConfig = configfield(
        "prompts",
        env=False,
        help_txt="Prompt templates for chat and rag.",
        default=PromptsConfig(),
    )