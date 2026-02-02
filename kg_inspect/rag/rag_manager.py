# import os
# from kg_inspect.kg_inspect import KGInspect
# from lightrag.utils import EmbeddingFunc
# from kg_inspect.vlm.ollama import ollama_model_complete
# from lightrag.llm.ollama import ollama_embed
# from lightrag.kg.shared_storage import initialize_pipeline_status
# from lightrag.utils import priority_limit_async_func_call, logger



# async def initialize_rag() -> KGInspect:
#     """
#     Khởi tạo LightRAGCustom với:
#       - LLM: Ollama (model/host lấy từ ENV)
#       - Embedding: CLIPEmbedder (hỗ trợ async natively)
#       - Vector DB: FAISS (cosine similarity)
#       - KG: Neo4J (mặc định của LightRAG)
#     """
#     print("Initializing LightRAG instance...")

#     # --- 1. Tạo embedder ---
#     # clip_embedder = CLIPEmbedder()  # sẽ tự load model theo ENV hoặc mặc định
#     embedding_dim = int(os.getenv("EMBEDDING_DIM", "512"))
#     print(f"[initialize_rag] Embedding dimension: {embedding_dim}")

#     cosine_threshold_default = "0.25"

#     embedding_host = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")
#     print(f"[initialize_rag] Embedding host: {embedding_host}")
#     rag = KGInspect(
#         working_dir=os.getenv("WORKING_DIR", "./rag_storage"),
#         llm_model_func=ollama_model_complete,
#         llm_model_name=os.getenv("LLM_MODEL", "qwen2.5vl:7b"),
#         summary_max_tokens=8192,
#         llm_model_kwargs={
#             "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
#             "options": {"num_ctx": 8192},
#             "timeout": int(os.getenv("TIMEOUT", "300")),
#         },

#         embedding_func=EmbeddingFunc(
#             embedding_dim=embedding_dim,
#             max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
#             func=lambda texts: ollama_embed(
#                 texts,
#                 embed_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"),
#                 host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
#             ),
#         ),

#         graph_storage="Neo4JStorage",
#         vector_storage="FaissVectorDBStorage",
#         vector_db_storage_cls_kwargs={
#             "cosine_better_than_threshold": float(
#                 os.getenv("FAISS_COSINE_THRESHOLD", cosine_threshold_default)
#             ),
#         },
#     )

#     # --- 3. Khởi tạo các storage backend ---
#     await rag.initialize_storages()
#     await initialize_pipeline_status()

#     print("LightRAG instance initialized successfully.")
#     return rag



import os
from kg_inspect.kg_inspect import KGInspect
from lightrag.utils import EmbeddingFunc
from kg_inspect.vlm.openai import gpt_4o_complete  # VLM/LLM completion (OpenAI)
from lightrag.llm.ollama import ollama_embed            # Embedding (OpenAI)
from lightrag.kg.shared_storage import initialize_pipeline_status


async def initialize_rag() -> KGInspect:
    """
    Khởi tạo KGInspect với OpenAI cho:
      - LLM/VLM: gpt_4o_mini_complete
      - Embedding: openai_embed
      - Vector DB: FAISS
      - KG: Neo4J
    """
    print("Initializing LightRAG instance (OpenAI)...")
    
    # --- Embedding config ---
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "512"))
    print(f"[initialize_rag] Embedding dimension: {embedding_dim}")

    cosine_threshold_default = "0.25"

    embedding_host = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")
    print(f"[initialize_rag] Embedding host: {embedding_host}")

    # --- LLM/VLM config ---
    llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    summary_max_tokens = int(os.getenv("SUMMARY_MAX_TOKENS", "8192"))
    timeout = int(os.getenv("TIMEOUT", "300"))

    # Nếu bạn dùng base_url (OpenAI-compatible), set OPENAI_BASE_URL
    base_url = os.getenv("OPENAI_BASE_URL")  # optional

    rag = KGInspect(
        working_dir = os.getenv("WORKING_DIR", "./rag_storage"),

       
        llm_model_func=gpt_4o_complete,
        llm_model_name=llm_model,
        summary_max_tokens=summary_max_tokens,
        llm_model_kwargs={
            
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": base_url,  # None nếu không set
            "timeout": timeout,
        },

        
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            ),
        ),

        graph_storage="Neo4JStorage",
        vector_storage="FaissVectorDBStorage",
        vector_db_storage_cls_kwargs={
            "cosine_better_than_threshold": float(
                os.getenv("FAISS_COSINE_THRESHOLD", cosine_threshold_default)
            ),
        },
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    print("LightRAG instance initialized successfully (OpenAI).")
    return rag