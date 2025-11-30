import os

import dotenv
from langchain_chroma import Chroma
from pydantic import SecretStr

from src.embeddings.nomic import NomicEmbeddings

dotenv.load_dotenv()

embeddings = NomicEmbeddings(
    base_url=os.getenv("BASE_EMBEDDING_URL"),
    api_key=SecretStr("abc"),
    model=os.getenv("EMBEDDING_MODEL"),
    tiktoken_enabled=False,
)

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

docs = vector_store.similarity_search_with_score(
    "W studiu jubilerskim KLENOTA zazwyczaj kupuję biżuterię.",
    k=10,
)
for doc, _ in docs:
    print(doc.page_content)
