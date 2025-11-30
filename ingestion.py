import os

import dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from transformers import AutoTokenizer

from src.embeddings.nomic import NomicEmbeddings
from src.constants import CHAT_ID
from src.loaders import MetaMessengerLoader
from src.runnables.utils import remove_image_tags

dotenv.load_dotenv()
MODEL_NAME = os.getenv("EMBEDDING_MODEL")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def length_function(text: str) -> int:
    tokens = tokenizer.encode(text)
    return len(tokens)


embeddings = NomicEmbeddings(
    base_url=os.getenv("BASE_EMBEDDING_URL"),
    api_key=SecretStr("abc"),
    model=os.getenv("EMBEDDING_MODEL"),
    tiktoken_enabled=False,
)

loader = MetaMessengerLoader(data_dir="mb", allowed_dirs=[""])
documents = loader.load()
print(f"Loaded {len(documents)} documents")
# Remove image tags from the documents
documents = [remove_image_tags(doc) for doc in documents]
print(f"Documents after removing image tags: {len(documents)}")
# Split the documents into chunks
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=length_function)
texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} chunks")
# Give the documents a unique source ID that will also contain the file name and chunk number
for i, text in enumerate(texts):
    text.metadata["source"] = f"{text.metadata.get(CHAT_ID, 'blank')}-{i}"
print("Creating vector store...")
Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db",
)
