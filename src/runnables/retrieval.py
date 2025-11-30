import logging
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field

from src.constants import QUERY_VARIATIONS, RETRIEVED_MESSAGES, SOURCE

logger = logging.getLogger(__name__)


class RetrievalOutput(BaseModel):
    """Output schema for the MessageRetrieval runnable."""

    retrieved_messages: list[Document] = Field(description="A list of retrieved messages.", default=[])


class MessageRetrieval(Runnable):

    def __init__(
        self, vector_store: VectorStore, embeddings: Embeddings, max_returned_search: int = 10, top_k_results: int = 20
    ) -> None:
        super().__init__()
        logger.info(f"Initializing Message Retrieval")
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.max_returned_search = max_returned_search
        self.top_k_results = top_k_results

    def _process_documents(self, all_retrieved_docs: list[tuple[Document, float]]) -> list[Document]:
        """Process and clean the retrieved documents.

        Args:
            all_retrieved_docs (list[Document]): The list of documents to process.

        Returns:
            list[Document]: The processed and cleaned documents.
        """
        # Deduplicate documents based on their source
        seen_sources = set()
        unique_retrieved_docs: list[tuple[Document, float]] = []
        for doc, score in all_retrieved_docs:
            if doc.metadata[SOURCE] not in seen_sources:
                seen_sources.add(doc.metadata[SOURCE])
                unique_retrieved_docs.append((doc, score))
        # Sort documents by score (ascending order)
        unique_retrieved_docs.sort(key=lambda x: x[1])
        # Limit to top 20 documents
        top_docs: list[tuple[Document, float]] = unique_retrieved_docs[: self.top_k_results]
        # Extract documents from tuples
        retrieved_messages: list[Document] = [doc for doc, _ in top_docs]
        return RetrievalOutput(retrieved_messages=retrieved_messages).retrieved_messages

    def invoke(
        self,
        input_: Input,  # noqa: A002
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Output:
        """Retrieve relevant messages from the vector store.

        Given list of query variations, retrieve relevant chunks from the vector store.

        Args:
            input_: The input containing the query and vector store.
            config: Optional runnable config.
            **kwargs: Additional keyword arguments.

        Returns:
            Output: The retrieved messages.
        """
        query_variations: list[str] = input_.get(QUERY_VARIATIONS, [])
        logger.info(f"Query variations: {query_variations}")
        all_retrieved_docs: list[tuple[Document, float]] = []
        for query in query_variations:
            docs: list[tuple[Document, float]] = self.vector_store.similarity_search_with_score(
                query=query, k=self.max_returned_search
            )
            all_retrieved_docs.extend(docs)

        logger.info(f"Retrieved documents: {len(all_retrieved_docs)}")
        logger.info(f"Retrieved documents: {(doc.metadata[SOURCE] for doc, _ in all_retrieved_docs)}")
        return {RETRIEVED_MESSAGES: self._process_documents(all_retrieved_docs), **input_}

    async def ainvoke(
        self,
        input_: Input,  # noqa: A002
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Output:
        """Asynchronously retrieve relevant messages from the vector store.

        Given list of query variations, retrieve relevant chunks from the vector store.

        Args:
            input_: The input containing the query and vector store.
            config: Optional runnable config.
            **kwargs: Additional keyword arguments.

        Returns:
            Output: The retrieved messages.
        """
        query_variations: list[str] = input_.get(QUERY_VARIATIONS, [])
        logger.info(f"Query variations: {query_variations}")
        all_retrieved_docs: list[tuple[Document, float]] = []
        for query in query_variations:
            docs: list[tuple[Document, float]] = await self.vector_store.asimilarity_search_with_score(
                query=query, k=self.max_returned_search
            )
            all_retrieved_docs.extend(docs)

        logger.info(f"Retrieved documents: {len(all_retrieved_docs)}")
        logger.info(f"Retrieved documents: {(doc.metadata[SOURCE] for doc, _ in all_retrieved_docs)}")
        return {RETRIEVED_MESSAGES: self._process_documents(all_retrieved_docs), **input_}
