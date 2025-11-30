from typing import Any, Iterable, Optional, cast
from langchain_openai import OpenAIEmbeddings


class NomicEmbeddings(OpenAIEmbeddings):
    """Wrapper that prepares inputs for Nomic-style search and uses OpenAI
    Embeddings API to retrieve vectors.

    Documents are prefixed with "search_document: " and queries with
    "search_query: " as required by the nomic-embed-text-v2-moe style usage.
    """

    def _prefixed(self, texts: Iterable[str], prefix: str) -> list[str]:
        return [f"{prefix}{t}" for t in texts]

    def embed_documents(
        self, texts: list[str], chunk_size: Optional[int] = None, **kwargs: Any
    ) -> list[list[float]]:
        """Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.
            kwargs: Additional keyword arguments to pass to the embedding API.

        Returns:
            list of embeddings, one for each text.
        """
        chunk_size_ = chunk_size or self.chunk_size
        client_kwargs = {**self._invocation_params, **kwargs}
        # Prefix documents for nomic-style embeddings
        prefixed_texts = self._prefixed(texts, "search_document: ")
        if not self.check_embedding_ctx_length:
            embeddings: list[list[float]] = []
            for i in range(0, len(prefixed_texts), chunk_size_):
                response = self.client.create(
                    input=prefixed_texts[i : i + chunk_size_], **client_kwargs
                )
                if not isinstance(response, dict):
                    response = response.model_dump()
                embeddings.extend(r["embedding"] for r in response["data"])
            return embeddings

        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        engine = cast(str, self.deployment)
        return self._get_len_safe_embeddings(
            prefixed_texts, engine=engine, chunk_size=chunk_size_, **kwargs
        )

    async def aembed_documents(
        self, texts: list[str], chunk_size: Optional[int] = None, **kwargs: Any
    ) -> list[list[float]]:
        """Call out to OpenAI's embedding endpoint async for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.
            kwargs: Additional keyword arguments to pass to the embedding API.

        Returns:
            list of embeddings, one for each text.
        """
        chunk_size_ = chunk_size or self.chunk_size
        client_kwargs = {**self._invocation_params, **kwargs}
        # Prefix documents for nomic-style embeddings
        prefixed_texts = self._prefixed(texts, "search_document: ")
        if not self.check_embedding_ctx_length:
            embeddings: list[list[float]] = []
            for i in range(0, len(prefixed_texts), chunk_size_):
                response = await self.async_client.create(
                    input=prefixed_texts[i : i + chunk_size_], **client_kwargs
                )
                if not isinstance(response, dict):
                    response = response.model_dump()
                embeddings.extend(r["embedding"] for r in response["data"])
            return embeddings

        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        engine = cast(str, self.deployment)
        return await self._aget_len_safe_embeddings(
            prefixed_texts, engine=engine, chunk_size=chunk_size_, **kwargs
        )

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.
            kwargs: Additional keyword arguments to pass to the embedding API.

        Returns:
            Embedding for the text.
        """
        prefixed = self._prefixed([text], "search_query: ")
        return self.embed_documents(prefixed, **kwargs)[0]

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Call out to OpenAI's embedding endpoint async for embedding query text.

        Args:
            text: The text to embed.
            kwargs: Additional keyword arguments to pass to the embedding API.

        Returns:
            Embedding for the text.
        """
        prefixed = self._prefixed([text], "search_query: ")
        embeddings = await self.aembed_documents(prefixed, **kwargs)
        return embeddings[0]
