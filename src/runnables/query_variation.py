import logging
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from pydantic import BaseModel, Field

from src.constants import QUERY_VARIATIONS, QUESTION
from src.prompts import query_variation_prompt

logger = logging.getLogger(__name__)


class QueryVariationOutput(BaseModel):
    query_variations: list[str] = Field(description="A list of query variations to generate.", default=[])


class QueryVariation(Runnable):

    def __init__(self, llm_instance: BaseChatModel) -> None:
        """Initialize the QueryVariation runnable with a language model instance.

        Args:
            llm_instance (BaseChatModel): An instance of a language model to use for generating query variations.
        """
        super().__init__()
        logger.info(f"Initializing Query Variation")
        self.llm_instance = llm_instance

    def invoke(self, input_: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Output:
        """Given a user query, generate variations for search query.

        Take the user query and extract their intended message.
        Imagine potential messages that could've been sent and generate their variations.

        Args:
            input_: The user query.
            config: Optional runnable config.
            **kwargs: Additional keyword arguments.

        Returns:
            Output: The generated query variations.
        """
        query: str = input_.get(QUESTION, "")
        logger.info(f"Query: {query}")
        structured_llm = self.llm_instance.with_structured_output(schema=QueryVariationOutput)
        messages = [SystemMessage(content=query_variation_prompt), HumanMessage(content=query)]
        response: QueryVariationOutput = structured_llm.invoke(messages)
        logger.info(f"Generated query variations: {response.query_variations}")
        return {QUERY_VARIATIONS: response.query_variations, **input_}

    async def ainvoke(
        self,
        input_: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Output:
        """Asynchronously generate query variations.

        Args:
            input_: The user query.
            config: Optional runnable config.
            **kwargs: Additional keyword arguments.

        Returns:
            Output: The generated query variations.
        """
        query: str = input_.get(QUESTION, "")
        logger.info(f"Query: {query}")
        structured_llm = self.llm_instance.with_structured_output(schema=QueryVariationOutput)
        messages: list[BaseMessage] = [SystemMessage(content=query_variation_prompt), HumanMessage(content=query)]
        response: QueryVariationOutput = await structured_llm.ainvoke(messages)
        logger.info(f"Generated query variations: {response.query_variations}")
        return {QUERY_VARIATIONS: response.query_variations, **input_}
