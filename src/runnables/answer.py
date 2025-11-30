import logging
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from pydantic import BaseModel, Field

from src.constants import ANSWER, QUESTION, RETRIEVED_MESSAGES
from src.prompts import answer_formulation_prompt, answer_formulation_template

logger = logging.getLogger(__name__)


class FormulatedAnswerOutput(BaseModel):
    """Output schema for the FormulateAnswer runnable."""

    answer: str = Field(description="The answer.")


class FormulateAnswer(Runnable):
    def __init__(self, llm_instance: BaseChatModel) -> None:
        """Initialize the FormulateAnswer runnable with a language model instance.

        Args:
            llm_instance (BaseChatModel): An instance of a language model to use for generating answers.
        """
        super().__init__()
        logger.info(f"Initializing Formulate Answer")
        self.llm_instance = llm_instance

    def invoke(
        self,
        input_: Input,  # noqa: A002
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Output:
        """Generate an answer based on the provided chunks of messages and question.

        Args:
            input_: The input containing context and question.
            config: Optional runnable config.
            **kwargs: Additional keyword arguments.

        Returns:
            Output: The generated answer.
        """
        retrieved_messages: list[Document] = input_.get(RETRIEVED_MESSAGES, [])
        question: str = input_.get(QUESTION, "")
        logger.info(f"Question: {question}")
        logger.info(f"Retrieved messages: {len(retrieved_messages)}")
        structured_llm = self.llm_instance.with_structured_output(schema=FormulatedAnswerOutput)
        # Flatten the list of messages into a single string
        messages_content: str = "\n".join([msg.page_content for msg in retrieved_messages])
        messages_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=answer_formulation_prompt),
                HumanMessagePromptTemplate.from_template(template=answer_formulation_template),
            ]
        )
        chain = messages_template | structured_llm
        response: FormulatedAnswerOutput = chain.invoke({QUESTION: question, RETRIEVED_MESSAGES: messages_content})
        logger.info(f"Generated answer: {response.answer}")
        return {ANSWER: response.answer, **input_}
