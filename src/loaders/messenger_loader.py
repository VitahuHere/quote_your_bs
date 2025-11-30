import json
import logging
from pathlib import Path
from typing import Iterator

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from src.constants import (
    ARCHIVED_THREADS,
    CONTENT,
    E2EE_CUTOVER,
    FILTERED_THREADS,
    INBOX,
    MESSAGES,
    PHOTOS,
    SENDER_NAME,
    TIMESTAMP,
    URI,
)
from src.loaders.utils import convert_timestamp_to_datetime, extract_conversation_meta, get_decoded_content

logger = logging.getLogger(__name__)

MESSAGE_TEMPLATE = """{sender_name} ({timestamp}): {content}"""
PHOTO_TEMPLATE = """{sender_name} ({timestamp}): ![]({photo_path})"""


class MetaMessengerLoader(BaseLoader):
    """Loader for Meta Messenger messages."""

    def __init__(
        self, data_dir: str | Path = "your_facebook_activity/messages", allowed_dirs: list[str] = None
    ) -> None:
        """
        Initialize the loader with a directory containing conversations.

        Args:
            data_dir (str | Path): Directory where conversations are stored.
            allowed_dirs (list[str], optional): List of allowed subdirectories to search for JSON files.
            Defaults to ["inbox", "archived_threads", "e2ee_cutover", "filtered_threads"].
        """
        if allowed_dirs is None:
            allowed_dirs: list[str] = [
                INBOX,
                ARCHIVED_THREADS,
                E2EE_CUTOVER,
                FILTERED_THREADS,
            ]
        self.data_dir: Path = Path(data_dir)
        self.allowed_dirs: list[str] = allowed_dirs
        logger.info(f"Meta Messenger data directory: {self.data_dir}")

    def _parse_file(self, file_path: Path) -> Document:
        """Parse a single JSON file and extract messages as Document objects.

        Args:
            file_path (Path): Path to the JSON file containing conversation data.

        Returns:
            Document: A Document object containing the parsed messages.
        """
        logger.info(f"Parsing file {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            conversation_data: dict = json.load(f)

        messages: list[dict] = conversation_data.get(MESSAGES, [])

        document_text: str = ""
        discarded_messages = []

        for message in messages[::-1]:  # Reverse order to maintain chronological order
            sender: str = get_decoded_content(message.get(SENDER_NAME, "Unknown Sender"))
            timestamp: str = convert_timestamp_to_datetime(message.get(TIMESTAMP, "Unknown Timestamp"))
            logger.debug(f"Processing message from {sender} at {timestamp}")
            if CONTENT in message:
                document_text += (
                    MESSAGE_TEMPLATE.format(
                        sender_name=sender,
                        timestamp=timestamp,
                        content=get_decoded_content(message[CONTENT]),
                    )
                    + "\n"
                )
            elif PHOTOS in message:
                for photo in message[PHOTOS]:
                    uri = photo.get(URI, "")
                    document_text += (
                        PHOTO_TEMPLATE.format(
                            sender_name=sender,
                            timestamp=timestamp,
                            photo_path=uri,
                        )
                        + "\n"
                    )
            else:
                # If the message does not contain CONTENT or PHOTOS, log it as unsupported
                discarded_messages.append(message)
        if discarded_messages:
            logger.debug(
                f"Discarded {len(discarded_messages)} unsupported messages in {file_path}: {discarded_messages}"
            )

        if not document_text:
            logger.warning(f"No valid messages found in {file_path}. Skipping.")
            return Document(page_content="", metadata={})

        metadata: dict[str, str | list | dict] = extract_conversation_meta(conversation_data)
        logger.info(f"Extracted {len(messages)} messages from {file_path}")
        return Document(page_content=document_text, metadata=metadata)

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents from all JSON files in the data directory.

        Yields:
            Iterator[Document]: An iterator over Document objects.
        """
        for allowed_dir in self.allowed_dirs:
            dir_path: Path = self.data_dir / allowed_dir
            if not dir_path.exists():
                logger.warning(f"Directory {dir_path} does not exist.")
                continue

            logger.info(f"Searching for JSON files in {dir_path}")
            file_paths: list[Path] = list(dir_path.glob("**/*.json"))
            if not file_paths:
                logger.warning(f"No JSON files found in {dir_path}.")
                continue

            for file_path in file_paths:
                if file_path.is_file():
                    try:
                        yield self._parse_file(file_path)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse {file_path}: {e}")
                    except Exception as e:
                        logger.error(f"Unexpected error while processing {file_path}: {e}")
                else:
                    logger.warning(f"Skipping non-file path: {file_path}")
