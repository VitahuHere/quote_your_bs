from datetime import datetime

from src.constants import CHAT_ID, NO_TITLE, PARTICIPANTS, THREAD_PATH, TITLE


def extract_chat_id(data: dict) -> str:
    """Extract chat ID from the conversation data.

    Args:
        data (dict): Conversation data containing the thread path.

    Returns:
        str: The chat ID extracted from the thread path.
    """
    thread_path: str = data[THREAD_PATH]
    _, thread_name = thread_path.rsplit("/", 1)
    if "_" not in thread_name:
        # meaning no text in group name
        return thread_name
    _, chat_id = thread_name.split("_", 1)
    return chat_id


def _extract_participants(data: list[dict[str, str]]) -> str:
    """Flatten a list of participant names into a single string.

    Args:
        data (list[dict[str, str]]): List of participant dictionaries.

    Returns:
        str: A comma-separated string of participant names.
    """
    return ", ".join(get_decoded_content(participant.get("name", "Unknown")) for participant in data)


def extract_conversation_meta(conversation_data: dict) -> dict[str, str | list]:
    """Extract metadata from conversation data.
    Args:
        conversation_data (dict): A dictionary containing conversation data.

    Returns:
        dict: A dictionary containing chat ID, title, and participants.
    """
    return {
        CHAT_ID: extract_chat_id(conversation_data),
        TITLE: conversation_data.get(TITLE, NO_TITLE),
        PARTICIPANTS: _extract_participants(conversation_data.get(PARTICIPANTS, [])),
    }


def get_decoded_content(content: str) -> str:
    """Decode content from double-encoded Unicode.

    Args:
        content (str): The content string to decode.

    Returns:
        str: The decoded content.
    """
    try:
        # Fix double-encoded Unicode
        fixed_content = content.encode('latin1').decode('utf-8')
    except UnicodeEncodeError:
        # Fallback if not double-encoded
        fixed_content = content
    return fixed_content


def convert_timestamp_to_datetime(timestamp: str | int) -> str:
    """Convert a timestamp in milliseconds to a human-readable datetime string.

    Args:
        timestamp (int): The timestamp in milliseconds.

    Returns:
        str: A formatted datetime string.
    """
    if isinstance(timestamp, str):
        try:
            timestamp = int(timestamp)
        except ValueError:
            return "Invalid Timestamp"

    dt = datetime.fromtimestamp(timestamp / 1000.0)
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "Invalid Timestamp"
