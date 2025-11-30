import base64
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAIError
from pydantic import BaseModel, Field

from src.constants import ALLOWED_IMAGE_EXTENSIONS, IMAGES
from src.prompts import image_description_prompt
from src.utils import get_vision_model

logger = logging.getLogger(__name__)


class ImageDescriptionOutput(BaseModel):
    """Model to represent the output of image description."""

    description: str = Field(description="Description of the image.")


def _is_img_valid(img_path: str | Path) -> bool:
    """Check if the image file is valid."""
    img_path = Path(img_path)
    if not img_path.exists():
        raise ValueError(f"{img_path} does not exist")

    if not img_path.is_file():
        raise ValueError(f"{img_path} is not a file")

    if img_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"{img_path} is not a valid image file")

    return True


def _encode_image_to_base64(img_path: str | Path) -> str:
    """Encode the image file to a base64 string."""
    img_path = Path(img_path)
    with open(img_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/{img_path.suffix[1:]};base64,{encoded_string}"


def _get_image_payload(base_64_image: str) -> dict:
    return {"type": "image_url", "image_url": {"url": base_64_image}}


def describe_image(base_64_image: str) -> str:
    """Returns a description of the image file.

    Args:
        base_64_image (str): The base64 encoded image string.

    Returns:
        str: A description of the image file.
    """
    vision_model = get_vision_model()
    structured_model = vision_model.with_structured_output(schema=ImageDescriptionOutput)
    response: ImageDescriptionOutput = structured_model.invoke(
        [
            SystemMessage(content=image_description_prompt),
            HumanMessage(content=[_get_image_payload(base_64_image)]),
        ]
    )
    return response.description


def _find_image_tags(content: str) -> list[str]:
    """Find all image tags in the Markdown content."""
    import re

    # Regular expression to match Markdown image syntax ![alt text](image_url)
    pattern = r'!\[.*?\]\((.*?)\)'
    return re.findall(pattern, content)


def describe_images_in_document(document: Document) -> Document:
    """Returns a document with descriptions of images in the document.

    Given a Document with Markdown content, this function extracts image tags
    ![]() from the content, checks if the image in present in the metadata,
    and if so, generates a description for each image using a vision model
    and puts the description in the content in square brackets.
    ![](img_path) -> ![description](img_path)

    Args:
        document (Document): The document containing Markdown content with images.

    Returns:
        Document: A new Document with the same content, but with image descriptions
        added to the metadata.
    """
    if not document.metadata.get(IMAGES):
        raise ValueError(f"Document metadata must contain '{IMAGES}' key.")

    content: str = document.page_content
    image_tags: list[str] = _find_image_tags(content)
    if not image_tags:
        return document

    images = document.metadata[IMAGES]
    for tag in image_tags:
        img_path = tag.split("](")[-1].rstrip(")")
        if img_path not in images:
            logger.warning(f"Image {img_path} not found in document metadata.")
            continue

        try:
            base_64_image = images[img_path]
            description = describe_image(base_64_image)
            content = content.replace(f"![]({tag}", f"![{description}]({img_path})")
        except (ValueError, OpenAIError) as e:
            logger.error(f"Error processing image {img_path}: {e}")

    return Document(page_content=content, metadata=document.metadata)
