from langchain_core.documents import Document


def remove_image_tags(document: Document) -> Document:
    """
    Remove image tags from the document content.

    Args:
        document (Document): The input document.

    Returns:
        Document: The document with image tags removed.
    """
    import re

    # Regular expression to match image tags like ![description](image_url)
    image_tag_pattern = r"!\[.*?\]\(.*?\)"
    cleaned_content = re.sub(image_tag_pattern, "", document.page_content)
    return Document(page_content=cleaned_content, metadata=document.metadata)
