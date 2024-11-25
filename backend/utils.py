from typing import List

from langchain_core.documents import Document


def format_documents(documents: List[Document]):
    return "\n\n".join(doc.page_content for doc in documents)
