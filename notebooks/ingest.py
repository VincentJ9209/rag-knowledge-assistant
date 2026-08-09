"""Data ingestion and search-index utilities for the RAG pipeline."""

import requests
from minsearch import Index


def load_faq_data():
    """Load FAQ documents from the DataTalks.Club course knowledge base."""
    docs_url = "https://datatalks.club/faq/json/courses.json"

    response = requests.get(docs_url)
    response.raise_for_status()
    courses_raw = response.json()

    documents = []
    url_prefix = "https://datatalks.club/faq"

    for course in courses_raw:
        course_url = f"{url_prefix}{course['path']}"

        course_response = requests.get(course_url)
        course_response.raise_for_status()

        documents.extend(course_response.json())

    return documents


def build_index(documents):
    """Build a MinSearch index from FAQ documents."""
    index = Index(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
    )

    index.fit(documents)

    return index