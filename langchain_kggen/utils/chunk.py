# -*- coding: utf-8 -*-
"""
This module provides a function to chunk text into smaller pieces while 
respecting sentence boundaries.
"""
import nltk


# Ensure the punkt tokenizer is downloaded
def ensure_nltk_resource(resource_path, resource_name):
    """
    Ensure that the specified NLTK resource is available, downloading it if necessary.
    :param resource_path: The path to the NLTK resource.
    :param resource_name: The name of the NLTK resource to download if not found.
    """
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(resource_name, quiet=True)

ensure_nltk_resource('tokenizers/punkt', 'punkt')
ensure_nltk_resource('tokenizers/punkt_tab', 'punkt_tab')


def chunk_text(text: str, max_chunk_size=500) -> list[str]:
    """
    Chunk text by sentence, respecting a maximum chunk size.
    Falls back to word-based chunking if a single sentence is too large.
    
    :param text: The text to chunk.
    :param max_chunk_size: The maximum length (in characters) of any chunk.
    :return: A list of text chunks.
    """
    # Step 1: Split text into sentences
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence stays within the limit, append it.
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            # If the current chunk has some content, push it and start a new one.
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Check if the sentence itself is larger than the limit.
            # If yes, chunk it by words (fallback).
            if len(sentence) > max_chunk_size:
                words = sentence.split()
                temp_chunk = ""

                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= max_chunk_size:
                        temp_chunk += word + " "
                    else:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = word + " "

                # Add the leftover if any
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
            else:
                # If the sentence is smaller than max_chunk_size, just start a new chunk with it.
                current_chunk = sentence + " "

    # If there's a leftover chunk that didn't get pushed, add it
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
