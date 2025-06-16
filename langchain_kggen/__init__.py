"""
LangChain Knowledge Graph Generator

A powerful Python library for generating knowledge graphs from unstructured text
using LangChain and Large Language Models (LLMs).
"""

__version__ = "0.1.0"
__author__ = "LangChain KGGen Team"
__email__ = "contact@langchain-kggen.com"

from .kggen import KGGen
from .models.state import Graph
from .extractors.get_entities import get_entities
from .extractors.get_relations import get_relations
from .clustering.get_clusters import cluster_graph
from .utils.chunk import chunk_text

__all__ = [
    "KGGen",
    "Graph", 
    "get_entities",
    "get_relations", 
    "cluster_graph",
    "chunk_text",
    "__version__"
]