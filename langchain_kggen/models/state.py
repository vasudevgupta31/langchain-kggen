"""
Graph Data Model
================

This module defines the `Graph` class, a Pydantic model used to represent
a structured knowledge graph. It supports sets of entities and relations,
as well as optional cluster mappings for both entities and edges.

The model is designed to serve as the output format for information extraction,
knowledge graph generation, or entity-relation clustering tasks.

Classes:
--------
- Graph: Represents a graph with entities, edges, and their relationships, 
         along with optional clustering information.
"""
from typing import Tuple, Optional
from pydantic import BaseModel, Field


class Graph(BaseModel):
    """
    A Pydantic model representing a structured knowledge graph.

    This model holds:
    - Unique entities and edges.
    - Directed relations in the form of (subject, predicate, object) triples.
    - Optional clusters of semantically related entities or edge types.

    :ivar entities: A set of all identified entities (including canonical and additional).
    :vartype entities: set[str]

    :ivar edges: A set of all identified edge types (predicates).
    :vartype edges: set[str]

    :ivar relations: A set of (subject, predicate, object) triples forming the graph's structure.
    :vartype relations: set[Tuple[str, str, str]]

    :ivar entity_clusters: Optional dictionary mapping canonical entities to sets of clustered variants.
    :vartype entity_clusters: Optional[dict[str, set[str]]]

    :ivar edge_clusters: Optional dictionary mapping canonical edge types to sets of clustered variants.
    :vartype edge_clusters: Optional[dict[str, set[str]]]
    """
    entities: set[str] = Field(
        ..., description="All entities including additional ones from response")
    edges: set[str] = Field(..., description="All edges")
    relations: set[Tuple[str, str, str]
                   ] = Field(..., description="List of (subject, predicate, object) triples")
    entity_clusters: Optional[dict[str, set[str]]] = None
    edge_clusters: Optional[dict[str, set[str]]] = None
