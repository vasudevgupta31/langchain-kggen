"""
# kggen_langchain.main
"""
import os
import time
import json
from typing import Union, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from langchain_core.language_models import BaseChatModel

from langchain_kggen.models.state import Graph
from langchain_kggen.extractors.get_entities import get_entities
from langchain_kggen.extractors.get_relations import get_relations
from langchain_kggen.clustering.get_clusters import cluster_graph
from langchain_kggen.utils.chunk import chunk_text


class KGGen:

    """
    Knowledge Graph Generation class using LangChain models.

    This class provides high-level methods to:
    - Generate knowledge graphs from input text or conversation data
    - Cluster semantically similar entities and relations
    - Aggregate multiple graphs into one unified structure
    """

    def __init__(self, llm: BaseChatModel, input_data: Union[str, List[Dict]]):
        """
        Initialize the KGGen class with a language model and input data.

        :param llm: The language model to use for entity/relation extraction and clustering.
        :type llm: BaseChatModel
        :param input_data: Either a raw string of text or a list of messages. 
                        If a list, each item must be a dictionary with 'role' and 'content' keys.
        :type input_data: Union[str, List[Dict]]

        :raises ValueError: If message format is invalid during processing.
        """
        self.lm = llm
        self.input_data = input_data

    def generate(self,
                 llm: BaseChatModel = None,
                 context: str = "",
                 chunk_size: Optional[int] = None,
                 cluster: bool = False,
                 max_workers: int = 10,
                 llm_delay: float = 2.0,
                 output_folder: Optional[str] = None,
                 file_name: Optional[str] = None) -> Graph:

        """
        Generate a knowledge graph from input text or conversation history.

        This method extracts entities and relations from the input, optionally splits it into chunks 
        for parallel processing, clusters similar items, and optionally writes the result to disk.

        :param llm: Optional override for the language model used in this run.
        :type llm: BaseChatModel, optional
        :param context: Additional context to guide entity and relation extraction.
        :type context: str
        :param chunk_size: If specified, splits the input into smaller chunks of the given size.
        :type chunk_size: int, optional
        :param cluster: Whether to perform clustering on entities and relations.
        :type cluster: bool
        :param max_workers: Number of threads to use for parallel chunk processing.
        :type max_workers: int
        :param llm_delay: Delay in seconds between chunk-level LLM calls to respect rate limits.
        :type llm_delay: float
        :param output_folder: If provided, writes the resulting graph to this directory.
        :type output_folder: str, optional
        :param file_name: Optional file name for the saved graph JSON file (default: 'knowledge_graph.json').
        :type file_name: str, optional

        :return: A generated knowledge graph with entities, relations, and edges.
        :rtype: Graph
        """

        # Process input data
        is_conversation = isinstance(self.input_data, list)

        if is_conversation:
            # Extract text from messages
            text_content = []
            for message in self.input_data:
                if (
                    not isinstance(message, dict)
                    or 'role' not in message
                    or 'content' not in message
                ):
                    raise ValueError(
                        "Messages must be dicts with 'role' and 'content' keys")
                if message['role'] in ['user', 'assistant']:
                    text_content.append(
                        f"{message['role']}: {message['content']}")

            # Join with newlines to preserve message boundaries
            processed_input = "\n".join(text_content)
        else:
            processed_input = self.input_data

        # Reinitialize model if a new one is provided
        llm = llm or self.lm

        if not chunk_size:
            entities = get_entities(llm=llm,
                                    input_data=processed_input,
                                    is_conversation=is_conversation,
                                    context=context)
            relations = get_relations(llm=llm,
                                      input_data=processed_input,
                                      entities=entities,
                                      is_conversation=is_conversation,
                                      context=context)
        else:
            chunks = chunk_text(text=processed_input, max_chunk_size=chunk_size)
            entities = set()
            relations = set()

            def process_chunk(chunk):
                chunk_entities = get_entities(llm=llm,
                                              input_data=chunk,
                                              is_conversation=is_conversation,
                                              context=context)
                chunk_relations = get_relations(llm=llm,
                                                input_data=chunk,
                                                entities=chunk_entities,
                                                is_conversation=is_conversation,
                                                context=context)
                time.sleep(llm_delay) # Respect LLM rate limits
                return chunk_entities, chunk_relations

            # Process chunks in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_chunk, chunks))

            # Combine results
            for chunk_entities, chunk_relations in results:
                entities.update(chunk_entities)
                relations.update(chunk_relations)

        graph = Graph(entities=entities,
                      relations=relations,
                      edges={relation[1] for relation in relations})

        if cluster:
            graph = self.cluster(graph, context)

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            file_name = file_name or 'knowledge_graph.json'
            output_path = os.path.join(output_folder, file_name)

            graph_dict = {'entities': list(entities),
                          'relations': list(relations),
                          'edges': list(graph.edges)}

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_dict, f, indent=2)

        return graph

    def cluster(self,
                graph: Graph,
                context: str = "",
                llm: BaseChatModel = None) -> Graph:
        """
        Cluster the entities and relations in a knowledge graph.

        This method uses an LLM to group semantically similar entities and relations into clusters,
        assigning a representative to each group and updating the graph accordingly.

        :param graph: The input knowledge graph to be clustered.
        :type graph: Graph
        :param context: Optional contextual information to guide clustering (e.g., domain or topic).
        :type context: str
        :param llm: Optional override for the language model to use during clustering.
        :type llm: BaseChatModel, optional

        :return: A clustered knowledge graph with updated entities, relations, and edge mappings.
        :rtype: Graph
        """
        return cluster_graph(llm=llm or self.lm, graph=graph, context=context)

    def aggregate(self, graphs: list[Graph]) -> Graph:
        """
        Aggregate multiple knowledge graphs into a single unified graph.

        This method merges the entities, relations, and edges from a list of `Graph` objects,
        removing duplicates via set operations.

        :param graphs: A list of Graph objects to be combined.
        :type graphs: list[Graph]

        :return: A single Graph containing the union of all entities, relations, and edges.
        :rtype: Graph
        """
        # Initialize empty sets for combined graph
        all_entities = set()
        all_relations = set()
        all_edges = set()

        # Combine all graphs
        for graph in graphs:
            all_entities.update(graph.entities)
            all_relations.update(graph.relations)
            all_edges.update(graph.edges)

        # Create and return aggregated graph
        return Graph(entities=all_entities, relations=all_relations, edges=all_edges)
