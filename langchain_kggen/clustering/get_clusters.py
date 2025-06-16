"""
Clustering module for entity and relation abstraction in knowledge graphs.

This module provides utilities for clustering semantically similar entities and relations
extracted from unstructured data. It leverages large language models via LangChain to:
- Identify clusters of conceptually similar items (e.g., entity aliases, synonymous relations)
- Validate proposed clusters for coherence
- Choose representative labels for each cluster
- Update graph structure by merging clustered nodes and updating relation triplets accordingly

Typical usage involves calling `cluster_graph(...)` on a `Graph` object containing entities, edges,
and (subject, predicate, object) relations. The resulting graph has deduplicated entities and
relations, and additional cluster mappings.

Designed for use with `BaseChatModel` from LangChain.
"""
from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_kggen.models.state import Graph


LOOP_N = 8
BATCH_SIZE = 10
ItemType = Literal["entities", "edges"]


class Cluster(BaseModel):
    """Cluster"""
    representative: str
    members: set[str]


def get_suggested_cluster(llm: BaseChatModel, items: set[str], remaining_items: set[str], context: str):
    """
    Suggests a cluster of semantically similar items from the remaining set.

    Uses an LLM to identify one cluster of conceptually related terms.

    :param llm: The language model instance.
    :type llm: BaseChatModel
    :param items: The full set of items being clustered.
    :type items: set[str]
    :param remaining_items: Subset of items not yet clustered.
    :type remaining_items: set[str]
    :param context: Context or domain information for the items.
    :type context: str

    :returns: A list of items proposed to form one cluster.
    :rtype: list[str]
    """

    cluster_extract_prompt = """Find one cluster of related items from the list.
    A cluster should contain items that are the same in meaning, with different tenses, plural forms, stem forms, or cases. 
    Return populated list only if you find items that clearly belong together, else return empty list.
    <ITEMS>
    {ITEMS}
    </ITEMS>

    <CONTEXT>
    The larger context in which the items appear:
    {CONTEXT}
    </CONTEXT>
    """

    ItemsLiteral = Literal[tuple(items)]

    class Output(BaseModel):
        """Output Schema"""
        cluster: list[ItemsLiteral]

    # Base prompt
    prompt_template = PromptTemplate(input_variables=["ITEMS", "CONTEXT"],
                                     template=cluster_extract_prompt)
    prompt = prompt_template.format(ITEMS=remaining_items, CONTEXT=context)

    # Model Invoke
    model = llm.with_structured_output(Output)
    return model.invoke(prompt).cluster


def validate_cluster(llm: BaseChatModel, cluster: set[str], context: str):
    """
    Validates whether the given items belong together as a cluster.

    Ensures that all items share sufficient semantic similarity.

    :param llm: The language model instance.
    :type llm: BaseChatModel
    :param cluster: Set of items to validate as a coherent group.
    :type cluster: set[str]
    :param context: Contextual or domain information.
    :type context: str

    :returns: A list of items confirmed to belong in the cluster.
    :rtype: list[str]
    """

    validate_cluster_prompt = """Validate if these items belong in the same cluster.
    A cluster should contain items that are the same in meaning, with different tenses, plural forms, stem forms, or cases. 
    Return populated list only if you find items that clearly belong together, else return empty list.
    <CLUSTER>
    {CLUSTER}
    </CLUSTER>

    <CONTEXT>
    The larger context in which the items appear.
    {CONTEXT}
    </CONTEXT>
    """

    ClusterLiteral = Literal[tuple(cluster)]

    class Output(BaseModel):
        """Output Schema"""
        validated_items: list[ClusterLiteral] = Field(
            ..., desc="All the items that belong together in the cluster")

    # Base prompt
    prompt_template = PromptTemplate(
        input_variables=["CLUSTER", "CONTEXT"], template=validate_cluster_prompt)
    prompt = prompt_template.format(CLUSTER=cluster, CONTEXT=context)

    # Model
    model = llm.with_structured_output(Output)
    return model.invoke(prompt).validated_items


def check_existing_cluster(llm: BaseChatModel, items: list[str], clusters: list[Cluster], context: str):
    """
    Determines whether new items match any existing clusters.

    For each item, returns the representative it best matches, or None.

    :param llm: The language model instance.
    :type llm: BaseChatModel
    :param items: Items to check for inclusion in existing clusters.
    :type items: list[str]
    :param clusters: Current list of existing clusters.
    :type clusters: list[Cluster]
    :param context: Additional domain or task context.
    :type context: str

    :returns: List of cluster representatives (or None) for each item.
    :rtype: list[Optional[str]]
    """

    check_existing_cluster_prompt = """Let's think step by step in order to determine if the given items can be added to any of the existing clusters.
    Return representative of matching cluster for each item, or None if there is no match.

    <ITEMS>
    {ITEMS}
    </ITEMS>

    <CLUSTERS>
    Mapping of cluster representatives to their cluster members:
    {CLUSTERS}
    </CLUSTERS>

    <CONTEXT>
    The larger context in which the items appear:
    {CONTEXT}
    </CONTEXT>
    """

    class Output(BaseModel):
        """Output Schema"""
        cluster_reps_that_items_belong_to: list[Optional[str]] = Field(...,
                                                                       desc="Ordered list of cluster representatives where each is the cluster where that item belongs to, "
                                                                       "or None if no match. THIS LIST LENGTH IS SAME AS ITEMS LIST LENGTH")

    # Base prompt
    prompt_template = PromptTemplate(input_variables=[
                                     "ITEMS", "CLUSTERS", "CONTEXT"],
                                     template=check_existing_cluster_prompt)
    prompt = prompt_template.format(
        ITEMS=items, CLUSTERS=clusters, CONTEXT=context)

    # Model
    model = llm.with_structured_output(Output)
    return model.invoke(prompt).cluster_reps_that_items_belong_to


def choose_rep(llm: BaseChatModel, cluster: set[str], context: str):
    """
    Selects the most representative item from a cluster.

    Typically favors shorter, generalizable names.

    :param llm: The language model instance.
    :type llm: BaseChatModel
    :param cluster: Set of similar items.
    :type cluster: set[str]
    :param context: Additional context to inform the choice.
    :type context: str

    :returns: The chosen representative item.
    :rtype: str
    """

    representative_base_prompt = """Select the best item name to represent the cluster, ideally from the cluster.
    Prefer shorter names and generalizability across the cluster.
    <Cluster>
    {CLUSTER}
    </Cluster>

    <CONTEXT>
    The larger context in which the items appear:
    {CONTEXT}
    </CONTEXT>
    """

    class Output(BaseModel):
        """Output Schema"""
        representative: str

    # Base prompt
    prompt_template = PromptTemplate(
        input_variables=["CLUSTER", "CONTEXT"], template=representative_base_prompt)
    prompt = prompt_template.format(CLUSTER=cluster, CONTEXT=context)

    # Model
    model = llm.with_structured_output(Output)
    return model.invoke(prompt).representative


def cluster_items(llm: BaseChatModel,
                  items: set[str],
                  item_type: ItemType = "entities",
                  context: str = "") -> tuple[set[str], dict[str, set[str]]]:
    """
    Clusters a set of items into semantically coherent groups.

    This function forms clusters, validates them, assigns remaining items to
    existing clusters, and falls back to single-item clusters if needed.

    :param llm: The language model used for clustering tasks.
    :type llm: BaseChatModel
    :param items: The set of items to be clustered.
    :type items: set[str]
    :param item_type: Type of item (either "entities" or "edges").
    :type item_type: Literal["entities", "edges"]
    :param context: Optional context to guide the clustering.
    :type context: str

    :returns: Tuple of representative item set and representative-to-members mapping.
    :rtype: tuple[set[str], dict[str, set[str]]]
    """

    context = f"{item_type} of a graph extracted from source text." + context
    remaining_items = items.copy()
    clusters: list[Cluster] = []
    no_progress_count = 0

    while len(remaining_items) > 0:

        suggested_cluster = get_suggested_cluster(llm=llm,
                                                  items=items,
                                                  remaining_items=remaining_items,
                                                  context=context)

        if len(suggested_cluster) > 0:

            validated_cluster = validate_cluster(llm=llm,
                                                 cluster=suggested_cluster,
                                                 context=context)

            if len(validated_cluster) > 1:
                no_progress_count = 0
                representative = choose_rep(
                    llm=llm, cluster=validated_cluster, context=context)
                clusters.append(
                    Cluster(representative=representative, members=validated_cluster))
                remaining_items = {
                    item for item in remaining_items if item not in validated_cluster}
                continue

        no_progress_count += 1

        if no_progress_count >= LOOP_N or len(remaining_items) == 0:
            break

    if len(remaining_items) > 0:
        items_to_process = list(remaining_items)

        for i in range(0, len(items_to_process), BATCH_SIZE):
            batch = items_to_process[i:min(
                i + BATCH_SIZE, len(items_to_process))]

            if not clusters:
                for item in batch:
                    clusters.append(
                        Cluster(representative=item, members={item}))
                continue

            cluster_reps = check_existing_cluster(
                llm=llm, items=batch, clusters=clusters, context=context)

            # Map representatives to their cluster objects for easier lookup
            # Ensure cluster_map uses the most up-to-date list of clusters
            cluster_map = {c.representative: c for c in clusters}

            # Determine assignments for batch items based on validation
            # Stores item -> assigned representative. If None, item needs a new cluster.
            item_assignments: dict[str, Optional[str]] = {}

            for i, item in enumerate(batch):
                # Default: item might become its own cluster if no valid assignment found
                item_assignments[item] = None

                # Get the suggested representative from the LLM call
                rep = cluster_reps[i] if i < len(cluster_reps) else None

                target_cluster = None
                # Check if the suggested representative corresponds to an existing cluster
                if rep is not None and rep in cluster_map:
                    target_cluster = cluster_map[rep]

                if target_cluster:
                    # If the item is already the representative or a member, assign it definitively
                    if item == target_cluster.representative or item in target_cluster.members:
                        item_assignments[item] = target_cluster.representative
                        continue  # Move to the next item

                    # Validate adding the item to the existing cluster's members
                    potential_new_members = target_cluster.members | {item}
                    try:
                        # Call the validation signature
                        v_result = validate_cluster(
                            llm=llm, cluster=potential_new_members, context=context)
                        # Ensure result is a set
                        validated_items = set(v_result.validated_items)

                        # Check if the item was validated as part of the cluster AND
                        # the size matches the expected size after adding.
                        # This assumes 'validate' confirms membership without removing others.
                        if item in validated_items and len(validated_items) == len(potential_new_members):
                            # Validation successful, assign item to this cluster's representative
                            item_assignments[item] = target_cluster.representative
                        # Else: Validation failed or item rejected, item_assignments[item] remains None

                    except Exception as e:
                        # Handle potential errors during the validation call
                        # TODO: Add proper logging
                        print(
                            f"""Validation failed for item '{item}' potentially
                            belonging to cluster '{target_cluster.representative}': {e}""")
                        # Keep item_assignments[item] as None, indicating it needs a new cluster

                # Else (no valid target_cluster found for the suggested 'rep'):
                # item_assignments[item] remains None, will become a new cluster.

            # Process the assignments determined above
            new_cluster_items = set()  # Collect items needing a brand new cluster
            for item, assigned_rep in item_assignments.items():
                if assigned_rep is not None:
                    # Item belongs to an existing cluster, add it to the members set
                    # Ensure the cluster exists in the map (should always be true here)
                    if assigned_rep in cluster_map:
                        cluster_map[assigned_rep].members.add(item)
                    else:
                        # This case should ideally not happen if logic is correct
                        # TODO: Add logging for this unexpected state
                        print(
                            f"""Error: Assigned representative '{assigned_rep}' not found in
                            cluster_map for item '{item}'. Creating new cluster.""")
                        # Avoid creating if item itself is already a rep
                        if item not in cluster_map:
                            new_cluster_items.add(item)
                else:
                    # Item needs a new cluster, unless it's already a representative itself
                    if item not in cluster_map:
                        new_cluster_items.add(item)

            # Create the new Cluster objects for items that couldn't be assigned
            for item in new_cluster_items:
                # Final check: ensure a cluster with this item as rep doesn't exist
                if item not in cluster_map:
                    new_cluster = Cluster(representative=item, members={item})
                    clusters.append(new_cluster)
                    # Update map for internal consistency
                    cluster_map[item] = new_cluster

    # Prepare the final output format expected by the calling function:
    # 1. A dictionary mapping representative -> set of members
    # 2. A set containing all unique representatives
    final_clusters_dict = {c.representative: c.members for c in clusters}
    new_items = set(final_clusters_dict.keys())  # The set of representatives

    return new_items, final_clusters_dict


def cluster_graph(llm: BaseChatModel, graph: Graph, context: str = "") -> Graph:
    """
    Clusters both entities and edges in a knowledge graph.

    Updates the graph structure with representative terms and revised relations.

    :param llm: The language model used for clustering.
    :type llm: BaseChatModel
    :param graph: The input graph containing entities, edges, and relations.
    :type graph: Graph
    :param context: Optional text context for more accurate clustering.
    :type context: str

    :returns: Updated graph with abstracted clusters and revised relations.
    :rtype: Graph
    """
    entities, entity_clusters = cluster_items(llm=llm,
                                              items=graph.entities,
                                              item_type="entities",
                                              context=context)

    edges, edge_clusters = cluster_items(llm=llm,
                                         items=graph.edges,
                                         item_type="edges",
                                         context=context)

    # Update relations based on clusters
    relations: set[tuple[str, str, str]] = set()
    for s, p, o in graph.relations:
        # Look up subject in entity clusters
        if s not in entities:
            for rep, cluster in entity_clusters.items():
                if s in cluster:
                    s = rep
                    break

        # Look up predicate in edge clusters
        if p not in edges:
            for rep, cluster in edge_clusters.items():
                if p in cluster:
                    p = rep
                    break

        # Look up object in entity clusters
        if o not in entities:
            for rep, cluster in entity_clusters.items():
                if o in cluster:
                    o = rep
                    break

        relations.add((s, p, o))

    return Graph(entities=entities,
                 edges=edges,
                 relations=relations,
                 entity_clusters=entity_clusters,
                 edge_clusters=edge_clusters)
