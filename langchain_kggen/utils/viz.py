"""
Visualization utilities for graph representation.
"""
import networkx as nx


def draw_graph(graph):
    """
    Draws a directed graph using NetworkX and Matplotlib.
    Args:
        graph (Graph): A graph object containing nodes and relations.
    """

    # Create directed graph
    G = nx.DiGraph()

    # Add edges
    for src, rel, tgt in graph.relations:
        G.add_edge(src, tgt, label=rel)

    # Layout
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=8)

    # Draw edges with labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkred')
