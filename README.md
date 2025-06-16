# Knowledge Graph Generator with Langchain

[![PyPI version](https://badge.fury.io/py/langchain-kggen.svg)](https://badge.fury.io/py/langchain-kggen)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful Python library for generating knowledge graphs from unstructured text using LangChain and Large Language Models (LLMs). Extract entities, relationships, and create structured knowledge representations with support for clustering, chunking, and parallel processing.

Note: This project and the contents of it are inspired by KGgen as proposed by (https://arxiv.org/pdf/2502.09956) dated 14 Feb 2025.

## Features

-  `LLM-Powered Extraction`: Leverage any LangChain-compatible language model for intelligent entity and relation extraction
-  `Knowledge Graph Generation`: Create structured graphs with entities, relations, and edges from raw text or conversations
-  `Semantic Clustering`: Automatically cluster similar entities and relations using LLM-based semantic understanding
-  `Parallel Processing`: Handle large texts efficiently with concurrent chunk processing
-  `Conversation Support`: Extract knowledge graphs from conversational data (chat logs, dialogues)
-  `Flexible Input`: Process both plain text and structured conversation formats
-  `Customizable`: Fine-tune extraction with context, chunk sizes, and clustering parameters
-  `Export Support`: Save generated graphs to JSON format for further analysis

## Installation

```bash
pip install langchain-kggen
```

## Framework workflow
As proposed in the original paper.
![Framework](./assets/kggen-dig.png)


## Quick Start

### Basic Usage

```python
from langchain_kggen import KGGen
from langchain_openai import ChatOpenAI  # or any LangChain-compatible LLM
from langchain_ollama import ChatOllama

# Initialize your LLM
llm = ChatOpenAI(model="gpt-4o")
llm = Chatollama(model='deepseek-r1:32b')  # if working with local models

# Your input text
text = """
Apple Inc. is a technology company founded by Steve Jobs in 1976. 
The company is headquartered in Cupertino, California. 
Tim Cook is the current CEO of Apple.
"""

# Create KGGen instance
kg_generator = KGGen(llm=llm, input_data=text)

# Generate knowledge graph
graph = kg_generator.generate()

print("Entities:", graph.entities)
print("Relations:", graph.relations)
print("Edges:", graph.edges)
```

### Processing Conversations

```python
# Conversation format
conversation = [
    {"role": "user", "content": "Tell me about artificial intelligence"},
    {"role": "assistant", "content": "AI is a field of computer science focused on creating intelligent machines"},
    {"role": "user", "content": "What are the main types of AI?"},
    {"role": "assistant", "content": "The main types include narrow AI, general AI, and superintelligence"}
]

# Generate knowledge graph from conversation
kg_generator = KGGen(llm=llm, input_data=conversation)
graph = kg_generator.generate()
```

### Advanced Features

```python
# Generate with clustering and chunking
graph = kg_generator.generate(
    context="Technology and business domain",  # Additional context
    chunk_size=1000,                          # Split large texts
    cluster=True,                             # Enable semantic clustering
    max_workers=5,                            # Parallel processing
    llm_delay=1.0,                           # Rate limiting
    output_folder="./output",                 # Save to file
    file_name="my_knowledge_graph.json"      # Custom filename
)

# Access clustering information
if graph.entity_clusters:
    print("Entity clusters:", graph.entity_clusters)
if graph.edge_clusters:
    print("Edge clusters:", graph.edge_clusters)
```

### Standalone Clustering

```python
# Cluster an existing graph
clustered_graph = kg_generator.cluster(
    graph=graph,
    context="Business and technology context"
)
```

### Aggregating Multiple Graphs

```python
# Combine multiple knowledge graphs
graph1 = kg_generator.generate()
graph2 = kg_generator.generate()

aggregated_graph = kg_generator.aggregate([graph1, graph2])
```

## <Architecture

The library consists of several key components:

### Core Classes

- **`KGGen`**: Main interface for knowledge graph generation
- **`Graph`**: Pydantic model representing a knowledge graph structure

### Extraction Modules

- **`get_entities`**: Extract entities (subjects/objects) from text
- **`get_relations`**: Extract subject-predicate-object relations
- **`get_clusters`**: Perform semantic clustering of entities and relations

### Utilities

- **`chunk_text`**: Intelligent text chunking with sentence boundary respect
- **`state`**: Graph data models and structures

## Graph Structure

The generated `Graph` object contains:

```python
class Graph(BaseModel):
    entities: set[str]                              # All unique entities
    edges: set[str]                                 # All unique predicates/relations
    relations: set[Tuple[str, str, str]]           # (subject, predicate, object) triples
    entity_clusters: Optional[dict[str, set[str]]]  # Entity clustering mappings
    edge_clusters: Optional[dict[str, set[str]]]    # Edge clustering mappings
```
## Eval report 
The table shows average score across 10 text inputs of various complexity then using a frontier high rasining LLM as judge:
![Knowledge Graph Example](./assets/eval1.png)

##  Configuration Options

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `BaseChatModel` | None | Override default LLM |
| `context` | `str` | `""` | Additional context for extraction |
| `chunk_size` | `int` | None | Split text into chunks |
| `cluster` | `bool` | `False` | Enable semantic clustering |
| `max_workers` | `int` | `10` | Parallel processing threads |
| `llm_delay` | `float` | `2.0` | Delay between LLM calls |
| `output_folder` | `str` | None | Save location |
| `file_name` | `str` | `'knowledge_graph.json'` | Output filename |

##  Use Cases

- ** Document Analysis**: Extract key concepts and relationships from research papers, reports
- ** Conversation Mining**: Analyze chat logs, interviews, customer support tickets
- ** News Processing**: Build knowledge bases from news articles and press releases
- ** Information Extraction**: Transform unstructured data into structured knowledge
- ** AI Training Data**: Create structured datasets for machine learning applications
- ** Business Intelligence**: Extract insights from business documents and communications

## Best Practices

1. **Context Matters**: Provide relevant context to improve extraction accuracy
2. **Chunk Large Texts**: Use chunking for texts > 2000 tokens to avoid LLM limits
3. **Enable Clustering**: Use clustering to merge similar entities and reduce noise
4. **Rate Limiting**: Adjust `llm_delay` based on your LLM provider's rate limits
5. **Iterative Refinement**: Review outputs and adjust context/parameters as needed

##  Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- = [Documentation](https://github.com/yourusername/langchain-kggen/wiki)
- = [Issue Tracker](https://github.com/yourusername/langchain-kggen/issues)
- = [Discussions](https://github.com/yourusername/langchain-kggen/discussions)

## Acknowledgments

- Built on top of the excellent [LangChain](https://github.com/langchain-ai/langchain) framework
- Inspired by advances in Large Language Model capabilities for information extraction
- Thanks to the open-source community for feedback and contributions

---
