# SRSWTI Ultimate

search and ranking system integrating neural embeddings, graph-based document relationships, and hybrid scoring for advanced document retrieval.


## Quick Start

```python
from srswti_axis import SRSWTIUltimate

# Initialize search engine
search_engine = SRSWTIUltimate()

# Index your documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning models require significant computational resources."
]
search_engine.index_documents(documents)

# Perform search
results = search_engine.search(
    query="machine learning",
    n_results=5,
    ranking_method='combined'
)
```

## Core Components

### Search Ranker

```python
from srswti_axis import SRSWTISearchRanker

ranker = SRSWTISearchRanker(
    embedding_model='srswti-neural-embedder-v1',
    use_pagerank=True
)
```

#### Parameters
- `embedding_model`: Model for semantic embeddings (defaults to 'srswti-neural-embedder-v1')
- `use_pagerank`: Enable PageRank scoring (defaults to True)

### Document Graph Building

```python
graph = ranker.build_document_graph(
    documents=documents,
    threshold=0.5
)
```

#### Parameters
- `documents`: List of document strings
- `threshold`: Similarity threshold for edge creation (default: 0.5)

### Document Ranking

```python
results = ranker.rank_documents(
    query="your query",
    documents=documents,
    combine_method='weighted_sum',
    alpha=0.3
)
```

#### Parameters
- `query`: Search query string
- `documents`: List of documents to rank
- `combine_method`: Score combination method ('weighted_sum' or 'multiplication')
- `alpha`: Weight for PageRank score (1-alpha for similarity score)

## Complete Search Engine (SRSWTIUltimate)

### Initialization
```python
engine = SRSWTIUltimate()
```

### Document Indexing
```python
engine.index_documents(documents)
```

### Search Execution
```python
results = engine.search(
    query="your search query",
    n_results=5,
    ranking_method='combined'
)
```

#### Parameters
- `query`: Search query string
- `n_results`: Number of results to return (default: 5)
- `ranking_method`: Ranking method to use ('combined' or other)

## Result Format

The search results are returned as a list of dictionaries containing:
```python
{
    'document': str,        # The document text
    'score': float,        # Combined ranking score
    'pagerank': float,     # PageRank score
    'cluster': int         # Document cluster ID
}
```

## Implementation Examples

### Basic Search Implementation
```python
# Initialize search engine
search_engine = SRSWTIUltimate()

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning models require significant computational resources.",
    "Natural language processing helps computers understand human language."
]

# Index documents
search_engine.index_documents(documents)

# Perform search
results = search_engine.search(
    query="machine learning AI",
    n_results=3
)

# Process results
for result in results:
    print(f"Document: {result['document']}")
    print(f"Score: {result['score']:.3f}")
    print(f"PageRank: {result['pagerank']:.3f}")
    print(f"Cluster: {result['cluster']}\n")
```

### Document Clustering
```python
# Get document clusters
ranker = SRSWTISearchRanker()
ranker.build_document_graph(documents)
clusters = ranker.get_document_clusters()

# Access cluster information
for cluster_id, doc_indices in clusters.items():
    print(f"Cluster {cluster_id}: {doc_indices}")
```

## Memory Management

The implementation includes memory cleanup capabilities:
```python
# Clear large model references
search_engine.ranker.embedder = None
search_engine.ranker.nlp = None
```


## Model Features

### Document Graph
- Builds similarity-based document graph
- Uses cosine similarity for edge weights
- Implements minimum connectivity guarantee
- Supports custom similarity thresholds

### Ranking System
- Combines PageRank and semantic similarity
- Supports multiple combination methods
- Provides cluster information
- Includes document similarity analysis

### Search Features
- Multi-document indexing
- Configurable result count
- Combined ranking scores
- Cluster identification