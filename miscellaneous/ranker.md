#  Quick Ranker

a document ranking system that provides similarity-based ranking, filtering, and top-K selection.

## Quick Start

```python
from srswti_axis import SrswtiRanker

# Initialize ranker
ranker = SrswtiRanker()

# Rank documents
ranked_docs = ranker.rank_documents(
    query="your search query",
    candidates=list_of_documents
)
```

## Core Features

### Document Ranking

```python
ranked_docs = ranker.rank_documents(
    query="search query",
    candidates=documents,
    batch_size=64
)
```

Parameters:
- `query`: Search query string
- `candidates`: List of documents to rank
- `batch_size`: Processing batch size (default: 64)

Returns:
- Sorted list of documents by relevance

### Document Filtering

```python
filtered_docs = ranker.filter_documents(
    query="reference query",
    candidates=documents,
    threshold=0.3
)
```

Parameters:
- `query`: Reference query string
- `candidates`: Documents to filter
- `threshold`: Minimum similarity threshold (default: 0.3)

Returns:
- List of documents above threshold

### Top-K Selection

```python
top_docs = ranker.get_top_k(
    query="search query",
    candidates=documents,
    k=2
)
```

Parameters:
- `query`: Search query string
- `candidates`: Candidate documents
- `k`: Number of documents to retrieve (default: 2)

Returns:
- Top K most similar documents

## Example Usage

### Basic Document Ranking

```python
ranker = SrswtiRanker()

# Sample documents
documents = [
    "First document content",
    "Second document content",
    "Third document content"
]

# Rank documents
ranked = ranker.rank_documents(
    query="search terms",
    candidates=documents
)

# Print ranked results
for doc in ranked:
    print(doc)
```

### Filtering with Threshold

```python
# Filter documents above similarity threshold
filtered = ranker.filter_documents(
    query="reference document",
    candidates=documents,
    threshold=0.5
)

print(f"Found {len(filtered)} similar documents")
```

### Getting Top Results

```python
# Get top 3 most similar documents
top_results = ranker.get_top_k(
    query="search query",
    candidates=documents,
    k=3
)

print("Top 3 matching documents:")
for doc in top_results:
    print(doc)
```


## Best Practices

1. Document Preparation
   - Clean input text
   - Consistent formatting
   - Appropriate length

2. Query Formulation
   - Clear, focused queries
   - Relevant search terms
   - Appropriate length

3. Performance Optimization
   - Use appropriate batch sizes
   - Adjust thresholds as needed
   - Consider document count