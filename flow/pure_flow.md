
# Pure Flow

advanced document merging system using multi-strategy approaches, semantic coherence optimization, and adaptive processing for content integration.

#### nmf decomposition
topic modeling:
$V \approx WH$ where:
- V: document-term matrix
- W: document-topic weights
- H: topic-term weights

coherence optimization:
- topic keyword extraction
- semantic alignment
- hierarchical merging
- cross-topic linking

### 4. sequential merging

#### overlap handling
chunk creation:
$C_i = [S_{i-o}, ..., S_i, ..., S_{i+o}]$
where:
- $C_i$: chunk i
- $S_i$: sentence i
- o: overlap size

## Quick Start

```python
from srswti_axis import SRSWTIPureFlow

# Initialize merger
merger = SRSWTIPureFlow(
    embedding_model='all-MiniLM-L6-v2',
    language='en',
    spacy_model='en_core_web_sm'
)

# Merge documents using different strategies
similarity_merged = merger.process(documents, method='similarity')
sequential_merged = merger.process(documents, method='sequential')
graph_merged = merger.process(documents, method='graph')
topic_merged = merger.process(documents, method='topic')
```

## Merging Strategies

### 1. Similarity-Based Merging

```python
merged = merger.merge_by_similarity(
    documents,
    threshold=0.5,
    strategy='clustering',
    min_cluster_size=2,
    adaptive_threshold=True
)
```

Parameters:
- `threshold`: Similarity threshold (or adaptive)
- `strategy`: 'clustering' or 'pairwise'
- `min_cluster_size`: Minimum cluster size
- `adaptive_threshold`: Enable dynamic threshold

### 2. Sequential Merging

```python
merged = merger.merge_sequential(
    documents,
    max_chunk_size=1000,
    overlap=True
)
```

Parameters:
- `max_chunk_size`: Maximum chunk size
- `overlap`: Enable sentence overlap

### 3. Graph-Based Merging

```python
merged = merger.merge_by_graph(
    documents,
    threshold=0.7,
    merge_communities=True,
    min_community_size=2,
    edge_weight_method='combined'
)
```

Parameters:
- `threshold`: Edge creation threshold
- `merge_communities`: Enable community detection
- `min_community_size`: Minimum community size
- `edge_weight_method`: 'cosine', 'jaccard', or 'combined'

### 4. Topic-Based Merging

```python
merged = merger.merge_by_topic(
    documents,
    num_topics=5
)
```

Parameters:
- `num_topics`: Number of topics to extract

## Implementation Details

### Lazy Loading System

```python
def lazy_import(library):
    """Lazily import heavy libraries"""
    supports = {
        'sentence_transformers': (SentenceTransformer, util),
        'nltk': nltk,
        'spacy': spacy,
        'sklearn': (AgglomerativeClustering, NMF, TfidfVectorizer),
        'networkx': nx
    }
```

### Adaptive Threshold Calculation

```python
threshold = merger._calculate_adaptive_threshold(
    documents,
    embeddings
)
```

Features:
- Document length analysis
- Similarity distribution
- Dynamic adjustment

### Community Detection

```python
communities = merger._detect_communities(
    graph,
    min_size=2
)
```

Methods:
- Louvain
- Label Propagation
- Fluid Communities
- Fallback to connected components

## Text Processing Features

### Document Pair Merging

```python
merged = merger._merge_pair(doc1, doc2)
```

Process:
1. Entity extraction
2. Sentence similarity analysis
3. Optimal arrangement
4. Coherent combination

### Group Text Merging

```python
merged = merger._merge_text_group(texts)
```

Features:
- Iterative merging
- Coherence maintenance
- Entity preservation

## Example Usage

### Basic Document Merging

```python
merger = SRSWTIPureFlow()

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning enables advanced data analysis.",
    "AI technologies transform various industries."
]

# Merge using different strategies
similarity_result = merger.process(
    documents,
    method='similarity',
    threshold=0.5
)

graph_result = merger.process(
    documents,
    method='graph',
    threshold=0.7
)
```

### Topic-Based Organization

```python
# Merge and organize by topics
topic_result = merger.process(
    documents,
    method='topic',
    num_topics=5
)

# Access topic-specific content
for topic, content in topic_result.items():
    print(f"Topic: {topic}")
    print(f"Content: {content}\n")
```

## Performance Optimization

### Memory Management
- Lazy loading of heavy libraries
- Efficient embedding caching
- Optimized graph operations

### Processing Efficiency
- Adaptive thresholding
- Batch processing support
- Intelligent community detection

## Error Handling

The implementation includes error handling for:
- Library imports
- Model loading
- Processing failures
- Graph operations

## Dependencies

Required packages:
- sentence-transformers
- spacy
- nltk
- sklearn
- networkx
- numpy

## Best Practices

1. Document Preparation
   - Clean text input
   - Consistent formatting
   - Appropriate grouping

2. Strategy Selection
   - Similarity for semantic relation
   - Sequential for order importance
   - Graph for complex relationships
   - Topic for thematic organization

3. Performance Optimization
   - Use appropriate thresholds
   - Consider document sizes
   - Monitor memory usage

## Initialization Options

```python
merger = SRSWTIPureFlow(
    embedding_model='all-MiniLM-L6-v2',  # Model for embeddings
    language='en',                       # Processing language
    spacy_model='en_core_web_sm'        # SpaCy model
)
```

## Method Selection Guide

1. Use 'similarity' when:
   - Semantic relationships matter
   - Documents have clear similarities
   - Clustering is beneficial

2. Use 'sequential' when:
   - Order is important
   - Documents flow naturally
   - Overlap is needed

3. Use 'graph' when:
   - Complex relationships exist
   - Community structure matters
   - Multiple connections present

4. Use 'topic' when:
   - Thematic organization needed
   - Multiple subjects present
   - Category-based output desired