
# Graph Flow 

sophisticated document merging algorithm utilizing multi-level graph structures and flow-based ordering for coherent document combinations.

## overview
combines semantic analysis, spectral clustering, and centrality-based flow for optimal text organization.

## theoretical foundations

### hierarchical graph structure
three-level architecture:
```
L3: document-level connections
L2: paragraph relationships
L1: sentence coherence
```

edge weights computation:
$W_{ij} = 0.4S + 0.3T + 0.2E + 0.1K$
where:
- S: semantic similarity
- T: tf-idf overlap
- E: entity similarity
- K: keyphrase overlap

### community detection

#### spectral clustering
optimization objective:
$\min_{A_1,...,A_k} \sum_{i=1}^k \frac{cut(A_i,\bar{A_i})}{vol(A_i)}$

fallback mechanism:
1. connected components
2. spectral clustering
3. single community

### flow-based ordering

#### centrality computation
node importance:
$C(v) = \frac{PR(v) + DC(v) + EC(v)}{3}$
where:
- PR: pagerank score
- DC: degree centrality
- EC: eigenvector centrality

flow optimization:
$next = \argmax_{v \in R} (W_{last,v} + C(v))$
where:
- R: remaining nodes
- W: edge weights
- C: centrality scores

## Quick Start

```python
from srswti_axis import SRSWTIGraphFlow

# Initialize merger
merger = SRSWTIGraphFlow(
    embedding_model='srswti-neural-embedder-v1',  # Maps to 'all-MiniLM-L6-v2'
    spacy_model='en_core_web_sm'
)

# Merge documents
merged_docs = merger.merge_documents(documents)
```

## Core Components

### Edge Weight Calculation

```python
weights = merger._calculate_edge_weights(doc1, doc2, emb1, emb2)
```

Combines multiple similarity metrics:
```python
weights = {
    'semantic': float,    # Embedding similarity (0.4 weight)
    'tfidf': float,      # TF-IDF similarity (0.3 weight)
    'entity': float,     # Named entity overlap (0.2 weight)
    'keyphrase': float,  # Noun chunk overlap (0.1 weight)
    'combined': float    # Weighted combination
}
```

### Hierarchical Graph Structure

Creates a three-level graph hierarchy:

1. Document Level (Level 3)
```python
doc_graph = nx.Graph()
# Nodes: documents
# Edges: document similarities above 0.3 threshold
```

2. Paragraph Level (Level 2)
```python
para_graph = nx.Graph()
# Nodes: paragraphs with mapping to source documents
# Edges: paragraph relationships
```

3. Sentence Level (Level 1)
```python
sent_graph = nx.Graph()
# Nodes: sentences with mapping to paragraphs
# Edges: sentence relationships
```

## Community Detection

### Spectral Clustering

```python
communities = merger._detect_communities(graph, n_clusters=None)
```

Features:
- Automatic cluster number determination
- Connected component analysis
- Spectral clustering fallback
- Community refinement

### Centrality-Based Ordering

```python
ordered = merger._order_by_centrality_flow(graph, community)
```

Measures used:
- PageRank centrality
- Degree centrality
- Eigenvector centrality (for graphs < 500 nodes)

## Document Merging Process

### Main Merging Pipeline

```python
def merge_documents(self, documents: List[str]) -> List[str]:
    # 1. Build hierarchical graph
    doc_graph, para_graph, sent_graph = self._build_hierarchical_graph(documents)
    
    # 2. Detect communities
    doc_communities = self._detect_communities(doc_graph)
    
    # 3. Process each community
    for community in doc_communities:
        # 4. Order documents by centrality
        ordered_docs = self._order_by_centrality_flow(doc_graph, community)
        
        # 5. Merge with topic awareness
        merged_text = []
        current_topic = None
        
        # 6. Process paragraphs and maintain coherence
        for doc_idx in ordered_docs:
            # Process document paragraphs
            # Maintain topic continuity
            # Append coherent text
```

## Progress Tracking

The system includes rich progress tracking:

```python
with Progress(console=console) as progress:
    task1 = progress.add_task("[cyan]Encoding documents...")
    task2 = progress.add_task("[cyan]Building document graph...")
    task3 = progress.add_task("[cyan]Building paragraph graph...")
    task4 = progress.add_task("[cyan]Building sentence graph...")
```

## Logging System

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SRSWTI] %(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True, console=console),
        logging.FileHandler("srswti_graph_merge.log")
    ]
)
```

## Example Usage

### Basic Document Merging

```python
merger = SRSWTIGraphFlow()

documents = [
    """Blockchain technology is revolutionizing financial transactions.
    Decentralized ledgers provide transparent and secure record-keeping.""",
    
    """Cryptocurrency markets are evolving with advanced blockchain technologies.
    Bitcoin and Ethereum represent major innovations in digital finance."""
]

merged_docs = merger.merge_documents(documents)
```

### Processing Test Documents

```python
try:
    # Initialize merger
    merger = SRSWTIGraphFlow()
    
    # Merge documents
    merged_docs = merger.merge_documents(test_documents)
    
    # Process results
    for i, doc in enumerate(merged_docs, 1):
        print(f"Merged Document {i}:")
        print(doc)
        
except Exception as e:
    logger.error(f"Error during testing: {e}")
```

## Implementation Details

### Entity-Based Topic Detection

```python
topic_entities = {
    ent.text for ent in doc.ents 
    if ent.label_ in {'ORG', 'PRODUCT', 'GPE'}
}
```

### Graph Edge Creation

```python
if weights['combined'] > 0.3:  # Threshold for document connection
    doc_graph.add_edge(i, j, **weights)
```

### Paragraph Processing

```python
paragraphs = [p.strip() for p in doc.split('\n\n') if p.strip()]
for para in paragraphs:
    para_graph.add_node(para_idx, text=para, level='paragraph')
```

## Dependencies

Required packages:
- networkx
- numpy
- sklearn
- spacy
- sentence-transformers
- rich (for logging and progress)

## Best Practices

1. Document Preparation
   - Clean text input
   - Clear paragraph separation
   - Consistent formatting

2. Performance Optimization
   - Use appropriate thresholds
   - Monitor graph size
   - Consider centrality computation limits

3. Topic Handling
   - Maintain topic coherence
   - Track entity transitions
   - Preserve document flow