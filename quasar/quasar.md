# Quasar

A state-of-the-art topic modeling framework that combines embeddings, matrix factorization, and hierarchical clustering for sophisticated topic discovery.

## Quick Start

```python
from srswti_axis import SRSWTIQuasar

# Initialize the model
topic_model = SRSWTIQuasar(
    backend='srswti_bi_encoder',
    language='english',
    embedding_model='all-MiniLM-L6-v2'
)

# Fit and transform documents
results = topic_model.fit_transform(documents, num_topics=4)
```

## Available Backends

The framework supports multiple topic modeling approaches:

1. `srswti_bi_encoder`: Transformer-based embedding approach
2. `srswti_lsa`: Latent Semantic Analysis
3. `srswti_nmatrixfact`: Non-negative Matrix Factorization
4. `srswti_latent`: Probabilistic Latent Semantic Analysis
5. `srswti_simple`: Keyword-based approach using KeyBERT

## Detailed Usage Examples

### Technical Document Analysis

```python
# Sample technical documents
documents = [
    "Python programming is essential for data science and machine learning applications",
    "Deep learning models use neural networks for complex pattern recognition",
    "Cloud computing enables scalable infrastructure for big data processing",
    "Cybersecurity protects systems from malicious attacks and data breaches",
    "Artificial intelligence is transforming automation and decision making"
]

# Initialize model
topic_model = SRSWTIQuasar(
    backend='srswti_bi_encoder',
    language='english',
    embedding_model='all-MiniLM-L6-v2',
    random_state=42
)

# Fit and transform
results = topic_model.fit_transform(documents, num_topics=2)

# Access results
topics = results['topic_assignments']
probabilities = results['topic_probabilities']
```

### Healthcare Document Analysis

```python
healthcare_docs = [
    "Medical research advances treatment options for chronic diseases",
    "Preventive healthcare focuses on maintaining wellness through lifestyle",
    "Telemedicine provides remote access to healthcare services globally",
    "Vaccination programs prevent the spread of infectious diseases",
    "Mental health awareness promotes psychological well-being"
]

# Using LSA backend
topic_model = SRSWTIQuasar(backend='srswti_lsa')
results = topic_model.fit_transform(healthcare_docs, num_topics=3)

# Get topic words
topic_words = topic_model.get_topic_words()
```

## Result Analysis

### Displaying Results

```python
def display_topic_results(results, documents):
    topics = results['topic_assignments']
    probs = results['topic_probabilities']
    
    # Show topic distribution
    unique_topics = sorted(set(topics))
    for topic in unique_topics:
        docs_in_topic = [i for i, t in enumerate(topics) if t == topic]
        print(f"\nTopic {topic}: {len(docs_in_topic)} documents")
        for doc_idx in docs_in_topic:
            print(f"Doc {doc_idx}: {documents[doc_idx][:100]}...")
    
    # Show probabilities
    for idx, prob in enumerate(probs):
        print(f"\nDocument {idx} probabilities:")
        for topic_idx, topic_prob in enumerate(prob):
            print(f"Topic {topic_idx}: {topic_prob:.4f}")
```

### Matrix Factorization Results

```python
def analyze_nmf_results(results, topic_model):
    doc_topic_matrix = results['document_topic_matrix']
    
    # Get reconstruction error
    print(f"Reconstruction Error: {topic_model.model.reconstruction_err_:.4f}")
    
    # Display topics
    all_topics = topic_model.get_topic_words()
    for topic_id, words in all_topics.items():
        print(f"Topic {topic_id}: {', '.join(words)}")
```

## Common Use Cases

### Content Organization
- Document clustering for large collections
- Content recommendation systems
- Trend analysis in document streams
- Theme discovery in text corpora

### Research Analysis
- Academic paper categorization
- Research field mapping
- Citation network analysis
- Concept discovery and tracking

### Business Applications
- Customer feedback analysis
- Market research document processing
- Product review categorization
- News article classification

## Performance Considerations

### Processing Speeds
- Document embedding: ~100ms per document
- Topic clustering: ~200ms per batch
- Topic extraction: ~50ms per document

### Memory Usage
- Scales with document collection size
- Embedding model size impacts memory usage
- Batch processing recommended for large datasets

## Best Practices

1. Document Preparation
   - Clean and preprocess text data
   - Remove irrelevant content
   - Standardize document format

2. Topic Number Selection
   - Start with domain knowledge
   - Use coherence scores for optimization
   - Consider document collection size

3. Backend Selection
   - srswti_bi_encoder: Best for semantic understanding
   - srswti_lsa: Good for large document collections
   - srswti_nmatrixfact: Better interpretability
   - srswti_latent: Probabilistic approach
   - srswti_simple: Quick keyword extraction

## Future Developments

- Streaming topic modeling support
- Hierarchical topic structures
- Dynamic topic updating
- Cross-lingual topic modeling
- Interactive topic visualization

## Limitations

- Document length impacts performance
- Language model dependencies
- Computation resources for large collections
- Topic number selection sensitivity