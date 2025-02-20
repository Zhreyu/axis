
# DivergenceV2

Advanced semantic divergence framework combining Jensen-Shannon metrics, embeddings, and probabilistic topic analysis for high-precision content similarity measurement.

## motivation & innovation

### why we built it
traditional challenges:
- semantic blindness
- topic insensitivity
- numerical instability
- high dimensionality

our solution:
- hybrid semantic-topic analysis
- stable probability distributions
- efficient dimensionality reduction
- adaptive complexity weighting

## theoretical foundations

### enhanced jsd framework
multi-level divergence:
$D_{final} = \alpha D_{semantic} + (1-\alpha)D_{topic}$

where:
- $D_{semantic}$: semantic space divergence
- $D_{topic}$: topic space divergence
- $\alpha$: adaptive weight

#### semantic component
distribution creation:
$P(x) = \text{softmax}(\frac{sim(x, anchors)}{T})$
- T: temperature parameter
- anchors: learned semantic points

#### non matrix factorisation
nmf decomposition:
$V \approx WH$ where:
- V: tf-idf matrix
- W: document-topic matrix
- H: topic-term matrix

## algorithm details

### multi-space analysis
processing pipeline:
```python
# Semantic analysis
embeddings = encoder.encode(text)
semantic_dist = create_distribution(embeddings)

# Topic analysis
topic_dist = nmf_model.transform(tfidf_vector)

# Combined score
score = compute_weighted_divergence(
    semantic_dist,
    topic_dist
)
```

### adaptive weighting
weight computation:
$w_{semantic} = 0.6 + 0.2(1 - complexity)$

properties:
- complexity-aware
- topic-sensitive
- semantically grounded
- numerically stable, lol

## Quick Start

```python
from srswti_axis import SRSWTIDivergenceV2

# Initialize analyzer
analyzer = SRSWTIDivergenceV2(
    embedding_model='all-MiniLM-L6-v2',
    semantic_dims=128,
    semantic_temperature=0.1,
    n_topics=10,
    min_df=2
)

# Calculate divergence
score = analyzer.calculate_divergence(text1, text2)

# Get detailed metrics
metrics = analyzer.calculate_divergence(text1, text2, return_components=True)
```

## Core Components

### Initialization Parameters

```python
analyzer = SRSWTIDivergenceV2(
    embedding_model='all-MiniLM-L6-v2',  # SentenceTransformer model
    semantic_dims=128,                    # Semantic space dimensions
    semantic_temperature=0.1,             # Distribution temperature
    n_topics=10,                         # Number of NMF topics
    min_df=2                             # Minimum document frequency
)
```

### Divergence Calculation Components

The system calculates divergence using multiple metrics:

```python
metrics = analyzer.calculate_divergence(text1, text2, return_components=True)
```

Returns:
```python
{
    'divergence_score': float,     # Final divergence score
    'cosine_similarity': float,    # Embedding similarity
    'semantic_jsd': float,         # Semantic distribution JSD
    'topic_jsd': float,           # Topic distribution JSD
    'entropy_p': float,           # First text entropy
    'entropy_q': float,           # Second text entropy
    'text_complexity_1': float,   # First text complexity
    'text_complexity_2': float,   # Second text complexity
    'semantic_weight': float,     # Semantic component weight
    'topic_weight': float         # Topic component weight
}
```

## Topic Modeling Features

### Topic Extraction

```python
# Get top words for each topic
topics = analyzer.get_topic_words(top_n=10)

# Print topics
for idx, topic_words in enumerate(topics):
    print(f"Topic {idx + 1}: {', '.join(topic_words)}")
```

### Topic Distribution

```python
# Get topic distribution for a text
topic_dist = analyzer._get_topic_distribution(text)
```

## Document Processing

### Multiple Document Analysis

```python
results = analyzer.process(
    documents=list_of_documents,
    reference_doc=optional_reference,
    threshold=0.5
)
```

Returns:
```python
{
    'scores': List[float],         # Divergence scores
    'similar_texts': List[str],    # Texts below threshold
    'divergent_texts': List[str]   # Texts above threshold
}
```

## Implementation Details

### NMF Initialization

```python
analyzer._initialize_nmf(texts)
```

Features:
- TF-IDF vectorization
- Non-negative Matrix Factorization
- Topic distribution computation

Parameters:
- max_features: 1000
- min_df: Configurable
- stop_words: 'english'
- n_components: Configurable

### Semantic Distribution Creation

```python
dist, complexity = analyzer._create_semantic_distribution(text)
```

Process:
1. Sentence tokenization
2. Embedding computation
3. Semantic anchor projection
4. Distribution normalization
5. Complexity calculation

### Jensen-Shannon Divergence

```python
jsd = analyzer._improved_jensen_shannon(dist1, dist2)
```

Features:
- Numerical stability
- Proper normalization
- Safe KL divergence
- Bounded output [0,1]

## Example Usage

### Basic Text Comparison

```python
analyzer = SRSWTIDivergenceV2(n_topics=5)

text1 = """
Recent advances in deep learning have revolutionized the field of computer vision.
Convolutional Neural Networks (CNNs) have demonstrated remarkable performance in
image classification, object detection, and semantic segmentation tasks.
"""

text2 = """
The evolution of neural networks has transformed visual computing paradigms.
Deep learning approaches, particularly Convolutional Neural Networks, have
achieved unprecedented accuracy in computer vision tasks.
"""

score = analyzer.calculate_divergence(text1, text2)
print(f"Divergence score: {score}")
```

### Topic Analysis

```python
# Initialize with documents
all_texts = [text1, text2, text3]
analyzer._initialize_nmf(all_texts)

# Get topic information
topics = analyzer.get_topic_words(top_n=5)
for idx, words in enumerate(topics):
    print(f"Topic {idx + 1}: {', '.join(words)}")

# Get topic distribution
dist = analyzer._get_topic_distribution(text1)
```

## Dependencies

Required packages:
- numpy
- scipy
- sentence-transformers
- nltk
- sklearn
- torch

NLTK Resources:
- stopwords
- punkt

## Best Practices

1. Text Preparation
   - Clean input text
   - Proper sentence structure
   - Sufficient document length

2. Topic Modeling
   - Adjust n_topics based on corpus
   - Consider min_df for vocabulary
   - Review topic coherence

3. Performance
   - Batch similar length texts
   - Monitor memory usage
   - Cache embeddings when possible

## Error Handling

The implementation includes error handling for:
- Model loading failures
- NMF initialization issues
- Empty text inputs
- Numerical computation errors