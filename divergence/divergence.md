
# Divergence

semantic analysis system combining enhanced Jensen-Shannon divergence with topic modeling and neural embeddings for content similarity/divergence measurement.
## theoretical foundations

### traditional jsd
the jensen-shannon divergence (jsd) is a method of measuring the similarity between two probability distributions. it is a symmetric and smoothed version of the kullback-leibler divergence, which is not symmetric and can be undefined if one distribution assigns zero probability to an event that the other distribution considers possible. jsd is bounded between 0 and 1, where 0 indicates identical distributions and 1 indicates maximum divergence. the classical formulation of jsd is given by:
$jsd(p||q) = \frac{1}{2}d_{kl}(p||m) + \frac{1}{2}d_{kl}(q||m)$
where m is the average of the two distributions, $m = \frac{1}{2}(p + q)$. this formulation ensures that jsd is always defined and provides a meaningful measure of divergence even when the distributions have zero probabilities.
where M is mixture:
$M = \frac{1}{2}(P + Q)$

properties:
- symmetric metric
- bounded [0,1]
- square root of jsd is metric
- handles zero probabilities

### our enhanced approach

#### semantic space mapping
text to distribution:
```
1. embed text → R^d
2. project to semantic anchors
3. create probability distribution
4. stabilize numerically
```

enhanced formula:
$JSD_{enhanced} = \sqrt{\min(1.0, \max(0.0, JSD_{raw}))}$

## Quick Start

```python
from srswti_axis import SRSWTIDivergence

# Initialize analyzer
analyzer = SRSWTIDivergence(
    embedding_model='all-MiniLM-L6-v2',
    semantic_dims=128,
    semantic_temperature=0.1,
    projection_seed=42
)

# Calculate divergence
score = analyzer.calculate_divergence(text1, text2)

# Get detailed metrics
metrics = analyzer.calculate_divergence(text1, text2, return_components=True)
```

## Core Components

### Initialization Parameters

```python
analyzer = SRSWTIDivergence(
    embedding_model='all-MiniLM-L6-v2',  # SentenceTransformer model
    semantic_dims=128,                    # Fixed dimensionality
    semantic_temperature=0.1,             # Distribution sharpness
    projection_seed=42                    # Random projection seed
)
```

### Divergence Calculation

The system calculates divergence using multiple components:
- Cosine similarity
- Jensen-Shannon divergence
- Semantic complexity
- Text entropy

```python
metrics = analyzer.calculate_divergence(text1, text2, return_components=True)
```

Returns:
```python
{
    'divergence_score': float,          # Final divergence score
    'cosine_similarity': float,         # Direct embedding similarity
    'jensen_shannon_divergence': float, # Distribution divergence
    'entropy_p': float,                # First text entropy
    'entropy_q': float,                # Second text entropy
    'text_complexity_1': float,        # First text complexity
    'text_complexity_2': float,        # Second text complexity
    'cosine_weight': float,            # Dynamic weight for cosine
    'jsd_weight': float                # Dynamic weight for JSD
}
```

## Text Comparison Features

### Single Text Comparison

```python
# Basic comparison
score = analyzer.calculate_divergence(text1, text2)

# Detailed comparison
details = analyzer.calculate_divergence(text1, text2, return_components=True)
```

### Multiple Text Comparison

```python
results = analyzer.compare_texts(
    texts=list_of_texts,
    reference_text=optional_reference,
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

### Semantic Distribution Creation

```python
dist, complexity = analyzer._create_semantic_distribution(text)
```

Process:
1. Sentence tokenization
2. Embedding computation
3. Distribution creation
4. Complexity calculation

### Jensen-Shannon Divergence

```python
jsd = analyzer._improved_jensen_shannon(dist1, dist2)
```

Features:
- Numerical stability guarantees
- Proper normalization
- Sqrt transformation
- Bounded [0,1] output

### Semantic Complexity

```python
complexity = analyzer._calculate_semantic_complexity(
    distribution=dist,
    word_coherence=coherence_score
)
```

Components:
- Distribution entropy
- Word-level coherence
- Normalized scoring

## Example Usage

### Basic Text Comparison

```python
analyzer = SRSWTIDivergence()

text1 = "Cats are adorable domestic pets. They have soft fur and independent personalities."
text2 = "Felines are charming household companions. These animals possess silky coats and autonomous behaviors."

score = analyzer.calculate_divergence(text1, text2)
print(f"Divergence score: {score}")
```

### Detailed Analysis

```python
metrics = analyzer.calculate_divergence(text1, text2, return_components=True)

print("Analysis components:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
```

### Batch Processing

```python
texts = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning utilizes complex neural networks.",
    "Mechanical engineering involves physical machines."
]

results = analyzer.compare_texts(
    texts=texts,
    threshold=0.5
)

print("Similar texts:", len(results['similar_texts']))
print("Divergent texts:", len(results['divergent_texts']))
```

## Performance Considerations

### Memory Usage
- Lazy loading of sentence transformer
- Efficient projection matrix computation
- Distribution caching when possible

### Numerical Stability
- Safe softmax implementation
- Epsilon handling in JSD
- Proper normalization steps

## Error Handling

The implementation includes error handling for:
- Model loading failures
- Empty text inputs
- Numerical computation issues
- Batch processing errors



## Best Practices

1. Text Preparation
   - Clean input text
   - Proper sentence structure
   - Consistent formatting

2. Threshold Selection
   - Default: 0.5
   - Adjust based on use case
   - Consider text complexity

3. Performance Optimization
   - Batch similar length texts
   - Reuse reference comparisons
   - Monitor memory usage