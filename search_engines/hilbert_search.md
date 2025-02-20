
# SRSWTI Hilbert Search

Learning-to-rank system implementing pointwise, pairwise, and listwise ranking approaches.

## Quick Start

```python
from srswti_axis import SRSWTIHilbertSearch

# Initialize ranker
ranker = SRSWTIHilbertSearch(approach='pointwise')

# Train the model
ranker.train(
    queries=queries,
    documents=documents,
    relevance_scores=relevance_scores,
    epochs=100
)

# Rank new documents
results = ranker.rank_documents(query, documents)
```

## Core Components

### Feature Extraction

The system uses a FeatureExtractor class that combines multiple feature types:

```python
class FeatureExtractor:
    def __init__(self, embedding_model: str = 'srswti-neural-embedder-v1'):
        # Model mapping
        model_mapping = {
            'srswti-neural-embedder-v1': 'all-mpnet-base-v2'
        }
        actual_model = model_mapping.get(embedding_model, embedding_model)
        self.embedder = SentenceTransformer(actual_model)
        self.tfidf = TfidfVectorizer()
        self.scaler = StandardScaler()
```

Features extracted include:
- TF-IDF similarity
- Semantic similarity
- Document length
- Relative document length

### Ranking Models

#### 1. Pointwise Ranker
```python
class PointwiseRanker(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
```

#### 2. Pairwise Ranker
```python
class PairwiseRanker(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
```

#### 3. Listwise Ranker
```python
class ListwiseRanker(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
```

## Training Methods

### Training Parameters
```python
def train(self, 
         queries: List[str],
         documents: List[List[str]],
         relevance_scores: List[List[float]],
         epochs: int = 100):
```

### Training Examples

```python
# Training data structure
training_queries = [
    {
        "query": "machine learning artificial intelligence",
        "relevance": [0.95, 0.3, 0.8, 0.4, 0.1, 0.1, 0.1]
    },
    {
        "query": "sports physical training",
        "relevance": [0.2, 0.1, 0.2, 0.1, 0.8, 0.9, 0.85]
    }
]

# Prepare training data
train_queries = [q["query"] for q in training_queries]
train_docs = [documents] * len(training_queries)
relevance_scores = [q["relevance"] for q in training_queries]

# Train model
ranker.train(
    queries=train_queries,
    documents=train_docs,
    relevance_scores=relevance_scores,
    epochs=50
)
```

## Document Ranking

```python
def rank_documents(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
    """
    Returns: List of tuples containing (document_index, score)
    """
```

### Evaluation Example
```python
def evaluate_ranking(ranker, query: str, documents: List[str], 
                    expected_top_k: List[int] = None) -> Dict:
    results = ranker.rank_documents(query, documents)
    
    # Get ranked document indices and scores
    ranked_indices = [idx for idx, _ in results]
    scores = [score for _, score in results]
    
    metrics = {
        "query": query,
        "top_5_docs": [documents[idx] for idx in ranked_indices[:5]],
        "top_5_scores": [scores[idx] for idx in ranked_indices[:5]]
    }
    
    if expected_top_k:
        correct = len(set(ranked_indices[:3]).intersection(set(expected_top_k)))
        metrics["precision_at_3"] = correct / 3
    
    return metrics
```

## Model Architecture Details

### Neural Network Layers
All models use a similar architecture:
- Input layer based on feature dimension
- Hidden layer with 64 units (ReLU activation)
- Dropout layer (0.2)
- Hidden layer with 32 units (ReLU activation)
- Output layer (varies by approach)

### Model-Specific Details
- Pointwise: Sigmoid output for single score
- Pairwise: Score difference with sigmoid
- Listwise: Softmax over document list

## Training Components

Each approach uses specific training components:
- Pointwise: MSE Loss
- Pairwise: BCE Loss
- Listwise: Cross-entropy on probability distributions

The optimization is handled by Adam optimizer for all approaches.