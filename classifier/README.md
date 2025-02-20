
# Zero-Shot Classifier

A powerful classification system enabling label-free categorization through entailment modeling and probability calibration.



## Quick Start

```python
from srswti_axis import SRSWTIZeroShot

classifier = SRSWTIZeroShot()

# Single-label classification
result = classifier.classify_text(
    "the new ai model is impressive but has high compute needs",
    ["tech", "science", "economics"],
    multi_label=False
)
```

## Single-Label Classification

Single-label classification forces the model to choose one best-fitting label for the input text.

### Example: E-commerce Product Reviews (Single-Label)

```python
from srswti_axis import SRSWTIZeroShot

classifier = SRSWTIZeroShot()

# Single product review classification
text = "The noise-cancelling headphones exceeded my expectations. Crystal clear audio and incredible battery life, but the price point is a bit steep for budget-conscious consumers."
labels = ["positive", "neutral", "negative"]

result = classifier.classify_text(
    text=text,
    candidate_labels=labels,
    multi_label=False
)

# Batch processing with single labels
tasks = [{
    "name": "Product Reviews",
    "texts": [
        "The noise-cancelling headphones exceeded my expectations...",
        "Purchased a smart home security system that completely failed...",
        "This ergonomic office chair is a game-changer..."
    ],
    "labels": ["positive", "neutral", "negative"],
    "multi_label": False
}]

results = classifier.process_tasks(tasks)
```

## Multi-Label Classification

Multi-label classification allows each text to be associated with multiple relevant labels independently.

### Example: Technology News (Multi-Label)

```python
# Multi-label classification example
tasks = [{
    "name": "Global Technology & Innovation News",
    "texts": [
        "OpenAI's latest language model demonstrates unprecedented natural language understanding, raising both excitement and ethical concerns about AI's potential societal impact.",
        "Breakthrough in quantum computing: researchers at MIT develop a stable 1000-qubit processor that could revolutionize cryptography and scientific simulations.",
        "Climate tech startup secures $250 million in funding to develop carbon capture technology that promises to remove atmospheric CO2 at industrial scales."
    ],
    "labels": [
        "artificial_intelligence",
        "quantum_computing",
        "climate_tech",
        "economic_impact"
    ],
    "multi_label": True
}]

# Initialize classifier with batch processing
classifier = SRSWTIZeroShot(batch_size=8)
results = classifier.process_tasks(tasks)
```

### Example: Enterprise Software Support (Multi-Label)

```python
tasks = [{
    "name": "Enterprise Software Support",
    "texts": [
        "Critical security vulnerability discovered in our enterprise resource planning system. Immediate patch required to prevent potential data breaches.",
        "Performance bottleneck in our cloud-native microservices architecture is causing intermittent system-wide latency issues during peak load times."
    ],
    "labels": [
        "security_vulnerability",
        "performance_issue",
        "infrastructure",
        "architectural_challenge"
    ],
    "multi_label": True
}]

results = classifier.process_tasks(tasks)
```

## Advanced Configuration

### Classifier Initialization Options

```python
classifier = SRSWTIZeroShot(
    device=None,        # Auto-selects "rocm"/cuda/mps/cpu
    batch_size=8,       # Number of texts to process simultaneously
    model_name="SRSWTI-ZeroShot-v1"
)
```

### Batch Processing Parameters

```python
def process_tasks(
    self,
    tasks: List[Dict[str, Union[str, List[str], bool]]]
) -> Dict:
    """
    Process multiple classification tasks in batch.
    
    Parameters:
        tasks: List of task dictionaries with structure:
            {
                "name": str,          # Task identifier
                "texts": List[str],   # Texts to classify
                "labels": List[str],  # Candidate labels
                "multi_label": bool   # Enable multi-label classification
            }
    
    Returns:
        Dictionary containing classification results for each task
    """
```

## Performance Considerations

### Memory Usage
- With batch_size=8: Approximately 2GB VRAM
- CPU mode: Memory scales with batch_size

### Optimization Tips
```python
# For GPU processing
classifier = SRSWTIZeroShot(batch_size=16, device="cuda")

# For CPU processing
classifier = SRSWTIZeroShot(batch_size=4, device="cpu")
```

## Limitations

- Maximum text length: 512 tokens (longer texts should be chunked)
- Batch size affects memory usage
- Processing speed depends on hardware and batch size

## Future Developments

- Enhanced custom models
- Free API interface with rate limits
- Expanded logging capabilities
- Multilingual support with custom tokenizers

## Best Practices

1. Choose appropriate batch sizes based on available hardware
2. Use multi-label classification when texts might belong to multiple categories
3. Select distinct and well-defined labels for better classification results
4. Pre-process long texts into manageable chunks
5. Consider using GPU acceleration for large-scale processing