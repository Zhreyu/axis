
# Summarization

super-fast text summarization framework combining multiple transformer architectures and adaptive chunk processing for sophisticated document summarization.
# SRSWTI Summarizer

A text summarization system with multiple model options and efficient batch processing capabilities.

## Quick Start

```python
from srswti_axis import SRSWTISummarizer

# Initialize summarizer
summarizer = SRSWTISummarizer(
    device=None,  # Defaults to GPU if available
    batch_size=8,
    use_fp16=False
)

# Single text summarization
summary = summarizer.summarize_text(
    text="Your text here",
    model_key="SRSWTI-LW2",
    min_length=30,
    max_length=200
)

# Batch summarization
summaries = summarizer.summarize_batch(
    texts=["Text 1", "Text 2"],
    model_key="SRSWTI-LW2",
    min_length=30,
    max_length=200
)
```

## Available Models

### Lightweight Models (less than 200MB)

1. `SRSWTI-LW1`
   - Size: 60MB
   - Best for: Quick inference, mobile, low-resource environments

2. `SRSWTI-LW2`
   - Size: 150MB
   - Best for: Production, fast processing

3. `SRSWTI-LW3`
   - Size: 180MB
   - Best for: Headlines, short summaries

### Medium Models (200MB-500MB)

1. `SRSWTI-MD1`
   - Trained on: t5-base
   - Size: 220MB
   - Best for: General-purpose, balanced performance

2. `SRSWTI-MD2`
   - Trained on : facebook/bart-large-cnn
   - Size: 400MB
   - Best for: News, professional content

3. `SRSWTI-MD3`
   - Trained on : google/pegasus-cnn_dailymail
   - Size: 450MB
   - Best for: News articles, balanced summarization

### Heavy Models (1GB+)

1. `SRSWTI-HV3`
   - Base Model: google/pegasus-large
   - Size: 2.2GB
   - Best for: Highest-quality summaries, advanced NLP tasks

## Core Features

### Initialization Parameters

```python
summarizer = SRSWTISummarizer(
    device=None,      # Device ID (None = auto-select)
    batch_size=8,     # Batch processing size
    use_fp16=False    # Half-precision inference on GPU
)
```

### Single Text Summarization

```python
summary = summarizer.summarize_text(
    text="Your text here",
    model_key="SRSWTI-LW2",  # Model selection
    min_length=30,           # Minimum summary length
    max_length=200          # Maximum summary length
)
```

### Batch Processing

```python
summaries = summarizer.summarize_batch(
    texts=["Text 1", "Text 2"],
    model_key="SRSWTI-LW2",
    min_length=30,
    max_length=200
)
```

## Implementation Details

### GPU Optimization
```python
# Automatic GPU detection
device = 0 if torch.cuda.is_available() else -1

# cuDNN benchmark for potential speedup
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
```

### Memory Management
```python
# Multiprocessing strategy for semaphore handling
torch.multiprocessing.set_sharing_strategy('file_system')
```

### Logging and Metrics

```python
# Built-in logging
summarizer.log("Custom message", level="INFO")

# Timing metrics
summarizer.timing("Operation name", duration)
```

## Example Usage

### Basic Summarization
```python
summarizer = SRSWTISummarizer()

text = """
The evolution of artificial intelligence and machine learning has transformed 
numerous industries, reshaping how businesses operate and how we interact 
with technology.
"""

summary = summarizer.summarize_text(
    text=text,
    model_key="SRSWTI-LW2"
)
print(summary)
```

### Batch Processing
```python
batch_texts = [
    "First article for summarization...",
    "Second article for summarization...",
    "Third article for summarization..."
]

summaries = summarizer.summarize_batch(
    texts=batch_texts,
    model_key="SRSWTI-LW2"
)

for idx, summary in enumerate(summaries, 1):
    print(f"Summary {idx}: {summary}")
```

## Performance Considerations

1. Device Selection
   - Automatic GPU detection
   - CPU fallback (-1)
   - Custom device specification

2. FP16 Inference
   - Enable with `use_fp16=True`
   - Only effective on GPU
   - Reduces memory usage

3. Batch Processing
   - Default batch size: 8
   - Adjustable based on memory
   - Efficient for multiple texts

## Error Handling

The implementation includes built-in error handling for:
- Invalid model configurations
- Device availability
- Memory management
- Model loading issues


## Best Practices

1. Model Selection
   - Use LW2 for speed
   - Use heavy models for quality
   - Consider resource constraints

2. Batch Processing
   - Group similar length texts
   - Adjust batch size as needed
   - Monitor memory usage

3. Performance Optimization
   - Enable FP16 for GPU
   - Use appropriate batch sizes
   - Monitor timing metrics