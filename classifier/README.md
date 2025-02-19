
# Zero-Shot Classifier

classification system enabling label-free categorization through entailment modeling and probability calibration.

## Quick Start

```python
from srswti_axis import SRSWTIZeroShot

classifier = SRSWTIZeroShot()

result = classifier.classify_text(
    "the new ai model is impressive but has high compute needs",
    ["tech", "science", "economics"])
```

## Core Concepts

We use a clever transformer-based approach that turns classification into an entailment problem. It's like asking "is this text about X?" instead of forcing it into predefined boxes.

## Probability Transformations

### Multi-Label Scenario

When you want multiple labels (because why not?), we use:



This lets each label decide for itself. No fighting between labels, just pure independence.

### Single-Label Classification

When you need that one perfect label:



Forces labels to compete. May the best label win.

## API

### Classifier Initialization

```python
classifier = SRSWTIZeroShot(
    device=None,        # auto-picks "rocm"/cuda/mps/cpu
    batch_size=8,       # process multiple texts at once
    model_name="SRSWTI-ZeroShot-v1")
```

### Classification Methods

### Single Text Classification

```python
result = classifier.classify_text(
    text="impressive performance but high costs",
    candidate_labels=["positive", "negative", "neutral"],
    multi_label=False  # set True if you want multiple labels
)
```

text (str): Input text to classify

candidate_labels (List[str]): Possible classification labels

multi_label (bool): Enable multi-label classification

### Batch Processing

```python
tasks = [{
    "name": "product reviews",
    "texts": ["amazing product!", "terrible experience"],
    "labels": ["positive", "negative", "neutral"],
    "multi_label": False
}]
classifier.process_tasks(tasks)

def process_tasks(
    self,
    tasks: List[Dict[str, Union[str, List[str], bool]]])

Parameters:
tasks: List of task dictionaries with structure:
{
    "name": str,          # Task identifier
    "texts": List[str],   # Texts to classify
    "labels": List[str],  # Candidate labels
    "multi_label": bool   # Multi-label flag
}
```

## Features

- Zero-shot: no training needed
- Batch processing: because speed matters
- Multi-label support: why choose one?

## Under the Hood

Go to sauce page for the math behind it

## Practical Stuff

### Memory Usage

- batch_size=8: ~2GB VRAM
- CPU mode: scales with batch_size

### Speed Tips

```python
classifier = SRSWTIZeroShot(batch_size=16, device="cuda")
classifier = SRSWTIZeroShot(batch_size=4, device="cpu")
```

## Limits

- Texts: max 512 tokens, chunk it and process them

-- still faster than LLMs digesting 50000 tokens as chunks at once

- No other limits.

## Coming Soon

- Better custom models
- API interface, for free. With rate limits and batch_size=1
- More logging options
- Custom tokenizers, multilingual ones too.

## Notes

That's it. Let the classifier do its thing.