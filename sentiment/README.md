# Sentiment
framework combining aspect-based analysis, domain-specific modifiers, and contextual understanding for nuanced emotion detection.

## overview
enables sophisticated emotion detection through multi-dimensional sentiment scoring and hierarchical text decomposition. super fast.

## theoretical foundations

### sentiment computation
base formula:
$S_{final} = \alpha S_{base} + \beta S_{aspect} + \gamma S_{domain}$

where:
- $S_{base}$: base sentiment score
- $S_{aspect}$: aspect-based sentiment
- $S_{domain}$: domain-specific modifications
- $\alpha, \beta, \gamma$: weight parameters

### domain modification
score adjustment:
$S_{modified} = S_{base} \prod_{w \in W} M_d(w)$

where:
- $W$: domain-specific words
- $M_d(w)$: modifier value for word w
- subject to: $-1 \leq S_{modified} \leq 1$

### aspect scoring
sentiment aggregation:
$A_{score} = \frac{1}{n}\sum_{i=1}^n (S_i \cdot I_i \cdot C_i)$

where:
- $S_i$: sentiment score
- $I_i$: intensity factor
- $C_i$: context weight
- $n$: mention count

## Quick Start

```python
from srswti_axis import SRSWTISentimentAnalyzer

# Initialize analyzer
analyzer = SRSWTISentimentAnalyzer()

# Basic analysis
results = analyzer.analyze(
    text="Your text here",
    aspects=["food", "service"],
    domain="restaurant"
)
```

## Core Components

### Data Structures

#### SRSWTIAspect
```python
@dataclass
class SRSWTIAspect:
    text: str              # Aspect text
    sentiment_score: float # Sentiment score
    context: str          # Surrounding context
    position: Tuple[int, int]  # Position in text
    modifiers: List[str]  # Modifying words
```

#### SRSWTISentiment
```python
@dataclass
class SRSWTISentiment:
    compound: float    # Combined score
    positive: float   # Positive score
    negative: float   # Negative score
    neutral: float    # Neutral score
```

### Domain-Specific Modifiers

Default domains and modifiers:
```python
domain_modifiers = {
    'product': {
        'great': 1.3,
        'defective': -1.5
    },
    'service': {
        'quick': 1.2,
        'slow': -1.2
    },
    'price': {
        'worth': 1.4,
        'expensive': -1.1
    }
}
```

### Adding Custom Modifiers

```python
# During initialization
custom_modifiers = {
    'restaurant': {
        'delicious': 1.4,
        'expensive': -1.2,
        'crowded': -0.5
    }
}
analyzer = SRSWTISentimentAnalyzer(custom_domain_modifiers=custom_modifiers)

# Dynamic addition
analyzer.add_domain_modifier(
    domain='restaurant',
    word='friendly',
    modifier=1.3
)
```

## Analysis Features

### Comprehensive Analysis

```python
results = analyzer.analyze(
    text="text to analyze",
    aspects=["aspect1", "aspect2"],
    domain="domain_name"
)
```

Returns:
```python
{
    'overall': {
        'sentiment': SRSWTISentiment,
        'text_stats': {
            'sentence_count': int,
            'word_count': int,
            'avg_sentence_length': float
        }
    },
    'sentences': [
        {
            'text': str,
            'sentiment': SRSWTISentiment,
            'intensifiers': List[Dict],
            'length': int
        }
    ],
    'aspects': {
        'aspect_name': {
            'mentions': List[SRSWTIAspect]
        }
    },
    'summary': str
}
```

### Aspect-Based Analysis

```python
aspect_analysis = analyzer._analyze_aspects(
    text="text to analyze",
    aspects=["aspect1", "aspect2"],
    domain="domain_name"
)
```

Features:
- Context window analysis
- Modifier extraction
- Domain-specific sentiment
- Position tracking

### Intensifier Detection

```python
intensifiers = analyzer._find_intensifiers(text)
```

Returns:
```python
[
    {
        'intensifier': str,
        'modified_word': str
    }
]
```

## Example Usage

### Basic Sentiment Analysis

```python
analyzer = SRSWTISentimentAnalyzer()

text = """The restaurant was incredibly crowded but the food was delicious.
The staff's friendly demeanor made the experience enjoyable."""

results = analyzer.analyze(
    text=text,
    aspects=['food', 'staff', 'atmosphere'],
    domain='restaurant'
)

print(results['summary'])
print(results['overall']['sentiment'])
```

### Custom Domain Analysis

```python
# Initialize with custom modifiers
custom_modifiers = {
    'restaurant': {
        'delicious': 1.4,
        'expensive': -1.2,
        'crowded': -0.5
    }
}

analyzer = SRSWTISentimentAnalyzer(custom_domain_modifiers=custom_modifiers)

# Add additional modifier
analyzer.add_domain_modifier('restaurant', 'friendly', 1.3)

# Analyze with custom domain
results = analyzer.analyze(
    text="The restaurant was crowded but the staff was friendly.",
    aspects=['staff', 'atmosphere'],
    domain='restaurant'
)
```

## Sentiment Scoring

### Score Ranges
- Compound: -1.0 to 1.0
- Positive: 0.0 to 1.0
- Negative: 0.0 to 1.0
- Neutral: 0.0 to 1.0

### Sentiment Levels
```python
sentiment_level = (
    'very positive'  # compound > 0.5
    'positive'      # compound > 0
    'very negative' # compound < -0.5
    'negative'      # compound < 0
    'neutral'       # compound = 0
)
```

## Best Practices

1. Text Preparation
   - Clean input text
   - Proper sentence structure
   - Consistent formatting

2. Aspect Selection
   - Choose relevant aspects
   - Use consistent naming
   - Consider domain context

3. Domain Modifiers
   - Calibrate modifier values
   - Test with sample text
   - Update based on results

4. Performance
   - Process sentences in batches
   - Cache sentiment scores
   - Monitor modifier impact

