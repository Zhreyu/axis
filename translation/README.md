# Translation
a multilingual translation system with support for multiple languages, device optimization, and detailed translation metadata.

## Quick Start

```python
from srswti_axis import SRSWTIMultilingualTranslator

# Initialize translator
translator = SRSWTIMultilingualTranslator()

# Translate text
result = translator.translate_text(
    text="Your text here",
    src_lang="English",
    tgt_lang="French"
)
```

## Supported Languages

```python
SRSWTI_LANGUAGE_CODES = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'Dutch': 'nl',
    'Polish': 'pl',
    'Turkish': 'tr'
}
```

## Initialization Options

```python
translator = SRSWTIMultilingualTranslator(
    device=None,  # Auto-selects cuda/mps/cpu
    config=None   # Optional configuration
)
```

Device selection:
- CUDA if available
- MPS (Metal Performance Shaders) if available
- CPU as fallback

## Translation Features

### Basic Translation

```python
result = translator.translate_text(
    text="Text to translate",
    src_lang="English",
    tgt_lang="French"
)
```

Returns:
```python
{
    'translation': str,
    'metadata': {
        'source_language': str,
        'target_language': str,
        'processing_time': float,
        'model': str,
        'device': str,
        'timestamp': str
    }
}
```

### Error Handling

```python
{
    'translation': None,
    'error': str  # Error description
}
```

## Result Display

```python
from srswti_axis import print_translation_results

print_translation_results(result)
```

Output format:
```
🌐 SRSWTI Multilingual Translation Results
==================================================
• Source Language:   English
• Target Language:   French
• Processing Time:   1.2345 seconds
• Device:           CUDA
• Timestamp:        2025-02-20 12:34:56
• Model:            SRSWTI-Multilingual-en-fr

Translation:
--------------------------------------------------
[Translated text here]
```

## Implementation Examples

### Basic Translation

```python
translator = SRSWTIMultilingualTranslator()

result = translator.translate_text(
    text="The rapid advancement of artificial intelligence",
    src_lang="English",
    tgt_lang="French"
)
```

### Multi-Language Pipeline

```python
translations = [
    {
        "text": "Your text here",
        "from": "English",
        "to": "French"
    },
    {
        "text": "Otro texto aquí",
        "from": "Spanish",
        "to": "English"
    }
]

for translation in translations:
    result = translator.translate_text(
        text=translation["text"],
        src_lang=translation["from"],
        tgt_lang=translation["to"]
    )
    print_translation_results(result)
```

## Technical Details

### Model Management

```python
def _get_model_name(self, src_lang: str, tgt_lang: str) -> str:
    return f"SRSWTI-Multilingual-{src_lang}-{tgt_lang}"
```

### Model Caching

```python
def _load_translation_model(self, src_code: str, tgt_code: str):
    model_key = f"{src_code}-{tgt_code}"
    if model_key not in self.translators:
        model_name = f"Helsinki-NLP/opus-mt-{src_code}-{tgt_code}"
        self.translators[model_key] = pipeline(
            "translation",
            model=model_name,
            device=self.device
        )
    return self.translators[model_key]
```

## Logging System

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] SRSWTI-Multilingual-Translator: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
```

Logging includes:
- Initialization details
- Translation errors
- Model loading events
- Processing times


## Best Practices

1. Language Selection
   - Use correct language names
   - Verify language support
   - Consider language pairs

2. Device Optimization
   - Consider batch size

3. Error Handling
   - Check return values
   - Monitor logs
   - Handle exceptions

4. Performance
   - Use model caching
   - Batch similar translations
   - Monitor processing times