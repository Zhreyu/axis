# Common Errors and Solutions with TextAnalyzer

This document outlines common errors encountered while running


### Error
```
LookupError: 
**********************************************************************
    Resource punkt not found.
    Please use the NLTK Downloader to obtain the resource:

    >>> import nltk
    >>> nltk.download('punkt')
    
    For more information see: https://www.nltk.org/data.html
**********************************************************************
```

### Solution
NLTK requires additional data packages to function properly. To resolve this:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```


## Too Many Open Logs Issue

### Problem
Logs are flooding the console.
### Solution
1. **Configure logging levels:**

```python
import logging
logging.basicConfig(
        level=logging.WARNING,  # Change to ERROR, WARNING, INFO, DEBUG as needed
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

2. **Create a custom logger for TextAnalyzer:**

```python
def setup_logger():
        logger = logging.getLogger('textanalyzer')
        logger.setLevel(logging.INFO)
        
        # Create file handler for logs
        file_handler = logging.FileHandler('textanalyzer.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler with higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
```

3. **Reduce verbosity in model abstractions:**

4 . python -m spacy download en_core_web_sm


5. GraphFlow has maximum recursion depth issue