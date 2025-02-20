
# Chunk Analysis

syntactic decomposition through hybrid CFG-NLG architecture.

## theoretical foundations

### context-free grammar framework
base grammar structure:
```
NP -> (DT|PRP$)? JJ* NN+
VP -> MD? VB* (NP|PP)*
PP -> IN NP
ADJP -> RB? JJ
```

properties:
- recursive patterns
- hierarchical structure
- compositional rules
- phrase coherence


## Quick Start

```python
from srswti_axis import SRSWTIChunkAnalyzer

# Initialize analyzer
analyzer = SRSWTIChunkAnalyzer()

# Analyze text
results = analyzer.analyze_text(
    text="Your text here",
    use_rich=False  # Optional rich visualization
)
```

## Core Components

### Data Structures

#### SRSWTIChunk
```python
@dataclass
class SRSWTIChunk:
    text: str              # Chunk text
    type: str             # Chunk type
    level: int           # Hierarchical level
    start: int           # Start position
    end: int             # End position
    sub_chunks: List     # Nested chunks
    grammatical_role: str # Optional role
```

#### SRSWTIPhrase
```python
@dataclass
class SRSWTIPhrase:
    text: str            # Phrase text
    type: str           # Phrase type
    head_word: str      # Main word
    modifiers: List[str] # Modifying words
    dependencies: List[str] # Dependencies
```

### Grammar Patterns

The system supports sophisticated grammar patterns:

```python
grammar = r"""
    # Noun phrase patterns
    NP:
        {<DT|PRP\$>?<JJ.*>*<NN.*>+}  # Basic NP
        {<NNP>+}                      # Proper nouns
        {<PRP>}                       # Pronouns
        
    # Verb phrase patterns
    VP:
        {<MD>?<VB.*><NP|PP>*}        # Verb with optional object
        {<VBG><NP>}                   # Gerund phrases
        
    # Prepositional phrase
    PP:
        {<IN><NP>}                    # Basic PP
        {<TO><NP>}                    # To-phrases
        
    # Adjective phrase
    ADJP:
        {<RB.*>?<JJ.*>}              # Adjectives with optional adverbs
        {<JJR><IN>}                   # Comparative
        
    # Adverb phrase
    ADVP:
        {<RB.*>+}                     # Multiple adverbs
        
    # Clause patterns
    CLAUSE:
        {<NP><VP>}                    # Basic clause
"""
```

## Analysis Features

### Text Analysis

```python
results = analyzer.analyze_text(text)
```

Returns:
```python
{
    'overall_stats': {
        'sentence_count': int,
        'total_chunks': int,
        'chunk_distribution': Dict[str, int]
    },
    'sentence_analysis': List[Dict],
    'phrase_patterns': Dict[str, List],
    'hierarchical_structure': List[Dict],
    'tree_visualizations': List[Dict],
    'summary': str
}
```

### Visualization Options

#### ASCII Tree Visualization
```python
tree_str = analyzer.visualize_tree(
    tree,
    indent='',
    last=True
)
```

#### Rich Visualization (Optional)
```python
results = analyzer.analyze_text(
    text=text,
    use_rich=True  # Enables colorful tree display
)
```

### POS Tag Reference

Access POS tag descriptions:
```python
pos_legend = SRSWTIChunkAnalyzer.get_pos_tag_legend()
```

Includes categories:
- Determiners and Pronouns (DT, PRP, PRP$)
- Nouns (NN, NNS, NNP, NNPS)
- Verbs (VB, VBD, VBG, VBN, VBP, VBZ, MD)
- Adjectives (JJ, JJR, JJS)
- Adverbs (RB, RBR, RBS)
- Prepositions and Conjunctions (IN, TO)

## Implementation Examples

### Basic Analysis

```python
analyzer = SRSWTIChunkAnalyzer()

text = """
The experienced data scientist quickly analyzed the complex dataset.
She discovered several interesting patterns in the neural network's behavior.
"""

results = analyzer.analyze_text(text)
print(json.dumps(results, indent=2))
```

### Grammar Pattern Analysis

```python
# Print grammar pattern details
analyzer.print_grammar_legend()

# Access pattern information programmatically
patterns = analyzer.get_grammar_pattern_legend()
for pattern_name, description in patterns.items():
    print(f"{pattern_name}:\n{description}\n")
```

### Named Entity Analysis

```python
for sentence_analysis in results['sentence_analysis']:
    named_entities = sentence_analysis['named_entities']
    for entity in named_entities:
        print(f"Entity: {entity['text']}")
        print(f"Type: {entity['type']}")
        print(f"Confidence: {entity['confidence']}")
```

## Advanced Features

### Grammatical Role Detection

```python
def _determine_grammatical_role(tree):
    """Determines roles: subject, object, predicate, modifier"""
    if label == 'NP':
        if parent and parent.label() == 'S':
            return 'subject'
        elif parent and parent.label() == 'VP':
            return 'object'
    elif label == 'VP':
        return 'predicate'
    elif label == 'PP':
        return 'modifier'
```

### Hierarchical Analysis

```python
hierarchical = results['hierarchical_structure']
for structure in hierarchical:
    print(f"Sentence: {structure['sentence']}")
    print(f"Tree: {structure['tree']}")
```

## Best Practices

1. Text Preparation
   - Clean input text
   - Proper sentence structure
   - Consistent formatting

2. Analysis Selection
   - Use basic analysis for simple tasks
   - Enable rich visualization for debugging
   - Access specific analysis components as needed

3. Pattern Usage
   - Reference grammar patterns documentation
   - Understand POS tag meanings
   - Consider hierarchical structure
