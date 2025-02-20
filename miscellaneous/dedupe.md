

# Deduplication 

semantic deduplication 

### similarity computation
vector similarity:
no need to explain the math here, its pretty basic

filtering process:
$unique = \{d_i: sim(d_i, d_j) < threshold \,\, \forall j < i\}$

### usage
```python
deduplicator = SrswtiDeduplicator()

# Basic deduplication
unique_docs = deduplicator.deduplicate(
    documents, 
    threshold=0.5
)

# With indices
docs, indices = deduplicator.deduplicate(
    documents,
    threshold=0.5,
    return_indices=True
)
```
