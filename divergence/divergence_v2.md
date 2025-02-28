
# DivergenceV2

Advanced semantic divergence framework combining Jensen-Shannon metrics, embeddings, and probabilistic topic analysis for high-precision content similarity measurement.

## motivation & innovation

### why we built it
traditional challenges:
- semantic blindness
- topic insensitivity
- numerical instability
- high dimensionality

our solution:
- hybrid semantic-topic analysis
- stable probability distributions
- efficient dimensionality reduction
- adaptive complexity weighting

## theoretical foundations

### enhanced jsd framework
multi-level divergence:
$D_{final} = \alpha D_{semantic} + (1-\alpha)D_{topic}$

where:
- $D_{semantic}$: semantic space divergence
- $D_{topic}$: topic space divergence
- $\alpha$: adaptive weight

#### semantic component
distribution creation:
$P(x) = \text{softmax}(\frac{sim(x, anchors)}{T})$
- T: temperature parameter
- anchors: learned semantic points

#### non matrix factorisation
nmf decomposition:
$V \approx WH$ where:
- V: tf-idf matrix
- W: document-topic matrix
- H: topic-term matrix

## algorithm details

### multi-space analysis
processing pipeline:
```python
# Semantic analysis
embeddings = encoder.encode(text)
semantic_dist = create_distribution(embeddings)

# Topic analysis
topic_dist = nmf_model.transform(tfidf_vector)

# Combined score
score = compute_weighted_divergence(
    semantic_dist,
    topic_dist
)
```

### adaptive weighting
weight computation:
$w_{semantic} = 0.6 + 0.2(1 - complexity)$

properties:
- complexity-aware
- topic-sensitive
- semantically grounded
- numerically stable, lol

## Quick Start

```python

from srswti_axis import SRSWTIDivergenceV2
import numpy as np
if __name__ == "__main__":

    test_texts = {
        'research_paper': {
            'base': """
            Recent advances in deep learning have revolutionized the field of computer vision. Convolutional Neural Networks (CNNs) 
            have demonstrated remarkable performance in image classification, object detection, and semantic segmentation tasks. 
            These architectures leverage hierarchical feature extraction, where early layers capture basic visual elements like edges 
            and textures, while deeper layers learn increasingly complex representations. Transfer learning techniques have further 
            enhanced the practical applicability of these models, allowing practitioners to achieve state-of-the-art results with 
            limited training data through fine-tuning pre-trained networks.
            """,
            'similar': """
            The evolution of neural networks has transformed visual computing paradigms. Deep learning approaches, particularly 
            the application of Convolutional Neural Networks, have achieved unprecedented accuracy in computer vision tasks. 
            These deep architectures excel at automatically learning relevant features from raw image data, with initial layers 
            detecting basic patterns and subsequent layers identifying complex visual concepts. The advent of transfer learning 
            has made these powerful models more accessible, enabling researchers to adapt pre-trained networks for specific 
            applications with minimal additional training data.
            """,
            'different': """
            Quantum computing represents a fundamental shift in computational paradigms. Unlike classical computers that operate 
            on binary bits, quantum computers utilize quantum bits or qubits, which can exist in multiple states simultaneously 
            through superposition. This property, combined with quantum entanglement, enables quantum computers to perform certain 
            calculations exponentially faster than traditional computers. Recent developments in quantum error correction and 
            hardware stability have brought us closer to achieving practical quantum supremacy.
            """
        },
        'medical_article': {
            'base': """
            The human microbiome plays a crucial role in maintaining overall health and disease prevention. The diverse community 
            of microorganisms inhabiting the human gut influences everything from metabolism to immune system function. Recent 
            research has revealed strong connections between gut microbiota composition and various health conditions, including 
            obesity, inflammatory bowel disease, and even mental health disorders. Understanding these complex interactions has 
            led to new therapeutic approaches, including targeted probiotics and fecal microbiota transplantation.
            """,
            'similar': """
            The complex ecosystem of microorganisms within the human digestive system has emerged as a critical factor in health 
            maintenance. Scientists have discovered that the gut microbiome's composition significantly impacts numerous 
            physiological processes, from digestive efficiency to immune response. Growing evidence suggests that alterations in 
            gut bacterial populations are linked to various medical conditions, spanning metabolic disorders to psychological 
            health. These insights have spawned innovative treatments focusing on microbiome manipulation through probiotics 
            and bacterial transplant procedures.
            """,
            'different': """
            Renewable energy technologies have made significant strides in recent years, with solar and wind power becoming 
            increasingly cost-competitive with traditional fossil fuels. Advances in photovoltaic cell efficiency and wind 
            turbine design have dramatically reduced the levelized cost of electricity generation. Energy storage solutions, 
            particularly lithium-ion batteries, have also evolved to address intermittency issues. These developments are 
            reshaping the global energy landscape and accelerating the transition to sustainable power sources.
            """
        },
        'environmental_report': {
            'base': """
            Climate change poses unprecedented challenges to global ecosystems and biodiversity. Rising global temperatures 
            have led to significant alterations in weather patterns, causing more frequent extreme weather events and 
            disrupting natural habitats. Arctic ice melt has accelerated, threatening polar ecosystems and contributing to 
            sea level rise. Meanwhile, ocean acidification is severely impacting marine life, particularly coral reefs and 
            shellfish populations. These changes are creating cascading effects throughout food webs and ecosystems worldwide.
            """,
            'similar': """
            Global warming has emerged as a critical threat to Earth's biological systems and species diversity. The steady 
            increase in average temperatures worldwide has triggered substantial changes in climate patterns, resulting in 
            more severe weather phenomena and habitat destruction. Polar regions are experiencing rapid ice loss, endangering 
            local wildlife and coastal communities through rising sea levels. The increasing acidity of ocean waters poses a 
            grave danger to marine ecosystems, especially affecting coral communities and calcifying organisms.
            """,
            'different': """
            The development of artificial general intelligence (AGI) presents both unprecedented opportunities and challenges 
            for humanity. Current research focuses on developing systems that can match or exceed human-level reasoning 
            across multiple domains. Key challenges include ensuring alignment with human values, maintaining controllability, 
            and addressing potential ethical concerns. The development of robust safety protocols and governance frameworks 
            will be crucial as these technologies advance.
            """
        },
        'business_analysis': {
            'base': """
            Digital transformation has fundamentally altered the business landscape, forcing companies to reimagine their 
            operational models and customer engagement strategies. E-commerce platforms have revolutionized retail, while 
            cloud computing has enabled unprecedented scalability and flexibility. Data analytics and artificial intelligence 
            are providing deeper insights into customer behavior and optimizing business processes. Companies that fail to 
            adapt to this digital revolution risk becoming obsolete in an increasingly competitive market.
            """,
            'similar': """
            The digital revolution has reshaped how businesses operate and interact with consumers. Traditional business 
            models are being disrupted as organizations embrace digital technologies and online platforms. Cloud-based 
            solutions have transformed IT infrastructure, offering scalable and cost-effective alternatives to traditional 
            systems. Advanced analytics and machine learning are enabling companies to make data-driven decisions and 
            personalize customer experiences. This technological shift is creating a new paradigm in business operations.
            """,
            'different': """
            Advances in genetic engineering, particularly CRISPR-Cas9 technology, have opened new frontiers in medical 
            treatment and biological research. This precise gene-editing tool allows scientists to modify DNA sequences 
            with unprecedented accuracy, offering potential treatments for genetic disorders and improving crop resilience. 
            However, ethical considerations and safety concerns surrounding genetic modification continue to generate 
            significant debate in the scientific community.
            """
        }
    }

    # Initialize analyzer
    analyzer = SRSWTIDivergenceV2(
        semantic_dims=128,
        semantic_temperature=0.1,
        n_topics=5,
        min_df=1
    )
    
    print("\nTesting Enhanced Semantic Divergence Analysis with NMF:")
    print("=" * 80)
    
    # First, collect all texts for NMF initialization
    all_texts = []
    for category in test_texts.values():
        all_texts.extend(category.values())
    
    # Initialize NMF with all texts
    analyzer._initialize_nmf(all_texts)
    
    # Print topic information
    print("\nDiscovered Topics:")
    print("-" * 40)
    topics = analyzer.get_topic_words(top_n=5)
    for idx, topic_words in enumerate(topics):
        print(f"Topic {idx + 1}: {', '.join(topic_words)}")
    print("=" * 80)
    
    # Test each category
    for category_name, texts in test_texts.items():
        print(f"\nTesting {category_name.upper()} category:")
        print("-" * 40)
        
        # Compare similar texts
        similar_score = analyzer.calculate_divergence(
            texts['base'], 
            texts['similar'],
            return_components=True
        )
        print(f"Similar texts divergence components:")
        for k, v in similar_score.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
        
        # Compare different texts
        print(f"\nDifferent texts divergence components:")
        different_score = analyzer.calculate_divergence(
            texts['base'],
            texts['different'],
            return_components=True
        )
        for k, v in different_score.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
        
        # Print topic distributions
        print("\nTopic Distribution Analysis:")
        base_topics = analyzer._get_topic_distribution(texts['base'])
        similar_topics = analyzer._get_topic_distribution(texts['similar'])
        different_topics = analyzer._get_topic_distribution(texts['different'])
        
        # Get top topics for each text
        def get_top_topics(dist, n=3):
            top_indices = np.argsort(dist)[-n:][::-1]
            return [(i+1, dist[i]) for i in top_indices]
        
        print("\nTop Topics (Topic #, Weight):")
        print(f"Base text: {get_top_topics(base_topics)}")
        print(f"Similar text: {get_top_topics(similar_topics)}")
        print(f"Different text: {get_top_topics(different_topics)}")
        
        print("=" * 80)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 40)
    
    similar_scores = []
    different_scores = []
    topic_divergences = []
    semantic_divergences = []
    
    for texts in test_texts.values():
        sim_score = analyzer.calculate_divergence(texts['base'], texts['similar'])
        diff_score = analyzer.calculate_divergence(texts['base'], texts['different'])
        similar_scores.append(sim_score)
        different_scores.append(diff_score)
        
        # Get component scores
        sim_components = analyzer.calculate_divergence(texts['base'], texts['similar'], return_components=True)
        topic_divergences.append(sim_components['topic_jsd'])
        semantic_divergences.append(sim_components['semantic_jsd'])
    
    print(f"Average Similar Text Score: {np.mean(similar_scores):.4f} (±{np.std(similar_scores):.4f})")
    print(f"Average Different Text Score: {np.mean(different_scores):.4f} (±{np.std(different_scores):.4f})")
    print(f"Average Topic Divergence: {np.mean(topic_divergences):.4f} (±{np.std(topic_divergences):.4f})")
    print(f"Average Semantic Divergence: {np.mean(semantic_divergences):.4f} (±{np.std(semantic_divergences):.4f})")
    print("\nScore Ranges:")
    print(f"Similar Texts: {min(similar_scores):.4f} - {max(similar_scores):.4f}")
    print(f"Different Texts: {min(different_scores):.4f} - {max(different_scores):.4f}")
    
    print("\nAnalysis Complete!")

```

## Core Components

### Initialization Parameters

```python
analyzer = SRSWTIDivergenceV2(
    embedding_model='all-MiniLM-L6-v2',  # SentenceTransformer model
    semantic_dims=128,                    # Semantic space dimensions
    semantic_temperature=0.1,             # Distribution temperature
    n_topics=10,                         # Number of NMF topics
    min_df=2                             # Minimum document frequency
)
```

### Divergence Calculation Components

The system calculates divergence using multiple metrics:

```python
metrics = analyzer.calculate_divergence(text1, text2, return_components=True)
```

Returns:
```python
{
    'divergence_score': float,     # Final divergence score
    'cosine_similarity': float,    # Embedding similarity
    'semantic_jsd': float,         # Semantic distribution JSD
    'topic_jsd': float,           # Topic distribution JSD
    'entropy_p': float,           # First text entropy
    'entropy_q': float,           # Second text entropy
    'text_complexity_1': float,   # First text complexity
    'text_complexity_2': float,   # Second text complexity
    'semantic_weight': float,     # Semantic component weight
    'topic_weight': float         # Topic component weight
}
```

## Topic Modeling Features

### Topic Extraction

```python
# Get top words for each topic
topics = analyzer.get_topic_words(top_n=10)

# Print topics
for idx, topic_words in enumerate(topics):
    print(f"Topic {idx + 1}: {', '.join(topic_words)}")
```

### Topic Distribution

```python
# Get topic distribution for a text
topic_dist = analyzer._get_topic_distribution(text)
```

## Document Processing

### Multiple Document Analysis

```python
results = analyzer.process(
    documents=list_of_documents,
    reference_doc=optional_reference,
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

### NMF Initialization

```python
analyzer._initialize_nmf(texts)
```

Features:
- TF-IDF vectorization
- Non-negative Matrix Factorization
- Topic distribution computation

Parameters:
- max_features: 1000
- min_df: Configurable
- stop_words: 'english'
- n_components: Configurable

### Semantic Distribution Creation

```python
dist, complexity = analyzer._create_semantic_distribution(text)
```

Process:
1. Sentence tokenization
2. Embedding computation
3. Semantic anchor projection
4. Distribution normalization
5. Complexity calculation

### Jensen-Shannon Divergence

```python
jsd = analyzer._improved_jensen_shannon(dist1, dist2)
```

Features:
- Numerical stability
- Proper normalization
- Safe KL divergence
- Bounded output [0,1]

## Example Usage

### Basic Text Comparison

```python
analyzer = SRSWTIDivergenceV2(n_topics=5)

text1 = """
Recent advances in deep learning have revolutionized the field of computer vision.
Convolutional Neural Networks (CNNs) have demonstrated remarkable performance in
image classification, object detection, and semantic segmentation tasks.
"""

text2 = """
The evolution of neural networks has transformed visual computing paradigms.
Deep learning approaches, particularly Convolutional Neural Networks, have
achieved unprecedented accuracy in computer vision tasks.
"""

score = analyzer.calculate_divergence(text1, text2)
print(f"Divergence score: {score}")
```

### Topic Analysis

```python
# Initialize with documents
all_texts = [text1, text2, text3]
analyzer._initialize_nmf(all_texts)

# Get topic information
topics = analyzer.get_topic_words(top_n=5)
for idx, words in enumerate(topics):
    print(f"Topic {idx + 1}: {', '.join(words)}")

# Get topic distribution
dist = analyzer._get_topic_distribution(text1)
```

## Dependencies

Required packages:
- numpy
- scipy
- sentence-transformers
- nltk
- sklearn
- torch

NLTK Resources:
- stopwords
- punkt

## Best Practices

1. Text Preparation
   - Clean input text
   - Proper sentence structure
   - Sufficient document length

2. Topic Modeling
   - Adjust n_topics based on corpus
   - Consider min_df for vocabulary
   - Review topic coherence

3. Performance
   - Batch similar length texts
   - Monitor memory usage
   - Cache embeddings when possible

## Error Handling

The implementation includes error handling for:
- Model loading failures
- NMF initialization issues
- Empty text inputs
- Numerical computation errors
