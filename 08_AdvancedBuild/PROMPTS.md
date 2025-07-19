# PROMPTS.md

This file records prompts and their corresponding answers from our AI development sessions.

---

## Prompt 2: Are you familiar with LangGraph based built in semantic chunking constructs?

**Answer:**

Yes, I'm familiar with LangGraph's semantic chunking constructs! LangGraph provides several built-in tools for semantic chunking that are particularly useful for processing and analyzing text data. Let me search for more specific information about these constructs.

Based on the search results, I can see that LangGraph provides several built-in tools for semantic chunking that are particularly useful for processing and analyzing text data. Here's a comprehensive overview:

### **LangGraph's Built-in Semantic Chunking Constructs**

#### **1. RecursiveCharacterTextSplitter**
This is one of the most commonly used chunking tools in LangGraph:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
split_documents = text_splitter.split_documents(docs)
```

**Key Features:**
- **Intelligent Splitting**: Tries to split on natural boundaries (sentences, paragraphs, etc.)
- **Overlap Control**: The `chunk_overlap` parameter preserves context between chunks
- **Size Management**: Respects `chunk_size` while maintaining semantic coherence

#### **2. Semantic Chunking Approaches**

LangGraph supports several semantic chunking strategies:

**Sentence-Level Semantic Chunking**
```python
def semantic_sentence_chunking(text, similarity_threshold=0.7):
    sentences = split_into_sentences(text)
    sentence_embeddings = [embed(sent) for sent in sentences]
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(
            sentence_embeddings[i-1], 
            sentence_embeddings[i]
        )
        
        if similarity > similarity_threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
    
    return chunks
```

**Topic-Based Segmentation**
- Uses topic modeling (LDA, BERTopic) to identify topic boundaries
- Groups content by semantic topics rather than arbitrary lengths
- Maintains topical coherence within chunks

#### **3. TOC-Based Chunking**

LangGraph also supports Table of Contents (TOC) based chunking:

```python
class TOCChunker:
    def __init__(self, max_chunk_size=1000, preserve_hierarchy=True):
        self.max_chunk_size = max_chunk_size
        self.preserve_hierarchy = preserve_hierarchy
    
    def extract_toc_from_markdown(self, text):
        """Extract TOC from markdown-formatted text"""
        lines = text.split('\n')
        toc_entries = []
        current_content = []
        
        for line in lines:
            heading_match = re.match(r'^(#+)\s+(.+)$', line.strip())
            if heading_match:
                # Create new TOC entry
                level = len(heading_match.group(1))
                heading = heading_match.group(2)
                entry = TOCEntry(level, heading, "", [])
                toc_entries.append(entry)
        
        return toc_entries
```

#### **4. Advanced Semantic Chunking Techniques**

**Recursive Semantic Splitting**
- Hierarchical chunking that preserves semantic relationships
- Splits large sections into smaller semantic units
- Maintains parent-child relationships

**Multi-Modal Semantic Chunking**
- Considers text, images, tables, and other modalities
- Preserves relationships between different content types
- Useful for complex documents with mixed content

#### **5. Integration with LangGraph State Management**

LangGraph's semantic chunking constructs work seamlessly with its state management system:

```python
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    context: List[Document]
    response: str

def retrieve(state):
    retrieved_docs = retriever.invoke(state["question"])
    return {"context": retrieved_docs}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
```

#### **6. Benefits of LangGraph's Semantic Chunking**

1. **Better Context Preservation** - Maintains semantic coherence
2. **Improved Retrieval Quality** - More relevant chunks retrieved
3. **Reduced Noise** - Less irrelevant information in context
4. **Better Answer Quality** - More coherent and accurate responses
5. **Adaptive Granularity** - Chunk size adapts to content complexity

#### **7. Use Cases**

- **RAG Applications** - For building retrieval-augmented generation systems
- **Document Processing** - For analyzing large documents
- **Knowledge Management** - For organizing and retrieving information
- **Content Analysis** - For understanding document structure and meaning

#### **8. Evaluation with Ragas**

LangGraph's semantic chunking can be evaluated using Ragas:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy

# Evaluate the chunking quality
results = evaluate(
    evaluation_dataset,
    metrics=[faithfulness, answer_relevancy, context_relevancy],
    llm=evaluator_llm
)
```

Would you like me to help you implement any specific semantic chunking approach using LangGraph, or do you have questions about particular constructs?

---

## Prompt 3: Can you explain the 6 metrics - LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy, ContextEntityRecall, NoiseSensitivity. Please also outline how each is computed and which are "LLM as judge" based metrics and which are algorithmic?

**Answer:**

I'll explain each of the 6 metrics used in RAG evaluation, how they're computed, and whether they use LLM as judge or algorithmic approaches.

Based on the Ragas documentation, I can provide you with a comprehensive explanation of the 6 metrics you mentioned. Let me break down each metric, how they're computed, and whether they use LLM as judge or algorithmic approaches:

### **1. LLMContextRecall**

**What it measures:** How well the retrieved context covers the information needed to answer the question.

**Computation method:** **LLM as Judge**
- Uses an LLM to evaluate whether the retrieved context contains the necessary information to answer the question
- Compares the question against the retrieved context to determine recall completeness
- Returns a score between 0 and 1

**Example:**
```python
from ragas.metrics import LLMContextRecall

context_recall = LLMContextRecall(llm=evaluator_llm)
# Evaluates if retrieved context contains all necessary information for the question
```

### **2. Faithfulness**

**What it measures:** Whether the generated response is faithful to the retrieved context (no hallucination).

**Computation method:** **LLM as Judge**
- Uses an LLM to check if all claims in the response are supported by the retrieved context
- Identifies hallucinations by comparing response claims against context
- Returns a score between 0 and 1 (higher = more faithful)

**Example:**
```python
from ragas.metrics import Faithfulness

faithfulness = Faithfulness(llm=evaluator_llm)
# Checks if response only contains information from retrieved context
```

### **3. FactualCorrectness**

**What it measures:** The factual accuracy of the response compared to a reference answer.

**Computation method:** **LLM as Judge**
- Uses an LLM to compare the response against a reference answer
- Evaluates factual accuracy, completeness, and correctness
- Requires a reference answer for comparison
- Returns a score between 0 and 1

**Example:**
```python
from ragas.metrics import FactualCorrectness

factual_correctness = FactualCorrectness(llm=evaluator_llm)
# Compares response against reference answer for factual accuracy
```

### **4. ResponseRelevancy**

**What it measures:** How relevant the response is to the user's question.

**Computation method:** **LLM as Judge**
- Uses an LLM to evaluate if the response directly addresses the question
- Checks for relevance, completeness, and appropriateness
- Returns a score between 0 and 1

**Example:**
```python
from ragas.metrics import ResponseRelevancy

response_relevancy = ResponseRelevancy(llm=evaluator_llm)
# Evaluates if response is relevant to the question
```

### **5. ContextEntityRecall**

**What it measures:** How well the retrieved context captures the key entities mentioned in the question.

**Computation method:** **Algorithmic**
- Uses named entity recognition (NER) to extract entities from question and context
- Compares entity overlap between question and retrieved context
- Calculates recall based on entity coverage
- Returns a score between 0 and 1

**Example:**
```python
from ragas.metrics import ContextEntityRecall

entity_recall = ContextEntityRecall()
# Extracts entities and calculates recall without LLM
```

### **6. NoiseSensitivity**

**What it measures:** How sensitive the system is to irrelevant or noisy information in the context.

**Computation method:** **Algorithmic**
- Uses statistical methods to measure the impact of noise on response quality
- Analyzes how well the system filters out irrelevant information
- May use embedding similarity or other algorithmic approaches
- Returns a score indicating noise sensitivity

**Example:**
```python
from ragas.metrics import NoiseSensitivity

noise_sensitivity = NoiseSensitivity()
# Measures system's ability to handle noisy context
```

### **Summary: LLM vs Algorithmic Metrics**

#### **LLM as Judge Metrics (4):**
1. **LLMContextRecall** - Uses LLM to evaluate context completeness
2. **Faithfulness** - Uses LLM to detect hallucinations
3. **FactualCorrectness** - Uses LLM to compare against reference
4. **ResponseRelevancy** - Uses LLM to evaluate response relevance

#### **Algorithmic Metrics (2):**
1. **ContextEntityRecall** - Uses NER and entity matching
2. **NoiseSensitivity** - Uses statistical/embedding-based methods

### **Key Differences:**

**LLM as Judge Metrics:**
- More nuanced and context-aware
- Can understand semantic meaning
- May have some non-determinism
- Often closer to human judgment
- Require LLM configuration

**Algorithmic Metrics:**
- Deterministic and reproducible
- Faster computation
- Less expensive
- More objective but potentially less nuanced
- Don't require LLM setup

### **Usage Example:**

```python
from ragas import evaluate
from ragas.metrics import (
    LLMContextRecall, Faithfulness, FactualCorrectness,
    ResponseRelevancy, ContextEntityRecall, NoiseSensitivity
)

# LLM-based metrics
llm_metrics = [
    LLMContextRecall(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm),
    ResponseRelevancy(llm=evaluator_llm)
]

# Algorithmic metrics
algo_metrics = [
    ContextEntityRecall(),
    NoiseSensitivity()
]

# Combine all metrics
all_metrics = llm_metrics + algo_metrics

# Evaluate
results = evaluate(dataset=eval_dataset, metrics=all_metrics)
```

This combination gives you a comprehensive view of your RAG system's performance from both semantic (LLM-based) and statistical (algorithmic) perspectives.

--- 