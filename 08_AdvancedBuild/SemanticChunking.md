# Semantic Chunking Approaches for Vector Databases

This document contains a comprehensive discussion on various semantic chunking approaches for creating vector databases, including RAG applications, semantic chunking techniques, and TOC-based chunking methods.

---

## Q1: What is RAG application using Naive retrieval?

**RAG (Retrieval-Augmented Generation) with Naive Retrieval**

RAG is a technique that combines retrieval-based methods with generative models to improve the quality and factual accuracy of generated text by grounding it in external knowledge.

### Naive Retrieval in RAG

**Naive retrieval** refers to the simplest form of retrieval mechanism in RAG systems, typically involving:

#### 1. Basic Similarity Search
- Uses simple vector similarity (cosine similarity, dot product) between query and document embeddings
- No sophisticated ranking or re-ranking mechanisms
- Direct retrieval based on embedding proximity

#### 2. Fixed-K Retrieval
- Retrieves a fixed number of documents (e.g., top-5 or top-10)
- No dynamic adjustment based on query complexity or relevance scores
- No threshold-based filtering

#### 3. Single-Pass Retrieval
- One retrieval step with no iterative refinement
- No query expansion or reformulation
- No multi-hop reasoning

### Example of Naive RAG Pipeline

```python
# Naive RAG approach
def naive_rag_pipeline(query, vector_store, llm, k=5):
    # 1. Embed the query
    query_embedding = embedding_model.embed(query)
    
    # 2. Simple similarity search
    retrieved_docs = vector_store.similarity_search(query_embedding, k=k)
    
    # 3. Concatenate retrieved context
    context = "\n".join([doc.content for doc in retrieved_docs])
    
    # 4. Generate response
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm.generate(prompt)
    
    return response
```

### Limitations of Naive Retrieval

1. **No relevance filtering** - May include irrelevant documents
2. **Fixed retrieval count** - Doesn't adapt to query complexity
3. **No semantic understanding** - Purely embedding-based matching
4. **Context window issues** - May exceed model limits with irrelevant content
5. **No multi-step reasoning** - Can't handle complex, multi-faceted queries

### Contrast with Advanced RAG

Advanced RAG systems use techniques like:
- **Hybrid search** (combining dense and sparse retrieval)
- **Re-ranking** based on relevance scores
- **Query expansion** and reformulation
- **Multi-hop retrieval** for complex reasoning
- **Adaptive retrieval** based on confidence scores

The naive approach is simpler to implement but often produces lower-quality results compared to more sophisticated retrieval strategies.

---

## Q2: Are you familiar with semantic chunking approaches to creating a vector database?

Yes, I'm familiar with semantic chunking approaches for creating vector databases. Semantic chunking is a more sophisticated alternative to traditional fixed-size chunking methods.

### Traditional vs Semantic Chunking

#### Traditional Chunking
- **Fixed-size chunks** (e.g., 500 tokens, 1000 characters)
- **Overlap-based** (e.g., 50-100 token overlap)
- **Arbitrary boundaries** that may split related concepts

#### Semantic Chunking
- **Meaning-based boundaries** that preserve semantic coherence
- **Variable-size chunks** based on content structure
- **Context-aware splitting** that maintains topical integrity

### Common Semantic Chunking Approaches

#### 1. **Sentence-Level Semantic Chunking**
```python
# Groups sentences by semantic similarity
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

#### 2. **Topic-Based Segmentation**
- Uses topic modeling (LDA, BERTopic) to identify topic boundaries
- Groups content by semantic topics rather than arbitrary lengths
- Maintains topical coherence within chunks

#### 3. **Structural Semantic Chunking**
- Respects document structure (headings, paragraphs, sections)
- Combines structural and semantic signals
- Preserves hierarchical relationships

#### 4. **Embedding-Based Clustering**
```python
# Clusters paragraphs by semantic similarity
def embedding_based_chunking(paragraphs, max_chunk_size=1000):
    embeddings = [embed(para) for para in paragraphs]
    
    # Cluster similar paragraphs
    clusters = cluster_by_similarity(embeddings, paragraphs)
    
    # Create chunks respecting size limits
    chunks = []
    for cluster in clusters:
        if len(cluster) <= max_chunk_size:
            chunks.append(" ".join(cluster))
        else:
            # Sub-divide large clusters
            chunks.extend(subdivide_cluster(cluster, max_chunk_size))
    
    return chunks
```

### Advanced Semantic Chunking Techniques

#### 1. **Recursive Semantic Splitting**
- Hierarchical chunking that preserves semantic relationships
- Splits large sections into smaller semantic units
- Maintains parent-child relationships

#### 2. **Multi-Modal Semantic Chunking**
- Considers text, images, tables, and other modalities
- Preserves relationships between different content types
- Useful for complex documents with mixed content

#### 3. **Query-Aware Chunking**
- Adapts chunking strategy based on expected query patterns
- Pre-processes content considering likely information needs
- Optimizes for specific retrieval scenarios

### Implementation Example

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

class SemanticChunker:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def chunk_by_semantic_similarity(self, text, threshold=0.7):
        # Split into sentences
        sentences = self.split_sentences(text)
        
        # Get embeddings
        embeddings = self.model.encode(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Group semantically similar sentences
        chunks = []
        used = set()
        
        for i, sentence in enumerate(sentences):
            if i in used:
                continue
                
            # Find similar sentences
            similar_indices = np.where(similarity_matrix[i] > threshold)[0]
            chunk_sentences = [sentences[j] for j in similar_indices if j not in used]
            
            if chunk_sentences:
                chunks.append(" ".join(chunk_sentences))
                used.update(similar_indices)
        
        return chunks
```

### Benefits of Semantic Chunking

1. **Better Context Preservation** - Maintains semantic coherence
2. **Improved Retrieval Quality** - More relevant chunks retrieved
3. **Reduced Noise** - Less irrelevant information in context
4. **Better Answer Quality** - More coherent and accurate responses
5. **Adaptive Granularity** - Chunk size adapts to content complexity

### Challenges

1. **Computational Overhead** - More expensive than fixed chunking
2. **Embedding Quality Dependency** - Relies on good semantic embeddings
3. **Parameter Tuning** - Requires careful threshold and parameter selection
4. **Domain Specificity** - May need domain-specific adaptations

Semantic chunking is particularly valuable for complex documents, technical content, and scenarios where maintaining context and meaning is crucial for accurate retrieval and generation.

---

## Q3: Are you familiar with chunking based on TOC of a document?

Yes, I'm familiar with TOC (Table of Contents) based chunking! This is a structural semantic chunking approach that leverages the hierarchical organization of documents.

### TOC-Based Chunking Overview

TOC-based chunking uses the document's inherent structure (headings, sections, subsections) to create meaningful chunks that preserve the logical organization and hierarchy of information.

### How TOC-Based Chunking Works

#### 1. **Extract Document Structure**
```python
def extract_toc_structure(document):
    """Extract hierarchical structure from document"""
    headings = []
    
    # Extract headings with their levels
    for element in document.elements:
        if element.type == "heading":
            headings.append({
                "text": element.text,
                "level": element.level,  # H1, H2, H3, etc.
                "position": element.position
            })
    
    return headings
```

#### 2. **Build Hierarchical Tree**
```python
class DocumentNode:
    def __init__(self, heading, level, content=""):
        self.heading = heading
        self.level = level
        self.content = content
        self.children = []
        self.parent = None

def build_document_tree(headings, content_sections):
    """Build hierarchical tree from headings"""
    root = DocumentNode("Root", 0)
    stack = [root]
    
    for i, heading in enumerate(headings):
        node = DocumentNode(heading["text"], heading["level"])
        
        # Find appropriate parent
        while stack and stack[-1].level >= heading["level"]:
            stack.pop()
        
        parent = stack[-1]
        node.parent = parent
        parent.children.append(node)
        stack.append(node)
        
        # Add content between this heading and next
        if i < len(headings) - 1:
            start_pos = heading["position"]
            end_pos = headings[i+1]["position"]
            node.content = extract_content(content_sections, start_pos, end_pos)
    
    return root
```

#### 3. **Create Context-Aware Chunks**
```python
def create_toc_chunks(document_tree, max_chunk_size=1000):
    """Create chunks based on document structure"""
    chunks = []
    
    def traverse_node(node, parent_context=""):
        # Build hierarchical context
        context = f"{parent_context}\n# {node.heading}" if node.heading != "Root" else ""
        full_content = f"{context}\n{node.content}".strip()
        
        if len(full_content) <= max_chunk_size:
            # Single chunk for this section
            if full_content:
                chunks.append({
                    "content": full_content,
                    "metadata": {
                        "heading": node.heading,
                        "level": node.level,
                        "path": get_heading_path(node)
                    }
                })
        else:
            # Split large sections
            if node.children:
                # Process children individually
                for child in node.children:
                    traverse_node(child, context)
            else:
                # Split content semantically within section
                sub_chunks = split_large_section(full_content, max_chunk_size)
                for i, chunk in enumerate(sub_chunks):
                    chunks.append({
                        "content": chunk,
                        "metadata": {
                            "heading": f"{node.heading} (part {i+1})",
                            "level": node.level,
                            "path": get_heading_path(node)
                        }
                    })
        
        # Process children if section wasn't split
        if len(full_content) <= max_chunk_size:
            for child in node.children:
                traverse_node(child, context)
    
    traverse_node(document_tree)
    return chunks
```

### Advanced TOC-Based Techniques

#### 1. **Hierarchical Context Preservation**
```python
def create_hierarchical_chunks(node, ancestors=[]):
    """Include parent context in chunks"""
    # Build context from ancestors
    context_path = " > ".join([anc.heading for anc in ancestors])
    
    chunk = {
        "content": node.content,
        "metadata": {
            "heading": node.heading,
            "context_path": context_path,
            "level": node.level,
            "ancestors": [anc.heading for anc in ancestors]
        }
    }
    
    return chunk
```

#### 2. **Multi-Level Chunking**
```python
def multi_level_chunking(document_tree):
    """Create chunks at multiple hierarchical levels"""
    chunks = []
    
    # Level 1: Major sections (H1)
    for h1_node in document_tree.children:
        if h1_node.level == 1:
            # Create section-level chunk
            section_content = aggregate_section_content(h1_node)
            chunks.append({
                "content": section_content,
                "type": "section",
                "metadata": {"heading": h1_node.heading, "level": 1}
            })
            
            # Level 2: Subsections (H2)
            for h2_node in h1_node.children:
                subsection_content = aggregate_section_content(h2_node)
                chunks.append({
                    "content": subsection_content,
                    "type": "subsection",
                    "metadata": {
                        "heading": h2_node.heading,
                        "level": 2,
                        "parent": h1_node.heading
                    }
                })
    
    return chunks
```

#### 3. **Smart Section Merging**
```python
def smart_section_merging(chunks, min_size=200, max_size=1000):
    """Merge small adjacent sections intelligently"""
    merged_chunks = []
    current_chunk = ""
    current_metadata = None
    
    for chunk in chunks:
        if len(current_chunk) + len(chunk["content"]) <= max_size:
            # Merge with current chunk
            current_chunk += f"\n\n{chunk['content']}"
            if current_metadata is None:
                current_metadata = chunk["metadata"]
        else:
            # Start new chunk
            if current_chunk and len(current_chunk) >= min_size:
                merged_chunks.append({
                    "content": current_chunk,
                    "metadata": current_metadata
                })
            current_chunk = chunk["content"]
            current_metadata = chunk["metadata"]
    
    return merged_chunks
```

### Practical Implementation Example

```python
import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TOCEntry:
    level: int
    heading: str
    content: str
    children: List['TOCEntry']
    page_number: Optional[int] = None

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
                # Save previous content
                if current_content:
                    if toc_entries:
                        toc_entries[-1].content = '\n'.join(current_content)
                    current_content = []
                
                # Create new TOC entry
                level = len(heading_match.group(1))
                heading = heading_match.group(2)
                entry = TOCEntry(level, heading, "", [])
                toc_entries.append(entry)
            else:
                current_content.append(line)
        
        # Add final content
        if current_content and toc_entries:
            toc_entries[-1].content = '\n'.join(current_content)
        
        return toc_entries
    
    def create_chunks(self, toc_entries):
        """Create chunks from TOC entries"""
        chunks = []
        
        for entry in toc_entries:
            if self.preserve_hierarchy:
                # Include heading in content
                full_content = f"{'#' * entry.level} {entry.heading}\n{entry.content}"
            else:
                full_content = entry.content
            
            if len(full_content) <= self.max_chunk_size:
                chunks.append({
                    "content": full_content.strip(),
                    "metadata": {
                        "heading": entry.heading,
                        "level": entry.level,
                        "type": "toc_section"
                    }
                })
            else:
                # Split large sections
                sub_chunks = self._split_large_section(full_content, entry)
                chunks.extend(sub_chunks)
        
        return chunks
```

### Benefits of TOC-Based Chunking

1. **Logical Organization** - Maintains document structure and flow
2. **Context Preservation** - Keeps related information together
3. **Hierarchical Metadata** - Enables sophisticated filtering and retrieval
4. **User-Friendly** - Aligns with how humans organize information
5. **Scalable** - Works well with large, complex documents

### Use Cases

- **Technical Documentation** - API docs, user manuals
- **Academic Papers** - Research papers with clear sections
- **Legal Documents** - Contracts, policies with numbered sections
- **Books and Reports** - Chapter-based organization
- **Wiki Articles** - Structured encyclopedia entries

### Challenges

1. **Inconsistent Structures** - Documents may have poor or inconsistent heading hierarchy
2. **Missing TOC** - Some documents lack clear structural markers
3. **Nested Complexity** - Deep hierarchies can be challenging to handle
4. **Size Imbalances** - Some sections much larger than others

TOC-based chunking is particularly powerful when combined with other techniques like semantic similarity and is ideal for well-structured documents where maintaining logical organization is crucial for understanding and retrieval.

---

## Summary

This document covers three important aspects of chunking for vector databases:

1. **Naive RAG Retrieval** - Understanding the baseline approach and its limitations
2. **Semantic Chunking** - Advanced techniques for meaning-preserving document segmentation
3. **TOC-Based Chunking** - Structural approaches that leverage document hierarchy

Each approach has its strengths and use cases, and the best choice depends on the specific requirements of your application, the nature of your documents, and the types of queries you expect to handle.

The key insight is that moving beyond simple fixed-size chunking to more sophisticated semantic and structural approaches can significantly improve retrieval quality and, consequently, the performance of RAG systems.