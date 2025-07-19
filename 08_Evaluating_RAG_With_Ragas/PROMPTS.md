# PROMPTS.md

This file contains all prompts and their corresponding answers used in the RAG evaluation project with Ragas.

## Project Overview

This project focuses on evaluating RAG (Retrieval-Augmented Generation) systems using the Ragas framework. The evaluation covers various aspects including faithfulness, answer relevancy, context precision, and other key metrics.

## Model Separation Strategy

In RAG evaluation, it's crucial to use different models for **generation** and **judging** to avoid evaluation bias:

### Why Different Models?

1. **Avoiding Self-Evaluation Bias**: Using the same model for both generation and evaluation can lead to inflated scores, as the model may be more lenient when evaluating its own outputs.

2. **Independent Assessment**: Different models provide independent perspectives on quality, leading to more reliable evaluation results.

3. **Realistic Performance Measurement**: In production systems, the generation model and evaluation criteria are typically separate, so this setup better reflects real-world scenarios.

### Model Configuration in This Project:

- **Generation Model**: `gpt-4.1-nano` - Used to generate responses in the RAG pipeline
- **Judge Model**: `gpt-4.1-mini` - Used by Ragas to evaluate the quality of generated responses
- **Synthetic Data Generation**: `gpt-4.1` - Used to create test questions and reference data

This separation ensures that the evaluation metrics (faithfulness, relevancy, etc.) are assessed by a model that didn't generate the responses being evaluated.

## Reranking with Cohere

The notebook implements **reranking** using Cohere's rerank model to improve retrieval quality. Here's how it works:

### What is Reranking?

Reranking is a two-stage retrieval process that combines:
1. **Initial Retrieval**: Fast vector similarity search to get a broad set of candidates
2. **Reranking**: More sophisticated model that reorders candidates based on relevance

### Implementation in the Notebook:

```python
# Step 1: Increase initial retrieval to get more candidates
adjusted_example_retriever = vector_store.as_retriever(search_kwargs={"k": 20})

# Step 2: Apply Cohere reranking to compress and reorder
def retrieve_adjusted(state):
  compressor = CohereRerank(model="rerank-v3.5")
  compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=adjusted_example_retriever, 
    search_kwargs={"k": 5}
  )
  retrieved_docs = compression_retriever.invoke(state["question"])
  return {"context": retrieved_docs}
```

### Why Use Reranking?

1. **Better Precision**: Reranking models are specifically trained to understand query-document relevance better than simple vector similarity
2. **Contextual Understanding**: Unlike embedding similarity, reranking can understand nuanced relationships between queries and documents
3. **Quality vs Speed Trade-off**: Initial retrieval is fast, reranking is slower but more accurate
4. **Compression**: Reduces from 20 candidates to 5 final documents while improving relevance

### The Process:
1. **Initial Retrieval**: Get 20 documents using vector similarity
2. **Reranking**: Cohere's `rerank-v3.5` model reorders these 20 documents by relevance
3. **Compression**: Select the top 5 most relevant documents for the final context
4. **Generation**: Use these 5 high-quality documents to generate the response

This approach typically improves metrics like **Context Precision** and **Response Relevancy** by providing more relevant context to the generation model.

## Q&A: Vector Similarity vs. Reranking

### Question: 
"So essentially what you are saying is that if we do retrieval using the vector DB of the most similar chunks it is not ranked correctly based on similarity metric?"

### Answer:

Great question! You're touching on a key limitation of vector similarity search. Here's the difference:

#### Vector Similarity (Initial Retrieval)
- **What it does**: Finds documents with similar **semantic meaning** to the query
- **How it works**: Compares embeddings (dense vectors) using cosine similarity or other distance metrics
- **Limitation**: Only considers **semantic similarity**, not **relevance to the specific query**

#### Reranking (Cohere's Approach)
- **What it does**: Reorders documents based on **relevance to the specific query**
- **How it works**: Uses a specialized model trained to understand query-document relationships
- **Advantage**: Can understand nuanced, contextual relevance that goes beyond semantic similarity

#### Example Problem with Vector-Only Retrieval

**Query**: "What are the interest rates for student loans?"

**Document A**: "Student loans have interest rates ranging from 3.73% to 7.54% depending on the loan type and borrower's credit score."

**Document B**: "Interest rates for various financial products include mortgages at 4.5%, car loans at 6.2%, and student loans at 5.8%."

**Vector Similarity Result**: Document B might rank higher because it contains more "interest rate" mentions and similar vocabulary.

**Reranking Result**: Document A would rank higher because it's specifically about student loan interest rates, even though it has fewer "interest rate" mentions.

#### Why This Matters

1. **Semantic vs. Contextual**: Vector similarity finds semantically similar content, but reranking finds contextually relevant content
2. **Query-Specific Understanding**: Reranking models understand what the query is actually asking for
3. **Nuanced Relationships**: Can distinguish between "mentions the topic" vs. "answers the specific question"

#### The Two-Stage Approach

```
Query → Vector Search (20 docs) → Reranking (reorder by relevance) → Top 5 → Generation
```

This combines:
- **Speed** of vector search (fast initial filtering)
- **Accuracy** of reranking (precise relevance scoring)

**Conclusion**: Vector similarity alone doesn't always rank documents correctly for the specific query. Reranking adds that crucial layer of query-specific relevance understanding that vector similarity misses.

## Q&A: ContextualCompressionRetriever Base Compressor Options

### Question:
"Take a look at the cell tagged 'compression_retriever' in the same notebook. The ContextualCompressionRetriever takes in a 'base_compressor' param. For this case we are using the CohereRerank model. What other options are there for the 'base_compressor' param?"

### Answer:

The `ContextualCompressionRetriever` accepts various types of compressors for the `base_compressor` parameter. Here are the main categories and options:

#### 1. **Reranking Compressors** (Like CohereRerank)
- **CohereRerank**: Uses Cohere's rerank models (`rerank-v3.5`, `rerank-english-v3.0`)
- **FlashrankRerank**: Uses FlashRank for document reranking
- **JinaRerank**: Uses Jina's reranking service
- **DashScopeRerank**: Uses Alibaba Cloud's DashScope reranking
- **InfinityRerank**: Uses Infinity reranking models
- **VertexAIRank**: Uses Google Cloud's Vertex AI reranking
- **WatsonxRerank**: Uses IBM's Watsonx reranking
- **CrossEncoderReranker**: Uses HuggingFace cross-encoder models

#### 2. **LLM-Based Compressors**
- **LLMChainExtractor**: Uses LLM to extract relevant content from documents
- **LLMChainFilter**: Uses LLM to filter out irrelevant documents
- **LLMListwiseRerank**: Uses LLM for zero-shot listwise document reranking
- **LLMLinguaCompressor**: Uses LLMLingua for prompt compression

#### 3. **Embedding-Based Compressors**
- **EmbeddingsFilter**: Filters documents based on embedding similarity threshold
- **EmbeddingsRedundantFilter**: Removes redundant documents based on embeddings

#### 4. **Pipeline Compressors**
- **DocumentCompressorPipeline**: Chains multiple compressors together
- **EmbeddingsContextReorder**: Reorders documents based on embedding similarity

#### 5. **Specialized Compressors**
- **LLMLinguaCompressor**: Compresses prompts using LLMLingua models
- **CharacterTextSplitter**: Splits documents into smaller chunks
- **RecursiveCharacterTextSplitter**: Splits documents recursively

#### Example Usage Patterns:

```python
# Reranking with different providers
compressor = CohereRerank(model="rerank-v3.5")
compressor = FlashrankRerank()
compressor = JinaRerank()

# LLM-based compression
compressor = LLMChainExtractor.from_llm(llm)
compressor = LLMChainFilter.from_llm(llm)
compressor = LLMListwiseRerank.from_llm(llm, top_n=3)

# Embedding-based filtering
compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
compressor = EmbeddingsRedundantFilter(embeddings=embeddings)

# Pipeline approach
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)
```

#### Key Considerations:

1. **Performance vs. Cost**: Reranking models are more accurate but slower and more expensive
2. **Model Requirements**: Some compressors require specific LLM capabilities (e.g., structured output)
3. **Use Case**: Choose based on whether you need filtering, reranking, or content extraction
4. **Integration**: All compressors work seamlessly with `ContextualCompressionRetriever`

The choice depends on your specific needs for accuracy, speed, cost, and the type of compression you want to achieve.

## Q&A: RAG vs. Reranked RAG Performance Analysis

### Question:
"Take a look at the cells tagged 'result_rag' and 'result_rerank' in the notebook. In the first case we have a basic RAG, in the 2nd we use the Cohere rerank model to rerank the chunks retrieved from the vector store. Why is it that, while the RAGAS metrics 'answer_relevancy' and 'noise_sensitivity_relevant' have increased, the rest have decreased?"

### Answer:

Looking at the evaluation results, we can see a clear trade-off pattern:

#### Results Comparison:

**Basic RAG (No Reranking):**
- `context_recall`: 0.8506
- `faithfulness`: 0.9032
- `factual_correctness`: 0.6217
- `answer_relevancy`: 0.8783
- `context_entity_recall`: 0.3755
- `noise_sensitivity_relevant`: 0.2780

**With Cohere Reranking:**
- `context_recall`: 0.7519 ⬇️ (-0.0987)
- `faithfulness`: 0.8402 ⬇️ (-0.0630)
- `factual_correctness`: 0.6017 ⬇️ (-0.0200)
- `answer_relevancy`: 0.9491 ⬆️ (+0.0708)
- `context_entity_recall`: 0.3318 ⬇️ (-0.0437)
- `noise_sensitivity_relevant`: 0.2941 ⬆️ (+0.0161)

#### Why This Trade-off Occurs:

**Important Clarification**: You're absolutely correct! The metrics like `context_recall`, `faithfulness`, `factual_correctness`, and `context_entity_recall` are measured against the **final context provided to the generation model** (the 5 reranked chunks), NOT the original 20 chunks from semantic similarity.

Here's what's actually happening:

1. **Context Compression Effect**: 
   - Reranking reduces from 20 initial documents to 5 final documents
   - The **evaluation metrics are measured against these 5 final chunks**
   - This means we're evaluating how well the system performs with a more focused, but smaller context set

2. **Quality vs. Quantity Trade-off**:
   - **Basic RAG**: Uses 5 chunks from broader semantic search
   - **Reranked RAG**: Uses 5 chunks from precision-focused reranking
   - Both systems provide the same amount of context to the generation model, but with different quality

3. **Metric-Specific Impacts**:

   **Improved Metrics:**
   - **`answer_relevancy`**: Reranking provides more focused, relevant context
   - **`noise_sensitivity_relevant`**: Better context filtering reduces noise

   **Decreased Metrics:**
   - **`context_recall`**: The 5 reranked chunks may contain less comprehensive information than the 5 best semantic chunks
   - **`faithfulness`**: More focused context may lead to less faithful reproduction of source material
   - **`factual_correctness`**: Precision-focused context may miss some factual details that were in the broader semantic results
   - **`context_entity_recall`**: Fewer entities available in the precision-focused context

4. **The Real Trade-off**:
   - **Semantic Retrieval**: Gets 5 chunks with broader coverage but potentially less relevance
   - **Reranked Retrieval**: Gets 5 chunks with higher relevance but potentially narrower coverage
   - The evaluation shows that **relevance wins** (better answer quality) but **coverage loses** (less comprehensive information)

#### Key Insight:
This demonstrates that reranking doesn't just compress context - it **changes the nature** of the context provided to the generation model. The 5 reranked chunks are more relevant but less comprehensive than the 5 best semantic chunks, leading to better answer quality but potentially less complete information coverage.

## Prompt Categories

### 1. Evaluation Prompts

#### Faithfulness Evaluation
**Prompt:** "Evaluate the faithfulness of the generated answer to the provided context. The answer should be factually consistent with the context and not introduce information that is not present in the context."

**Expected Output:** Binary classification (faithful/not faithful) with reasoning.

#### Answer Relevancy Evaluation
**Prompt:** "Assess the relevance of the generated answer to the given question. The answer should directly address the question asked without being overly verbose or missing key information."

**Expected Output:** Score on a scale of 1-5 with detailed reasoning.

#### Context Precision Evaluation
**Prompt:** "Evaluate the precision of the retrieved context in relation to the question. The context should contain information that is directly relevant to answering the question."

**Expected Output:** Score on a scale of 1-5 with explanation.

### 2. Question Generation Prompts

#### Question Generation for Evaluation
**Prompt:** "Generate diverse questions that test different aspects of the knowledge base. Include questions that require:
- Factual recall
- Comparative analysis
- Procedural understanding
- Complex reasoning"

**Expected Output:** List of questions with varying complexity levels.

### 3. Answer Generation Prompts

#### Standard Answer Generation
**Prompt:** "Based on the provided context, generate a comprehensive answer to the question. Ensure the answer is:
- Accurate and faithful to the context
- Relevant to the question asked
- Well-structured and clear
- Complete without being overly verbose"

**Expected Output:** Generated answer with proper formatting.

### 4. Context Retrieval Prompts

#### Context Retrieval Optimization
**Prompt:** "Retrieve the most relevant context passages for the given question. Focus on:
- Semantic similarity
- Information density
- Coverage of key concepts
- Avoiding redundancy"

**Expected Output:** Ranked list of context passages.

## Usage Guidelines

1. **Consistency**: Use the same prompt format across all evaluations for consistency
2. **Documentation**: Always document any modifications to prompts
3. **Versioning**: Track prompt versions and their impact on evaluation results
4. **Testing**: Test prompts with sample data before full evaluation

## Evaluation Metrics

The following metrics are evaluated using these prompts:

- **Faithfulness**: Measures if the generated answer is factually consistent with the provided context
- **Answer Relevancy**: Assesses how well the answer addresses the question
- **Context Precision**: Evaluates the relevance of retrieved context to the question
- **Context Recall**: Measures the completeness of retrieved context
- **Answer Correctness**: Evaluates the factual accuracy of answers

## Template Structure

Each prompt should follow this structure:

```markdown
### [Prompt Name]

**Purpose:** [Brief description of what this prompt evaluates]

**Prompt:** [The actual prompt text]

**Expected Output:** [Description of expected output format]

**Usage Notes:** [Any specific considerations or modifications]
```

## Q&A: Ragas convert_to_ragas_messages Function

### Question:
"Look at the cell tagged 'convert_to_ragas_messages' in the notebook Evaluating_Agents_with_Ragas. What does the Ragas function 'convert_to_ragas_messages' do?"

### Answer:

The `convert_to_ragas_messages` function is a **format conversion utility** provided by Ragas that transforms LangChain/LangGraph message objects into Ragas-compatible message format for evaluation purposes.

#### Key Functions:

1. **Format Translation**: Converts LangChain message types (like `HumanMessage`, `AIMessage`, `ToolMessage`) into Ragas message objects that the evaluation framework can understand.

2. **Trace Creation**: Creates a "trace" - which is essentially a chronological record of the conversation/interaction between the user, AI agent, and any tools used during the conversation.

3. **Evaluation Preparation**: Prepares the conversation data in the format required by Ragas evaluation metrics like:
   - Tool Call Accuracy
   - Agent Goal Accuracy  
   - Topic Adherence

#### Usage Pattern:
```python
from ragas.integrations.langgraph import convert_to_ragas_messages

# Convert LangChain messages to Ragas format
ragas_trace = convert_to_ragas_messages(result["messages"])
```

#### What is a "Trace"?
A trace represents the complete interaction flow - it's like a conversation log that captures:
- User inputs
- AI responses
- Tool calls and their results
- The sequence and timing of all interactions

This trace is then used by Ragas evaluation metrics to assess how well the agent performed in terms of accuracy, goal achievement, and topic adherence.

The function essentially bridges the gap between LangGraph's internal message format and Ragas's evaluation framework, enabling comprehensive agent performance assessment.

## Q&A: Agent Performance Metrics Calculation

### Question:
"Take a look at the cell tagged 'agent_perf_measures' in the same notebook. There are 3 measures listed there (with links to their descriptions): 'Tool call Accuracy', 'Agent Goal accuracy', 'Topic Adherence'. Review the links of each that describe that they do and explain HOW each of these are calculated"

### Answer:

Based on the Ragas documentation, here's how each of the three agent performance metrics are calculated:

#### 1. **Tool Call Accuracy**

**What it measures**: Evaluates whether the agent correctly identifies and calls the required tools with appropriate parameters.

**How it's calculated**:
- **Input**: Agent's conversation trace + reference tool calls (expected tool calls)
- **Process**: 
  1. Compares the agent's actual tool calls against a predefined list of expected tool calls
  2. Checks both tool names and arguments for exact matches (by default)
  3. Can be customized to use semantic similarity for argument comparison
- **Output**: Binary score (1.0 for perfect accuracy, 0.0 for failures)
- **Algorithm**: Exact string matching for tool names and arguments, with optional semantic similarity for arguments

**Example from notebook**:
```python
sample = MultiTurnSample(
    user_input=ragas_trace,
    reference_tool_calls=[
        r.ToolCall(name="get_metal_price", args={"metal_name": "copper"})
    ],
)
# Returns 1.0 if agent called get_metal_price with correct args
```

#### 2. **Agent Goal Accuracy**

**What it measures**: Evaluates whether the agent successfully achieves the user's intended goal.

**How it's calculated**:
- **Input**: Agent's conversation trace + reference goal (or inferred goal)
- **Process**:
  1. **With Reference**: Compares agent's final outcome against a predefined reference goal
  2. **Without Reference**: Uses LLM to infer the user's goal from the conversation and assess achievement
- **Output**: Binary score (1.0 for goal achieved, 0.0 for goal not achieved)
- **Algorithm**: LLM-as-a-judge evaluates whether the conversation successfully accomplished the intended goal

**Example from notebook**:
```python
sample = MultiTurnSample(
    user_input=ragas_trace,
    reference="Price of 10 grams of silver",
)
# Returns 1.0 if agent successfully provided the silver price
```

#### 3. **Topic Adherence**

**What it measures**: Evaluates whether the agent stays "on-topic" and doesn't stray from the intended use case.

**How it's calculated**:
- **Input**: Agent's conversation trace + reference topics
- **Process**:
  1. Uses LLM to evaluate if the conversation stays within the specified topic domains
  2. Can be configured for **precision** (how much of the response is on-topic) or **recall** (how much of the topic is covered)
- **Output**: Score between 0.0-1.0 (higher = better topic adherence)
- **Algorithm**: LLM-as-a-judge evaluates topic relevance across the entire conversation

**Example from notebook**:
```python
sample = MultiTurnSample(
    user_input=ragas_trace,
    reference_topics=["metals"]
)
scorer = TopicAdherenceScore(llm=evaluator_llm, mode="precision")
# Returns lower score when agent talks about birds instead of metals
```

#### **Key Differences in Calculation Approach**:

1. **Tool Call Accuracy**: Uses exact matching against predefined expectations
2. **Agent Goal Accuracy**: Uses LLM judgment to evaluate goal achievement
3. **Topic Adherence**: Uses LLM judgment to evaluate topic relevance with configurable precision/recall modes

All three metrics use **LLM-as-a-judge** evaluation, which provides more nuanced assessment than rule-based metrics, but introduces some non-determinism and requires careful prompt engineering for consistent results.

## Q&A: LLM vs Algorithmic Implementation of Agent Performance Metrics

### Question:
"Do each of these metrics utilize LLMs to obtain the measure or are they algorithmic? You may have to search more documentation or search the web for that"

### Answer:

Based on the Ragas documentation, here's how each of the three metrics is implemented:

#### 1. **Tool Call Accuracy** - **Algorithmic (with optional LLM customization)**

**Primary Implementation**: Algorithmic/rule-based
- **Default**: Uses exact string matching for tool names and arguments
- **Process**: Compares actual tool calls against reference tool calls using exact matching
- **No LLM required by default**

**Optional LLM Enhancement**:
- Can be customized to use `NonLLMStringSimilarity` for argument comparison
- This allows for semantic similarity matching instead of exact matching
- Only requires LLM if you want more flexible argument matching

**Example from documentation**:
```python
# Default (algorithmic)
tool_accuracy_scorer = ToolCallAccuracy()

# With LLM for semantic argument comparison
from ragas.metrics._string import NonLLMStringSimilarity
metric = ToolCallAccuracy()
metric.arg_comparison_metric = NonLLMStringSimilarity()
```

#### 2. **Agent Goal Accuracy** - **LLM-based**

**Implementation**: Fully LLM-based
- **Requires**: An LLM evaluator (`evaluator_llm`)
- **Process**: Uses LLM-as-a-judge to evaluate whether the agent achieved the user's goal
- **Two variants**:
  - `AgentGoalAccuracyWithReference`: Compares against predefined reference goal
  - `AgentGoalAccuracyWithoutReference`: Uses LLM to infer goal from conversation

**Example from documentation**:
```python
scorer = AgentGoalAccuracyWithReference(llm=evaluator_llm)
# or
scorer = AgentGoalAccuracyWithoutReference(llm=evaluator_llm)
```

#### 3. **Topic Adherence** - **LLM-based**

**Implementation**: Fully LLM-based
- **Requires**: An LLM evaluator (`evaluator_llm`)
- **Process**: Uses LLM to evaluate if conversation stays within specified topic domains
- **Configurable modes**: "precision" or "recall"

**Example from documentation**:
```python
scorer = TopicAdherenceScore(llm=evaluator_llm, mode="precision")
# or
scorer = TopicAdherenceScore(llm=evaluator_llm, mode="recall")
```

#### **Summary**:

| Metric | Implementation | LLM Required |
|--------|---------------|--------------|
| **Tool Call Accuracy** | Primarily algorithmic, optional LLM | No (by default) |
| **Agent Goal Accuracy** | Fully LLM-based | Yes |
| **Topic Adherence** | Fully LLM-based | Yes |

#### **Key Insights**:

1. **Tool Call Accuracy** is the most efficient and deterministic since it's primarily algorithmic
2. **Agent Goal Accuracy** and **Topic Adherence** provide more nuanced evaluation but require LLM calls and introduce non-determinism
3. All three can be customized with different LLM providers (OpenAI, Vertex AI, etc.)
4. The LLM-based metrics are more expensive and slower but provide more human-like judgment

This explains why Tool Call Accuracy is often the fastest and most reliable metric, while the other two provide more sophisticated evaluation at the cost of speed and determinism.

## Version History

- **v1.0**: Initial prompt set for basic RAG evaluation
- **v1.1**: Added specialized prompts for different evaluation scenarios
- **v1.2**: Enhanced prompts for better evaluation accuracy
- **v1.3**: Added Q&A about Ragas convert_to_ragas_messages function
- **v1.4**: Added Q&A about agent performance metrics calculation methods
- **v1.5**: Added Q&A about LLM vs algorithmic implementation of agent metrics

---

*This file should be updated whenever new prompts are added or existing prompts are modified to ensure proper documentation and reproducibility.* 