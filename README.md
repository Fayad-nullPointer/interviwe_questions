# Interview Questions Repository

This repository contains a polished set of interview-style questions and answers covering four major areas: Data Science, Generative AI and RAG, Agentic AI Systems, and Reinforcement Learning. The content below is written to be useful for study, revision, and interview preparation.

---

## 1. Data Science

### 1. What is data leakage, and how can it be detected?

Data leakage happens when information from the future or from the target variable unintentionally leaks into the training data. This makes the model look unrealistically strong during evaluation.

Common examples:
- Using target-related features that should not be available at prediction time
- Scaling features before splitting train/test data
- Using data from the future in time series problems

How to detect it:
- Check whether any feature is derived from the target or future information
- Ensure preprocessing is fit only on training data
- Use a proper time-based split for time series data
- Compare training and validation performance carefully; suspiciously perfect results may indicate leakage
- Review feature engineering pipelines for hidden target leakage

### 2. What is data drift and concept drift, and how can they be detected?

- Data drift: the input data distribution changes over time, even if the relationship between input and output stays the same.
- Concept drift: the relationship between input features and the target changes over time.

How to detect them:
- Data drift: monitor feature distributions using PSI, KS test, Jensen-Shannon divergence, or change detection methods
- Concept drift: monitor model performance over time using accuracy, F1-score, AUC, or error rate
- Track prediction confidence and calibration drift as well

### 3. Overfitting and underfitting

- Overfitting: the model learns the training data too well, including noise, and performs poorly on unseen data.
- Underfitting: the model is too simple to capture the pattern in the data and performs poorly on both training and test data.

How to address them:
- Overfitting: use more data, regularization, cross-validation, pruning, or simpler models
- Underfitting: increase model complexity, add useful features, or train longer

### 4. What is the Predictive Power Score (PPS)?

The Predictive Power Score is a model-agnostic metric that measures how well one feature predicts another feature or the target. Unlike correlation, PPS can capture nonlinear relationships and asymmetry.

It is useful because:
- It works for both numerical and categorical data
- It can reveal hidden relationships that correlation may miss
- It helps feature selection and explainability

### 5. What is PCA, and what is kernelized PCA?

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms data into a smaller set of uncorrelated features called principal components while preserving as much variance as possible.

Kernelized PCA extends PCA using the kernel trick. It maps data into a higher-dimensional space so that nonlinear patterns can be captured more effectively.

### 6. What are eigenvalues and eigenvectors, and how are they used in PCA?

An eigenvector is a direction that remains unchanged after a transformation, and an eigenvalue tells us how much the data is stretched or compressed along that direction.

In PCA:
- The covariance matrix of the data is computed
- Its eigenvectors give the principal directions
- The corresponding eigenvalues indicate the amount of variance explained by each direction
- The largest eigenvalues correspond to the most important components

### 7. How can we detect non-linearity, and what models are suitable for it?

Non-linearity can be detected by:
- Residual plots showing curved patterns
- The relationship between input and output not being approximately linear
- Feature importance from tree-based models
- Statistical tests such as Ramsey RESET

Suitable models for non-linear data:
- Decision Trees
- Random Forests
- Gradient Boosting models such as XGBoost and LightGBM
- SVM with RBF kernel
- Neural networks

### 8. What is cross-entropy and information gain?

- Cross-entropy measures how different a predicted probability distribution is from the true distribution. It is commonly used as a loss function in classification.
- Information gain measures how much uncertainty is reduced by splitting data using a feature. It is commonly used in decision trees.

In simple terms:
- Cross-entropy evaluates prediction quality.
- Information gain evaluates feature usefulness for splitting.

---

## 2. Generative AI and RAG

### 1. What is chunking, and how is it used?

Chunking means splitting documents into smaller pieces before indexing or retrieval. This helps the system process large documents efficiently and improves retrieval quality.

Common chunking techniques:
- Fixed-size chunking
- Recursive chunking
- Sliding window chunking
- Semantic chunking

How to use it:
- Choose chunk size based on the embedding model and the task
- Preserve context by keeping related sentences together
- Use overlap between chunks to avoid losing important information

### 2. How do we handle noisy documents containing image noise and text noise?

Noisy documents should be cleaned before embedding or retrieval. A typical pipeline includes:
- OCR and layout analysis for scanned or image-based documents
- Text cleaning such as removing junk characters or repeated artifacts
- Filtering irrelevant pages or sections
- Using document structure and metadata to improve quality
- Applying quality checks before indexing

### 3. How are precision and recall affected by long and short chunks?

- Short chunks generally improve retrieval precision because they are more focused and less likely to include irrelevant content.
- Long chunks can improve recall by containing more context, but they may reduce precision if they include too much unrelated material.
- A balanced chunk size usually performs best, often with overlap.

### 4. How can RAG be used on tabular data?

RAG can be used on tabular data by converting tables into a text or JSON representation and then retrieving the most relevant rows, columns, or summaries.

A suitable approach:
- Convert rows into structured text such as “Column: value” pairs
- Chunk by logical groups, such as related rows or table sections
- Use hybrid retrieval with keyword search and semantic search
- For complex tasks, combine retrieval with SQL generation or query execution

Suitable models:
- Embedding model for retrieval: text-embedding-3-large, BGE, or similar strong embedding models
- LLM for reasoning: a strong instruction-following model such as GPT-4.1-class or similar models

### 5. What is text-to-SQL?

Text-to-SQL is the task of converting a natural language question into a SQL query that can be executed against a database. It is useful for building natural-language interfaces to structured data.

### 6. What are the different types of similarity?

Common similarity measures include:
- Cosine similarity: measures the angle between vectors
- Dot product: measures the magnitude and direction of vectors
- Euclidean distance: measures straight-line distance
- Manhattan distance: measures absolute distance along axes
- Jaccard similarity: measures overlap between sets
- Semantic similarity: measures meaning rather than exact word overlap

### 7. How can we choose a suitable embedding model?

Choose an embedding model based on:
- Task quality requirements
- Language coverage
- Context length
- Cost and latency
- Domain specificity

A good rule is:
- Use a smaller model for speed and cost when the task is simple
- Use a larger or more specialized model when semantic quality matters most

### 8. What re-ranking techniques are suitable?

Re-ranking improves the quality of retrieved results after an initial retrieval step.

Common methods:
- Cross-encoder reranking
- Learned sparse reranking
- LLM-based reranking
- Diversity-aware reranking for reducing redundancy

### 9. How does semantic chunking work?

Semantic chunking splits text based on meaning rather than fixed sizes. It usually groups sentences or paragraphs that belong to the same topic or concept, often using embeddings or similarity scores between adjacent sentences.

This approach usually improves retrieval quality because it preserves context better than arbitrary fixed-size splitting.

### 10. What is self-attention, and what is multi-head self-attention?

Self-attention allows a token to look at other tokens in the same sequence and decide which ones are important for understanding context.

Multi-head self-attention extends this by running several attention mechanisms in parallel. Each head can focus on different relationships, such as syntax, long-range dependency, or topic relevance.

A useful analogy is that different heads learn different “views” of the same input.

A CNN analogy:
- In CNNs, multiple kernels act as different feature detectors.
- They are not identical to attention heads, but both ideas involve projecting the same input into multiple learned representations to capture different patterns.

---

## 3. Agentic AI Systems

### 1. What are the different patterns of Agentic AI systems?

Common agent patterns include:
- Single-agent systems
- Tool-using agents
- Planner-executor agents
- Supervisor-worker systems
- Router-based systems
- Multi-agent debate or collaboration systems
- Workflow-based agents

### 2. What is the difference between a supervisor, an orchestrator, and a router?

- Supervisor: makes high-level decisions and delegates tasks to other agents or tools.
- Orchestrator: coordinates the full workflow and manages execution order.
- Router: selects which path, tool, or agent should handle a request based on the input.

### 3. What is the meaning of stateful and stateless?

- Stateful: the system remembers context, history, or prior interaction state.
- Stateless: each request is processed independently without retaining prior context unless it is explicitly passed in.

### 4. What is the difference between LangChain and LangGraph?

LangChain is a framework for building LLM applications and chaining components such as prompts, tools, retrievers, and memory.

LangGraph is more focused on workflow orchestration and stateful agent execution. It is especially useful when tasks involve multi-step reasoning, branching, and persistent state.

---

## 4. Reinforcement Learning

### 1. What are the main components of reinforcement learning?

The main components are:
- Agent: the learner or decision-maker
- Environment: the world the agent interacts with
- State: the current situation
- Action: what the agent chooses to do
- Reward: feedback from the environment
- Policy: the strategy used to choose actions
- Value function: expected long-term reward from a state or action

### 2. What is multi-objective reinforcement learning?

Multi-objective reinforcement learning deals with problems where an agent must optimize several conflicting objectives at the same time, such as accuracy, cost, safety, and speed.

### 3. What are exploration-exploitation algorithms, and what is the Bellman equation?

Exploration-exploitation is the trade-off between trying new actions and using known good actions.

Common methods include:
- Epsilon-greedy
- Upper Confidence Bound (UCB)
- Thompson sampling

The Bellman equation expresses the value of a state in terms of immediate reward plus future discounted rewards.

It is one of the core mathematical foundations of dynamic programming and reinforcement learning.

### 4. What is the gamma parameter in reinforcement learning?

Gamma, usually written as $\gamma$, is the discount factor. It determines how much future rewards are valued compared to immediate rewards.

- $\gamma$ close to 1 means the agent cares strongly about long-term rewards
- $\gamma$ close to 0 means the agent focuses more on immediate rewards

---

## Final Note

These questions are commonly asked in interviews for machine learning, AI engineering, and data science roles. A strong answer usually combines a clear definition, a practical example, and a real-world application.
