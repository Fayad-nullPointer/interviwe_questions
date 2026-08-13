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

### 9. What do you know about Dataiku (Dataiku DSS)?

Dataiku is an all-in-one platform for data science, analytics, and machine learning. It helps data scientists and business analysts work together in one workspace.

Key features:
- **Visual & Code Interface**: Allows non-coders to use visual drag-and-drop tools while developers can write Python, R, or SQL.
- **Data Preparation**: Connects to databases (like Snowflake or SQL) and cleans data efficiently.
- **AutoML & Deployment**: Quickly builds models automatically and deploys them as API endpoints with one click.
- **MLOps**: Monitors model health, data drift, and governance.
- **Generative AI**: Helps build RAG applications and connect to LLMs like OpenAI or Claude securely.

### 10. What strategies can you use to detect and respond to model drift after deployment?

Model drift happens when a machine learning model's accuracy drops over time because real-world data or patterns have changed.

Types of drift:
- **Data drift**: The input data changes over time (e.g., customer behavior shifts during a holiday).
- **Concept drift**: The relationship between inputs and output changes (e.g., economic shifts change how credit risk is scored).

How to detect it:
- **Statistical tests**: Compare production data against training data using tests like PSI (Population Stability Index) or KS-test.
- **Performance monitoring**: Continuously track metrics (like Accuracy or F1-score) as new actual results arrive.
- **Prediction shifts**: Watch for sudden changes in the model's predicted outputs.

How to respond:
- **Retrain the model**: Automatically retrain the model on newer data when drift passes a threshold.
- **Use fallback rules**: Temporarily switch to simple rules or human review if performance drops significantly.
- **Shadow deployment**: Test the retrained model alongside the current model before replacing it.

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

### 11. How do you save 1,000 PDFs into a vector database?

To efficiently store 1,000 PDFs in a vector database:

1. **Extract text**: Use a PDF parser (like `PyMuPDF` or `pdfplumber`) to extract text and tables. Use OCR for scanned documents.
2. **Clean data**: Remove page headers, footers, and unwanted spaces.
3. **Chunk text**: Split documents into smaller pieces (e.g., 500 tokens with 10% overlap) to keep related text together.
4. **Attach metadata**: Tag each chunk with details like PDF name, page number, and document ID.
5. **Create embeddings**: Convert chunks into vector numbers using an embedding model (like OpenAI or Hugging Face) in batches.
6. **Upload in batches**: Store the vectors and metadata in a vector database (like Pinecone, Qdrant, or Milvus).

### 12. If a user updates content in a source PDF, how do you handle it in a RAG system?

If a PDF is updated, the vector database will contain outdated information, leading to incorrect answers.

How to handle it:
- **Track file hashes**: Save a unique hash (like SHA-256) for each PDF when it is ingested.
- **Detect changes**: When a file is updated, re-hash it to spot changes quickly.
- **Delete old vectors**: Search the vector database for all chunks matching that PDF's ID and delete them.
- **Embed & upload new content**: Chunk and embed only the updated PDF, then save the new vectors.
- **Clear cache**: Clear any cached answers related to that document.

### 13. If a good RAG system suddenly starts hallucinating, how do you debug it?

If a stable RAG system suddenly degrades, troubleshoot it step-by-step:

1. **Check for model/API changes**: Did the LLM provider update default model versions or system prompts?
2. **Check recent document imports**: Did newly uploaded documents contain bad, noisy, or conflicting text?
3. **Isolate Retrieval vs. Generation**:
   - Pass the correct context text directly to the LLM. If the answer is still wrong, the **LLM or prompt** is the issue.
   - If the retrieved chunks do not contain the answer, the **retrieval or embedding** step is failing.
4. **Check vector store health**: Verify if the vector database index is missing data, slow, or returning corrupted results.
5. **Check context window limits**: Are key retrieved facts getting truncated because the context is too long?

### 14. What techniques can you use to manage context window limitations in LLM applications?

To work around token limits when handling large documents or long chats:

- **Selective retrieval (RAG)**: Fetch only the top 3–5 most relevant chunks instead of feeding full documents.
- **Summarization**: Summarize long sections or old chat history before passing context to the model.
- **Sliding window memory**: Keep only the most recent chat messages and save older history in a database.
- **Prompt pruning**: Strip out unnecessary filler words and stop words to save tokens.
- **Use long-context models**: Choose LLMs that natively support large context sizes (e.g., 128k+ tokens).

### 15. How would you determine whether poor RAG performance comes from chunking, embeddings, retrieval, reranking, or generation?

Test each component independently to find the bottleneck:

- **Test Generation (LLM)**: Feed the ground-truth text directly to the LLM. If it still gives a bad response $\rightarrow$ **Generation/Prompt problem**.
- **Test Retrieval**: Check if the top retrieved chunks contain the correct answer. If missing $\rightarrow$ **Retrieval problem**.
- **Test Chunking**: Check if the answer was split in half across chunk boundaries $\rightarrow$ **Chunking size problem**.
- **Test Embeddings**: Compare vector search against simple keyword (BM25) search. If keyword search performs much better $\rightarrow$ **Embedding model problem** (doesn't understand domain terms).
- **Test Reranking**: Compare results before and after reranking. If reranking pushed the right chunk down $\rightarrow$ **Reranker problem**.

### 16. Why might a larger embedding model perform worse than a smaller one for a specific retrieval task?

A larger embedding model is not always better for every task because:

- **Domain mismatch**: A smaller model fine-tuned on your specific topic (e.g., medical or legal code) will beat a large general model.
- **Information dilution**: Large models aggregate semantics across broad contexts, which can smooth out key details needed to find short specific facts.
- **Wrong distance metric**: Using cosine similarity when the model expects dot product or Euclidean distance.
- **Overfitting to noise**: Larger models might capture tiny stylistic or formatting details instead of focusing on core topic meaning.

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

---

## Special Question

1. **How database tables are saved**

- Tables are logical schemas mapped to on-disk storage managed by the DBMS storage engine. Data is stored as rows organized into pages/blocks within files. Indexes (B-tree/LSM) provide fast lookup. A catalog stores table schema/metadata. Writes go through a transaction log (WAL) for durability; background processes flush dirty pages to disk.

2. **How to serve LLMs and tools used**

- Approaches: containerize model + expose HTTP/gRPC endpoints; use model servers (NVIDIA Triton, TorchServe), or frameworks (BentoML, Ray Serve, Hugging Face Inference, LangChain + FastAPI). Or use managed serving (Hugging Face Hub, Vertex AI). Production patterns: batching, request queuing, autoscaling (Kubernetes), model parallelism, quantization, and inference optimization (ONNX, TensorRT).

3. **Why use GPUs instead of CPUs**

- GPUs provide massive SIMD/SIMT parallelism, higher memory bandwidth, and specialized tensor cores for matrix ops. They deliver much higher throughput for dense linear algebra (batch inference/training) than general-purpose CPUs.

4. **Difference between Authentication and Authorization**

- Authentication verifies who the user is (identity). Authorization determines what the authenticated user is allowed to do (permissions).

5. **How tool-calling happens in OpenAI Chat Completions (details)**

- The client registers function schemas (name, params JSON schema) in the request. The model can return a response with a `function_call` payload containing the chosen function name and arguments. The application receives that, executes the corresponding tool/function, and then sends the tool’s result back to the model as a new assistant message, allowing the model to continue the conversation using the tool output.

6. **Detecting when a user ends chat with a bot (customer service)**

- Options: explicit end intent ("bye", "end chat") detected by an intent classifier; inactivity timeout; user clicks "End chat" button; session state in DB and webhook/on-close events; confirmatory message asking if further help is needed. Combine intent detection + timeout + explicit UI action for reliability.

7. **Optimizing read/write when storing user and LLM messages**

- Techniques: write batching and async writes, use a message queue (Kafka/RabbitMQ) and background workers for DB persistence, use caching (Redis) for hot reads, separate write and read stores (append-only event log + read replicas), bulk inserts/updates, proper indexing, TTL for ephemeral data, and contention-reducing patterns (sharding, optimistic concurrency).

8. **(Duplicate) Difference between Authentication and Authorization**

- Same as above: authentication = identity verification; authorization = permission checks.

9. **What is JWT authentication**

- JWT (JSON Web Token) is a compact, self-contained token: header.payload.signature. The payload contains claims (subject, exp, roles). The signature (HMAC or RSA/EC) lets servers verify integrity and authenticity without server-side session storage (stateless), though revocation and refresh mechanisms must be handled.

10. **Difference between webhook and WebSocket**

- Webhook: server-to-server HTTP callback triggered by events (one-way push). Requires the receiver to expose an HTTP endpoint. WebSocket: persistent TCP/HTTP-upgraded bi-directional channel for real-time communication between client and server.

11. **Why Python lacks parallelism and difference with C++ (multithreading details)**

- CPython has the Global Interpreter Lock (GIL) which prevents multiple native threads from executing Python bytecode in parallel in the same process; this simplifies memory management but limits CPU-bound parallelism. Workarounds: multiprocessing (multiple processes), native extensions (C/C++ release GIL), or alternative interpreters. C++ uses OS threads with true concurrent execution, low-level atomic operations, mutexes, lock-free structures, and libraries (pthreads, std::thread, OpenMP). Multi-threaded programming involves shared-memory synchronization (locks, condition variables), race conditions, memory models, and careful resource management.

12. **Difference between encoding and encryption**

- Encoding transforms data into another representation for interoperability (Base64, UTF-8); it’s reversible and not secret. Encryption transforms data to protect confidentiality using keys; without the key the ciphertext should be unreadable.

13. **Common encryption techniques**

- Symmetric: AES (AES-GCM for AEAD). Asymmetric: RSA, ECC (e.g., ECDSA for signatures, ECDH for key exchange). Hybrid schemes combine asymmetric (key exchange) + symmetric (bulk data). Hashing: SHA-2/3, and MAC/HMAC. Protocols: TLS for transport security.

14. **SQL query optimization techniques**

- Add/select proper indexes (including covering/compound indexes), analyze/explain query plans, avoid SELECT *, limit scanned rows, rewrite queries (push predicates, use joins appropriately), use pagination and keyset paging, denormalize or use materialized views when needed, partition large tables, keep statistics up-to-date (ANALYZE), and use caching/read replicas.

15. **How FastAPI works (brief)**

- FastAPI is an ASGI framework built on Starlette. Path operations declare async/sync functions; request bodies and responses are validated and serialized using Pydantic models. It runs on ASGI servers (Uvicorn/Hypercorn) enabling async concurrency. FastAPI auto-generates OpenAPI/Swagger docs and uses dependency injection for request-scoped components.

---

If you want these answers expanded, converted into slides, or added into a separate file, tell me which format to use next.

16. **What is an auth token for login and difference between API key and auth token**

- An auth token is a credential issued after a successful authentication (examples: session token, OAuth access token, JWT). It represents an authenticated identity and often carries claims (user id, scopes, expiry). Tokens are typically short-lived, validated on each request (bearer token in `Authorization` header), and can be revoked or refreshed.

- API key vs Auth token:
	- **API key:** a static credential (often long-lived) tied to a service or client. Used for authenticating machines or services, simple to use but harder to revoke/rotate per-user and offers limited context (no built-in expiry or scopes unless managed by the platform).
	- **Auth token:** issued after authentication, usually short-lived, can carry identity and scopes, and is designed for user sessions and finer-grained access control. Tokens can be JWTs or opaque tokens backed by a session store.

- Best practices: use HTTPS, prefer short-lived tokens with refresh tokens, apply least privilege/scopes, rotate and revoke keys, store secrets securely (server-side vaults), and avoid embedding sensitive keys in public clients.
