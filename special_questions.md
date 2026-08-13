## Special Question

1. **How database tables are saved**

- Tables are logical schemas mapped to on-disk storage managed by the DBMS storage engine. Data is stored as rows organized into pages/blocks within files. Indexes (B-tree/LSM) provide fast lookup. A catalog stores table schema/metadata. Writes go through a transaction log (WAL) for durability; background processes flush dirty pages to disk.

2. **How to serve LLMs and tools used**

- Approaches: containerize model + expose HTTP/gRPC endpoints; use model servers (NVIDIA Triton, TorchServe), or frameworks (BentoML, Ray Serve, Hugging Face Inference, LangChain + FastAPI). Or use managed serving (Hugging Face Hub, Vertex AI). Production patterns: batching, request queuing, autoscaling (Kubernetes), model parallelism, quantization, and inference optimization (ONNX, TensorRT).

3. **Why use GPUs instead of CPUs**

- GPUs provide massive SIMD/SIMT parallelism, higher memory bandwidth, and specialized tensor cores for matrix ops. They deliver much higher throughput for dense linear algebra (batch inference/training) than general-purpose CPUs.

4. **Difference between Authentication and Authorization**

- Authentication verifies who the user is (identity). Authorization determines what the authenticated user is allowed to do (permissions).

5. **How tool-calling happens in OpenAI Chat Completions (details)**n
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

16. **What is an auth token for login and difference between API key and auth token**

- An auth token is a credential issued after a successful authentication (examples: session token, OAuth access token, JWT). It represents an authenticated identity and often carries claims (user id, scopes, expiry). Tokens are typically short-lived, validated on each request (bearer token in `Authorization` header), and can be revoked or refreshed.

- API key vs Auth token:
  - **API key:** a static credential (often long-lived) tied to a service or client. Used for authenticating machines or services, simple to use but harder to revoke/rotate per-user and offers limited context (no built-in expiry or scopes unless managed by the platform).
  - **Auth token:** issued after authentication, usually short-lived, can carry identity and scopes, and is designed for user sessions and finer-grained access control. Tokens can be JWTs or opaque tokens backed by a session store.

- Best practices: use HTTPS, prefer short-lived tokens with refresh tokens, apply least privilege/scopes, rotate and revoke keys, store secrets securely (server-side vaults), and avoid embedding sensitive keys in public clients.
