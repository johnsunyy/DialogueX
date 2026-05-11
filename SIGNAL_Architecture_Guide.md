# SIGNAL — Complete Platform Architecture & Implementation Guide

### 1. EXECUTIVE OVERVIEW
SIGNAL is a real-time, AI-powered video communication platform designed to eradicate both spoken language and sign language barriers instantly. By seamlessly unifying low-latency audio translation and bi-directional sign language recognition/rendering, it enables fluid, natural conversations between hearing participants speaking different languages and deaf or hard-of-hearing individuals. Targeted at enterprises, educational institutions, healthcare providers, and global teams, SIGNAL provides an unprecedented level of inclusivity. Technically, SIGNAL distinguishes itself from traditional platforms like Zoom or Google Meet by deeply integrating streaming AI inference (Whisper STT, Neural Machine Translation, and MediaPipe/Transformer-based Sign Language models) directly into the WebRTC media pipeline, ensuring end-to-end multi-modal translation with sub-500ms latency without relying on bolt-on third-party bots.

---

### 2. PLATFORM CAPABILITIES MATRIX

| Feature Area | Capabilities |
| :--- | :--- |
| **Video Calling** | 1:1 and Multi-party Group Calls, Screen Sharing, Dynamic Bandwidth Adaptation (Simulcast/SVC), Active Speaker Detection. |
| **Spoken Translation** | Live audio-to-text (captions), live audio-to-translated-audio (TTS), multi-language code-switching support. |
| **Sign Language Recognition** | Live camera input to text/speech. Supports ASL (with architecture to extend to ISL/BSL). Utilizes spatial-temporal Transformer models. |
| **Sign Language Avatar** | Live text/speech to 3D animated signing avatar for deaf participants. |
| **Supported Languages** | English, French, Spanish, German, Hindi, Malayalam, and 50+ spoken languages. ASL as the primary sign language. |
| **Accessibility** | WCAG 2.1 AA compliant UI, high-contrast modes, scalable typography, screen-reader optimized. |
| **Platform Targets** | Web (React/Next.js), Mobile (React Native for iOS/Android), Desktop (Electron-ready). |
| **API & SDK** | REST APIs for room provisioning and user management; WebRTC/WebSocket SDKs for custom client integration. |

---

### 3. HIGH-LEVEL ARCHITECTURE DIAGRAM

The architecture separates signaling from media routing, and taps raw WebRTC streams for real-time AI processing.

```text
                          +-------------------------------------------------+
                          |               CLIENT LAYER                      |
                          |  [ Web (React) ]  [ iOS/Android (React Native) ]|
                          +-----------------------+-------------------------+
                                                  | (HTTPS / WSS / WebRTC)
                                                  v
+----------------+      +---------------------------------------------------+
|  AUTH SERVICE  | <--> |                   API GATEWAY                     |
|  (OAuth2, JWT) |      |                 (Nginx / Kong)                    |
+----------------+      +---------+-------------------+---------------------+
                                  |                   |
                                  v                   v
                        +-------------------+  +--------------------+
                        | SIGNALING SERVER  |  |    MEDIA SERVER    |
                        | (WebSockets/Redis)|  |    (Mediasoup)     |
                        +---------+---------+  +---------+----------+
                                  |                      |
                                  +-----------+----------+
                                              | (Raw Audio/Video Frames via RTP)
                                              v
                        +---------------------------------------------------+
                        |                 AI/ML PIPELINE                    |
                        | +-------------+ +-------------+ +---------------+ |
                        | | Speech-to-  | | Neural Mach.| | Text-to-Speech| |
                        | | Text (STT)  | | Translation | |     (TTS)     | |
                        | +-------------+ +-------------+ +---------------+ |
                        | +-------------+ +-------------+                   |
                        | | Sign Lang.  | | Sign Avatar |                   |
                        | | Recognition | | Rendering   |                   |
                        | +-------------+ +-------------+                   |
                        +----------------------+----------------------------+
                                               |
                                               v
                        +---------------------------------------------------+
                        |                 DATA & STATE                      |
                        |  [ PostgreSQL ]   [ Redis Pub/Sub ]   [ S3/CDN ]  |
                        +---------------------------------------------------+
```

---

### 4. DETAILED COMPONENT BREAKDOWN

**a) Frontend Web Client**
*   **Purpose:** Primary user interface for desktop and mobile browsers.
*   **Internal:** React functional components, WebRTC peer connection manager, local media track processors, canvas-based sign language overlay, Zustand store.
*   **Interfaces:** REST (API Gateway), WebSocket (Signaling), WebRTC (Media Server).
*   **Dependencies:** Next.js, Mediasoup-client, MediaPipe (WASM), Three.js.
*   **Failure Modes:** Network drop (triggers ICE restart), camera denied (graceful fallback to audio-only/text-only).

**b) Mobile Clients**
*   **Purpose:** Native applications for iOS and Android.
*   **Internal:** React Native bridging to native WebRTC libraries.
*   **Interfaces:** REST, WebSocket, WebRTC.
*   **Dependencies:** React Native, react-native-webrtc, Mediasoup-client.
*   **Failure Modes:** Backgrounding drops camera (disables video track, continues audio/captions).

**c) API Gateway**
*   **Purpose:** Single entry point for HTTP/WS requests. Handles rate limiting, SSL termination, and routing.
*   **Internal:** Reverse proxy rules, caching layer, rate limiter.
*   **Interfaces:** HTTPS, WSS.
*   **Dependencies:** Kong or Nginx.
*   **Failure Modes:** Instance crash (Load balancer routes to healthy node).

**d) Auth & Identity Service**
*   **Purpose:** Manages user registration, login, JWT issuance, and SSO.
*   **Internal:** Password hashing (Argon2), token generation, OAuth2 callback handlers.
*   **Interfaces:** REST (`/auth/*`).
*   **Dependencies:** PostgreSQL, Redis (session blocklist).
*   **Failure Modes:** DB unreachable (caches valid tokens in Redis to maintain active sessions).

**e) Signaling Server**
*   **Purpose:** Exchanges WebRTC SDP offers/answers and ICE candidates. Broadcasts room events.
*   **Internal:** WebSocket room manager, Redis Pub/Sub adapter for multi-node scaling.
*   **Interfaces:** WSS.
*   **Dependencies:** Node.js, Socket.io, Redis.
*   **Failure Modes:** Server restart (clients reconnect seamlessly and re-join room via Redis state).

**f) Media Server**
*   **Purpose:** Routes video and audio streams (SFU topology). **Chosen: Mediasoup** because its low-level C++ worker architecture is highly efficient, controllable via Node.js, and natively supports `PlainTransport` which is critical for directly extracting/injecting raw RTP packets to and from the AI/ML pipeline without client-side relay.
*   **Internal:** Worker threads, routers, producers, consumers.
*   **Interfaces:** WebRTC (SRTP/DTLS).
*   **Dependencies:** Mediasoup C++ workers, libuv.
*   **Failure Modes:** High CPU (mitigated by horizontal scaling via worker distribution across cores).

**g) Speech-to-Text Engine**
*   **Purpose:** Transcribes live audio to text in real-time.
*   **Internal:** Streaming VAD (Voice Activity Detection), audio chunking, inference runner.
*   **Interfaces:** gRPC from Media Server.
*   **Dependencies:** faster-whisper (CTranslate2).
*   **Failure Modes:** GPU out of memory (fallback to cloud API like Google STT).

**h) Language Translation Engine**
*   **Purpose:** Translates transcribed text to target language(s).
*   **Internal:** Prompt queuing, context caching, terminology injection.
*   **Interfaces:** REST/gRPC.
*   **Dependencies:** DeepL API (or self-hosted NMT).
*   **Failure Modes:** API rate limit (exponential backoff, fallback to local MarianMT).

**i) Text-to-Speech Engine**
*   **Purpose:** Converts translated text back to audio for the listener.
*   **Internal:** Voice cloning/mapping, streaming audio stream generator.
*   **Interfaces:** gRPC to Media Server (injects as audio track via `PlainTransport`).
*   **Dependencies:** ElevenLabs API or Coqui TTS.
*   **Failure Modes:** Generation lag (drops late packets, falls back to text captions only).

**j) Sign Language Recognition Engine**
*   **Purpose:** Translates video of signing into text.
*   **Internal:** MediaPipe holistic landmark extraction (often client-side to save bandwidth), temporal sequence windowing (server-side), spatial-temporal Transformer/LSTM model inference.
*   **Interfaces:** WebSocket (landmark data in) -> gRPC (text out).
*   **Dependencies:** Python (FastAPI), PyTorch, ONNX Runtime.
*   **Failure Modes:** Poor lighting/occlusion (notifies user via UI alert regarding confidence drop).

**k) Sign Language Avatar Rendering Engine**
*   **Purpose:** Converts text/speech to an animated sign language 3D avatar.
*   **Internal:** Text parser, SMPL/Custom rig animator, motion blending dictionary (FBX animations).
*   **Interfaces:** WebGL on client.
*   **Dependencies:** Three.js.
*   **Failure Modes:** Missing word in motion dictionary (falls back to procedural fingerspelling).

**l) Real-time Sync & State Management**
*   **Purpose:** Keeps distributed components in sync (e.g., who is in which room, active speaker).
*   **Internal:** Redis Pub/Sub, sorted sets for queues.
*   **Interfaces:** Redis protocol.
*   **Dependencies:** Redis cluster.
*   **Failure Modes:** Network partition (Sentinel failover).

**m) Recording & Playback Service**
*   **Purpose:** Mixes audio/video/captions into MP4s.
*   **Internal:** GStreamer/FFmpeg mixing pipelines, joining as a headless Mediasoup consumer.
*   **Interfaces:** REST (`/recordings`).
*   **Dependencies:** FFmpeg, AWS S3.
*   **Failure Modes:** Encoder crash (restarts job from raw stream dumps).

**n) Notification Service**
*   **Purpose:** Sends meeting invites, missed call alerts.
*   **Internal:** Email templates, Push notification queues.
*   **Interfaces:** REST.
*   **Dependencies:** SendGrid, FCM/APNs.
*   **Failure Modes:** Upstream API down (queue locally and retry).

**o) Analytics & Monitoring Service**
*   **Purpose:** Tracks call quality (packet loss, jitter), translation latency, user engagement.
*   **Internal:** Time-series aggregation.
*   **Interfaces:** UDP/TCP logs.
*   **Dependencies:** Prometheus, Grafana, OpenTelemetry.
*   **Failure Modes:** High disk I/O (log rotation and metric sampling).

**p) Admin Dashboard**
*   **Purpose:** Internal tool for managing users, monitoring server health, reviewing logs.
*   **Internal:** React SPA.
*   **Interfaces:** REST API.
*   **Dependencies:** Next.js admin frontend.
*   **Failure Modes:** Unauthorized access (strict RBAC middleware).

---

### 5. FULL TECH STACK

**Frontend Layer**
*   **Framework:** Next.js 14 (React 18). *Why:* SSR for SEO, unified routing, vast ecosystem. *Tradeoffs:* Slightly heavier initial bundle than vanilla Vite.
*   **State Management:** Zustand. *Why:* Lightweight, zero boilerplate compared to Redux, handles rapid WebRTC state changes well. *Tradeoffs:* Less structured middleware ecosystem than Redux.
*   **Styling:** Tailwind CSS. *Why:* Rapid UI development, strict design system enforcement. *Tradeoffs:* HTML class clutter.
*   **Real-time & Media:** Socket.io-client (Signaling), Mediasoup-client v3 (WebRTC).

**Backend Layer**
*   **Language/Framework:** Node.js v20 (TypeScript) with Express/Fastify. *Why:* Node's event-driven architecture is ideal for high-concurrency WebSocket signaling. *Tradeoffs:* Single-threaded CPU binding (solved via worker threads).
*   **AI Services Framework:** Python 3.11 with FastAPI. *Why:* De facto standard for ML inference. *Tradeoffs:* GIL limitations (solved via multi-processing and ONNX).

**AI/ML Layer**
*   **Models:** faster-whisper (STT), DeepL API (NMT), Coqui TTS/ElevenLabs (TTS), Custom Transformer (SLR).
*   **Inference Runtime:** ONNX Runtime / PyTorch 2.x. *Why:* ONNX provides significant speedups for Transformer inference. *Tradeoffs:* Model export complexity.
*   **Hosting:** AWS EC2 G5 instances (NVIDIA A10G/T4).

**Databases**
*   **Primary:** PostgreSQL 16. *Why:* ACID compliance, relational integrity for billing/users. *Tradeoffs:* Harder to horizontally scale writes than NoSQL.
*   **Cache/PubSub:** Redis 7.2. *Why:* Sub-millisecond latency for WebRTC state and signaling across nodes. *Tradeoffs:* Memory bound.
*   **Time-series/Vector (Optional):** pgvector (for translation memory). *Why:* Keeps stack simple by leveraging existing Postgres infrastructure.

**Infrastructure & DevOps**
*   **Cloud Provider:** AWS (EKS, RDS, ElastiCache, S3). *Why:* Maturity, global regions for low WebRTC latency. *Tradeoffs:* Vendor lock-in, complex pricing.
*   **Orchestration:** Kubernetes (EKS) + Docker. *Why:* Seamless scaling of microservices. *Tradeoffs:* Steep learning curve.
*   **CI/CD:** GitHub Actions -> AWS ECR -> Helm. *Why:* Integrated directly with source control.

**Monitoring**
*   **Stack:** Prometheus, Grafana, OpenTelemetry, ELK Stack (Logs). *Why:* Industry standard for microservices observability. *Tradeoffs:* Resource heavy to run.

**Security & CDN**
*   **Auth:** JWT (Access) + HttpOnly Cookies (Refresh). *Why:* Stateless, cross-domain friendly.
*   **Secrets:** AWS Secrets Manager + External Secrets Operator.
*   **CDN & Media Delivery:** Cloudflare (for static assets and Avatar 3D models).

---

### 6. COMPLETE DATA FLOW DIAGRAMS

**Flow A: User Joins a Call**
1. **Client** calls `GET /calls/:id/join` on **API Gateway**.
2. **Auth Service** validates JWT.
3. **Client** opens WebSocket to **Signaling Server**.
4. **Signaling Server** returns Mediasoup router RTP capabilities.
5. **Client** creates a WebRTC `SendTransport` and sends `dtlsParameters` to **Media Server**.
6. **Client** begins capturing webcam/mic, producing tracks to the **Media Server**.
7. **Signaling Server** broadcasts `peer:connected` to other users in the room.
8. Other peers create `RecvTransport` and request to consume the new tracks.
9. **Media Server** routes streams to consumers. Video appears.

**Flow B: Spoken Language Translation in a Live Call**
1. User A speaks French. Audio sent via WebRTC to **Media Server**.
2. **Media Server** routes a copy of the raw audio via `PlainTransport` to the **AI/ML Pipeline (Python)**.
3. **STT Engine** detects speech (VAD), chunks it, and transcribes to French text (<200ms latency budget).
4. **Language Translation Engine** translates French text to English text (<100ms).
5. **Signaling Server** broadcasts English text as a `translation:ready` WebSocket event. User B sees captions.
6. **TTS Engine** generates English audio from text (<200ms).
7. **Media Server** receives generated audio and injects it as a WebRTC audio track to User B. User B hears English.

**Flow C: Sign Language Input Translation**
1. User A (Deaf) signs at the camera.
2. **Frontend Client** extracts MediaPipe holistic landmarks locally via WASM (~15ms/frame) to save bandwidth.
3. **Frontend Client** batches landmarks and sends them via WebSocket to the **AI/ML Pipeline**.
4. **Sign Language Recognition Engine** buffers frames (e.g., 60-frame window) and runs Transformer inference, outputting text "Hello" (<50ms).
5. Text "Hello" is broadcasted via **Signaling Server** to User B (Caption).
6. **TTS Engine** converts "Hello" to audio.
7. **Media Server** injects audio into the call. Hearing User B hears the spoken word.

**Flow D: Sign Language Avatar Output**
1. Hearing User B speaks English. Audio is converted to English text via the **STT Engine**.
2. English text is broadcasted to User A via **Signaling Server**.
3. **Frontend Client** (User A) receives text, parses it, and maps words/phonemes to an animation dictionary.
4. **Sign Language Avatar Rendering Engine** (Three.js client-side) blends motion clips and renders the 3D avatar signing the phrase.

**Flow E: Call Recording with Translated Captions**
1. Host triggers `call:record` WebSocket event.
2. **Recording Service** joins the Mediasoup router as a hidden headless consumer.
3. **Recording Service** dumps raw RTP streams to disk.
4. Simultaneously, **Signaling Server** routes translation text events to **Recording Service**.
5. Post-call, **Recording Service** runs FFmpeg to mux video, audio, and generate `.vtt` subtitles.
6. Output MP4 is uploaded to S3. URL is saved to **PostgreSQL**.

**Flow F: Authentication & Session Start**
1. User submits login form.
2. `POST /auth/login` routed through **API Gateway** to **Auth Service**.
3. **Auth Service** queries **PostgreSQL**, verifies Argon2 password hash.
4. Generates short-lived JWT (body) and long-lived refresh token (HttpOnly Secure Cookie).
5. **Client** stores JWT in memory.
6. **Client** opens WSS connection, passing JWT in the connection payload. **Signaling Server** validates JWT before upgrading.

---

### 7. DATABASE SCHEMA

**PostgreSQL (Primary)**
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    preferred_spoken_language VARCHAR(10) DEFAULT 'en',
    preferred_sign_language VARCHAR(10) DEFAULT 'asl',
    accessibility_mode BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    host_id UUID REFERENCES users(id),
    title VARCHAR(255),
    scheduled_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'waiting', -- waiting, active, ended
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE call_participants (
    call_id UUID REFERENCES calls(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id),
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    left_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (call_id, user_id)
);

CREATE TABLE recordings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id UUID REFERENCES calls(id) ON DELETE CASCADE,
    s3_url TEXT NOT NULL,
    duration_seconds INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(255) NOT NULL,
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Redis**
*   `session:{userId}` (String): Stores JSON `{ "token": "...", "device": "..." }`. TTL: 24h.
*   `room:{callId}:participants` (Set): Contains `userId`s currently in the room.
*   `room:{callId}:webrtc_state` (Hash): Stores Mediasoup router capabilities and active producer IDs.
*   `translation_cache:{srcLang}:{tgtLang}:{textHash}` (String): Translated text. TTL: 30d.
*   `rate_limit:{ip}` (Integer): Token bucket counter. TTL: 1m.

**Vector DB (pgvector - Optional)**
*   Table `translation_memory`: Stores `id`, `source_text`, `translated_text`, `embedding (vector(1536))` for semantic caching of complex organizational phrases.

---

### 8. API REFERENCE (Core Endpoints)

**Auth**
*   `POST /auth/login`
    *   Auth: None
    *   Request: `{"email": "user@ex.com", "password": "..."}`
    *   Response: `{"accessToken": "eyJ...", "user": {"id": "..."}}`
*   `POST /auth/refresh`
    *   Auth: HttpOnly Cookie
    *   Response: `{"accessToken": "eyJ..."}`
*   `POST /auth/logout`
    *   Auth: Bearer JWT. Clears cookies and deletes Redis session.

**Users**
*   `GET /users/me`
    *   Auth: Bearer JWT
    *   Response: User object including language preferences.
*   `PATCH /users/settings`
    *   Auth: Bearer JWT
    *   Request: `{"preferred_spoken_language": "fr", "accessibility_mode": true}`

**Calls**
*   `POST /calls/create`
    *   Auth: Bearer JWT
    *   Request: `{"title": "Weekly Sync", "scheduled_at": "2026-05-06T10:00:00Z"}`
    *   Response: `{"callId": "uuid"}`
*   `POST /calls/:id/join`
    *   Auth: Bearer JWT
    *   Response: `{"signalingUrl": "wss://...", "roomConfig": {...}}`

**Recordings**
*   `GET /recordings/:callId`
    *   Auth: Bearer JWT
    *   Response: `[{"id": "uuid", "s3_url": "https...", "duration_seconds": 3600}]`

**Admin**
*   `GET /admin/calls/active`
    *   Auth: Bearer JWT (Role: Admin)
    *   Response: `[{"callId": "...", "participantCount": 4, "serverNode": "media-1"}]`

---

### 9. WEBSOCKET & REAL-TIME EVENTS PROTOCOL

All payloads are JSON wrapped in standard Socket.io event emissions.

| Event Name | Direction | Payload Schema | When it fires |
| :--- | :--- | :--- | :--- |
| `call:join` | C -> S | `{ "callId": "uuid", "preferences": {...} }` | Client successfully opens WebSocket and requests room entry. |
| `peer:connected` | S -> C | `{ "peerId": "uuid", "displayName": "...", "audio": bool }` | A new user joins the room. |
| `peer:disconnected`| S -> C | `{ "peerId": "uuid" }` | A user drops connection or leaves explicitly. |
| `media:produce` | C -> S | `{ "kind": "video\|audio", "rtpParameters": {...} }` | Client begins sending webcam or mic data. |
| `media:consume` | C -> S | `{ "producerId": "uuid" }` | Client requests to receive a specific media track. |
| `translation:ready`| S -> C | `{ "sourceId": "uuid", "text": "...", "lang": "en", "isFinal": true }`| The ML pipeline has generated a translated text chunk. |
| `caption:update` | S -> C | `{ "sourceId": "uuid", "text": "..." }` | Partial (non-final) text for live typing effect. |
| `sign:detect` | C -> S | `{ "landmarks": [ {x, y, z}, ... ], "timestamp": 1234 }` | Client pushes MediaPipe frame data to ML pipeline. |
| `avatar:frame` | S -> C | `{ "animationId": "uuid", "blendWeight": 0.8 }` | Server instructs Avatar to play specific motion dictionary item. |
| `call:ended` | S -> C | `{ "reason": "host_ended\|timeout" }` | Host terminates the room. |
| `error:fatal` | S -> C | `{ "code": 500, "message": "..." }` | Unrecoverable server error. |

---

### 10. AI/ML PIPELINE — DEEP DIVE

**STT (Speech-to-Text)**
*   **Architecture:** Whisper large-v3 optimized with `faster-whisper` (CTranslate2 backend) supporting INT8 quantization.
*   **Input/Output:** 16kHz mono PCM audio -> JSON (Text, Language, Confidence, Timestamps).
*   **Latency Target:** < 250ms.
*   **Real-time Strategy:** Uses WebRTC VAD (Voice Activity Detection) to slice audio into logical phonetic chunks (e.g., 1-2 seconds) rather than waiting for silence. Overlapping windows ensure context is preserved.
*   **Hosting:** Cloud GPU (NVIDIA A10G/T4).

**NMT (Neural Machine Translation)**
*   **Architecture:** DeepL API (primary for quality) or custom MarianMT/Helsinki-NLP model (fallback/privacy).
*   **Input/Output:** Source Text + Target Lang ID -> Translated Text.
*   **Latency Target:** < 100ms.
*   **Real-time Strategy:** Translates on "isFinal: true" STT events to prevent jarring grammatical rewrites mid-sentence.

**TTS (Text-to-Speech)**
*   **Architecture:** ElevenLabs WebSocket streaming API (or Coqui TTS for local).
*   **Input/Output:** Translated Text -> 24kHz PCM audio stream.
*   **Latency Target:** < 200ms TTFB (Time to First Byte).
*   **Real-time Strategy:** Streaming inference. Audio bytes are piped directly into Mediasoup `PlainTransport` and timestamp-aligned with the original speaker's video track to maintain lip-sync illusions where possible.

**SLR (Sign Language Recognition)**
*   **Architecture:** Dual-model. A spatial-temporal Transformer handles continuous word-level signing, falling back to a lightweight LSTM for alphabet fingerspelling.
*   **Input/Output:** Array of normalized MediaPipe holistic landmarks (pose, hands, face) -> Text Class (e.g., "HELLO").
*   **Latency Target:** < 50ms per window.
*   **Real-time Strategy:** Client-side WASM extracts landmarks to save upstream bandwidth. Server maintains a rolling 60-frame buffer. Inference runs every 15 frames (stride).
*   **Model Update:** Continuous fine-tuning pipeline utilizing user corrections (opt-in).

**SLA (Sign Language Avatar)**
*   **Architecture:** Procedural animation blending. Text is tokenized and mapped to a dictionary of high-quality FBX motion capture clips.
*   **Input/Output:** Text Token -> Array of SMPL bone rotations.
*   **Latency Target:** < 50ms (Compute).
*   **Real-time Strategy:** Runs 100% client-side. The server sends token arrays. The client (Three.js) loads GLTF avatars and crossfades animations seamlessly. Fingerspelling is generated procedurally if a word lacks a dictionary entry.

---

### 11. INFRASTRUCTURE & DEVOPS

*   **Cloud Provider:** AWS. Deployed in multi-region active-active setup (e.g., `us-east-1` and `eu-west-1`) to ensure WebRTC latency remains < 100ms globally.
*   **Kubernetes Cluster (EKS):**
    *   **Namespaces:** `ingress-nginx`, `signal-prod`, `signal-ml`, `monitoring`.
    *   **Deployments:** Stateless `api-gateway`, `auth`, and `signaling` pods.
    *   **DaemonSets:** Media Server (Mediasoup requires host networking for optimal UDP port allocation).
    *   **NodeGroups:** CPU nodes for Web/Signaling, GPU nodes (p3/g5) dynamically scaled for `signal-ml`.
*   **Docker Strategy:** Distroless base images or Alpine (`node:20-alpine`, `python:3.11-slim`) for minimal attack surface.
*   **CI/CD Pipeline:** GitHub Actions triggers on `main`.
    1.  Lint (ESLint/Flake8) & Unit Test (Jest/PyTest).
    2.  Build Docker Images.
    3.  Push to AWS ECR.
    4.  Update Helm charts -> ArgoCD synchronizes EKS state.
*   **Environments:** `dev` (feature branches), `staging` (pre-release copy of prod), `prod` (live).
*   **Secrets Management:** AWS Secrets Manager synced to K8s via External Secrets Operator. No secrets in Git.
*   **Autoscaling:**
    *   CPU Pods (HPA): Scale if CPU > 70% or Memory > 80%.
    *   Media Servers: Custom metric scaling based on active WebRTC transports.
*   **Disaster Recovery (DR) & Backup:**
    *   RDS Automated Backups (Point-in-time recovery up to 35 days).
    *   Cross-region replication for S3 buckets (Recordings).
    *   Infrastructure as Code (Terraform) allows full cluster rebuild in < 20 minutes in a new region.

---

### 12. SECURITY ARCHITECTURE

*   **Threat Model:** Primary threats are unauthorized room joining (Zoombombing), eavesdropping on unencrypted media, and DDoS on the signaling layer.
*   **Encryption in Transit:**
    *   Web/API: TLS 1.3 only.
    *   WebRTC: Media streams are end-to-end encrypted between the client and the Mediasoup SFU using SRTP with DTLS key exchange.
*   **Encryption at Rest:** AES-256 for PostgreSQL volumes (RDS) and S3 buckets containing call recordings.
*   **Auth Security:** JWT access tokens expire in 15 minutes. Refresh tokens are rotating, device-bound, and stored in HttpOnly, Secure, SameSite=Strict cookies.
*   **Rate Limiting:** Redis token bucket per IP and per User ID at the API Gateway to prevent brute force and DDoS.
*   **Input Validation:** Strict JSON schema validation (Zod in Node, Pydantic in Python) on all API and WebSocket payloads to prevent injection.
*   **Compliance:**
    *   **GDPR:** Automated data retention policies, `/users/me/delete` endpoint for right-to-be-forgotten.
    *   **HIPAA:** Architecture supports a "Zero-Persistence" mode where recordings, transcripts, and logs are disabled at the room level for PHI compliance.
    *   **WCAG 2.1 AA:** Frontend tested with Axe and VoiceOver/NVDA.
*   **Penetration Testing Plan:** Bi-annual third-party gray-box network and application pen tests.

---

### 13. PERFORMANCE & SCALABILITY DESIGN

*   **Expected Load:**
    *   Launch: 1,000 concurrent users.
    *   6 Months: 10,000 concurrent users.
    *   12 Months: 50,000 concurrent users.
*   **Latency SLAs:**
    *   Web API: < 100ms.
    *   Signaling (WS): < 50ms.
    *   Media (WebRTC RTT): < 150ms.
    *   End-to-end Translation Pipeline: < 800ms.
*   **Bottlenecks & Mitigation:**
    *   *Bottleneck:* Mediasoup router CPU saturation. *Mitigation:* Mediasoup workers map 1:1 to CPU cores. The signaling server load balances rooms across a fleet of EC2 instances. If a room exceeds 500 users, it cascades routers across multiple servers (PipeTransports).
    *   *Bottleneck:* AI GPU Inference Queue. *Mitigation:* ML pipeline is stateless. RabbitMQ/Redis queues distribute chunked media to the largest available GPU pool.
*   **CDN Strategy:** Cloudflare caches the heavy 3D GLTF Avatar assets, motion dictionaries, and compiled WASM files (MediaPipe), ensuring rapid client load times.

---

### 14. FRONTEND COMPONENT ARCHITECTURE

*   **Folder Structure (Next.js):**
    *   `/app`: Next.js 14 App Router pages (`/dashboard`, `/room/[id]`).
    *   `/components`: Reusable UI (`Button`, `Modal`).
    *   `/features/room`: Domain-specific components (`VideoGrid`, `Controls`, `CaptionOverlay`).
    *   `/features/avatar`: 3D rendering components (`AvatarCanvas`, `AvatarModel`).
    *   `/hooks`: Custom React hooks (`useWebRTC`, `useSignaling`, `useMediaPipe`).
    *   `/store`: Zustand state slices.
*   **Core Components:**
    *   `RoomProvider`: Context wrapper that initializes Mediasoup and Socket.io.
    *   `VideoTile`: Renders the `<video>` element, attaches the WebRTC MediaStream, and handles UI overlays (mute icons).
    *   `CaptionOverlay`: Virtualized list displaying real-time text arrays fed from Zustand.
    *   `SignLanguageOverlay`: Renders the MediaPipe landmark skeleton over local video for debugging/feedback.
*   **State Management Approach:** Zustand manages the `RoomStore` (peers, producers, consumers, transcripts). It avoids React Context re-render hell for rapidly changing WebRTC states.
*   **Avatar Rendering:** Utilizes `@react-three/fiber`. The `<AvatarCanvas />` listens to `translation:ready` events, maps the text to an `animationId`, and triggers `useAnimations` to play the clip.

---

### 15. IMPLEMENTATION ROADMAP

**Phase 1: Core Video Calling**
*   **Duration:** 2 Months.
*   **Team:** 2 Fullstack Engineers, 1 DevOps.
*   **Deliverables:** Auth, Room Provisioning, WebRTC SFU integration (Mediasoup), basic video/audio UI.
*   **Technical Risks:** STUN/TURN configuration complexities for restrictive corporate firewalls.

**Phase 2: Spoken Language Translation**
*   **Duration:** 2 Months.
*   **Team:** Core Team + 1 AI Engineer.
*   **Deliverables:** VAD chunking, Python gRPC integration, Whisper STT, DeepL NMT, ElevenLabs TTS pipeline.
*   **Technical Risks:** Managing audio sync and managing latency accumulation across three sequential AI models.

**Phase 3: Sign Language Input (SLR)**
*   **Duration:** 2 Months.
*   **Team:** Core Team + 1 ML Researcher.
*   **Deliverables:** WASM MediaPipe client integration, Transformer model deployment, WebSocket landmark streaming.
*   **Technical Risks:** High false-positive rates on conversational signing; handling variable frame rates from different client cameras.

**Phase 4: Sign Language Avatar Output (SLA)**
*   **Duration:** 2 Months.
*   **Team:** Core Team + 1 3D Graphics Engineer.
*   **Deliverables:** 3D asset pipeline, Three.js rendering engine, dictionary-based motion blending logic.
*   **Technical Risks:** "Uncanny valley" effect, missing vocabulary handling.

**Phase 5: Enterprise, Scaling, & Analytics**
*   **Duration:** 2 Months.
*   **Team:** Full Team.
*   **Deliverables:** Server-side recording (FFmpeg), Admin Dashboard, advanced analytics (Grafana), API SDKs.
*   **Technical Risks:** CPU saturation during concurrent HD recordings.

---

### 16. GLOSSARY

*   **ASL:** American Sign Language.
*   **BSL:** British Sign Language.
*   **DTLS (Datagram Transport Layer Security):** Communications protocol that provides security for datagram-based applications, used heavily in WebRTC.
*   **ICE (Interactive Connectivity Establishment):** A framework used by WebRTC to find the best path to connect peers (traversing NATs and Firewalls).
*   **ISL:** Indian Sign Language.
*   **Mediasoup:** A highly scalable, low-level C++ based WebRTC SFU library for Node.js.
*   **NMT (Neural Machine Translation):** AI models used to translate text from one language to another.
*   **SDP (Session Description Protocol):** A standard format for describing multimedia communication sessions for the purposes of session announcement and invitation.
*   **SFU (Selective Forwarding Unit):** A WebRTC architecture where each participant sends their media stream to a central server, which then forwards it to all other participants, optimizing bandwidth compared to mesh networks.
*   **SLA (Sign Language Avatar):** A 3D animated character driven by text or speech.
*   **SLR (Sign Language Recognition):** The computer vision process of translating visual gestures into text.
*   **SRTP (Secure Real-time Transport Protocol):** A security profile for RTP that adds confidentiality, message authentication, and replay protection.
*   **STT (Speech-to-Text):** Automatic transcription of audio into text.
*   **TTS (Text-to-Speech):** Artificial generation of human speech from text.
*   **VAD (Voice Activity Detection):** An algorithm used to detect the presence or absence of human speech in an audio stream.
*   **WASM (WebAssembly):** A binary instruction format that allows code written in languages like C++ or Rust (e.g., MediaPipe) to run at near-native speed in web browsers.
*   **WebRTC (Web Real-Time Communication):** An open-source project and API standard that enables real-time voice, video, and data communication directly between web browsers and devices.
