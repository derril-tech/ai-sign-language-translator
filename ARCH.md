# ARCH.md

## System Architecture — AI Sign Language Translator

### High-Level Diagram
```
Frontend (Next.js 14 + React 18)
   | WebRTC / gRPC / REST
   v
API Gateway (NestJS)
   | gRPC / NATS
   v
Python Workers (pose, gloss, semantic, rag, avatar synth, asr, tts, export)
   |
   +-- Postgres (pgvector)
   +-- Redis (cache/state)
   +-- NATS (event bus)
   +-- S3/R2 (avatar rigs, session exports)
```

### Frontend (Next.js + React)
- **Live Session UI**: CameraStage (pose overlay), CaptionRail (partial/commit captions), GlossStrip, TTSDock, AvatarPanel.  
- **Controls**: push-to-talk, clarify buttons, terminology packs.  
- **Exports**: TranscriptView for history, export options.  
- **Design**: Tailwind + shadcn/ui with glassmorphism, neon accents.  
- **Accessibility**: high-contrast, large fonts, captions customization.  

### Backend (NestJS)
- REST/gRPC API surface.  
- Zod validation, Problem+JSON error format.  
- RBAC with Casbin, RLS by org_id.  
- Idempotency-Key & Request-ID headers.  
- SSE fallback for partial captions.  

### Workers (Python + FastAPI)
- **pose-worker**: keypoint detection (body, hands, face) with smoothing.  
- **gloss-worker**: gloss decoding, fingerspelling, spatial role labeling.  
- **sem-worker**: semantic translator with NMM fusion + context.  
- **rag-worker**: terminology injection (BM25+vector hybrid).  
- **synth-worker**: avatar planning, animation curves, viseme/NMM.  
- **asr-worker**: speech→text (diarization, punctuation).  
- **tts-worker**: low-latency TTS synthesis.  
- **export-worker**: transcripts, SRT, JSON, analytics.  

### Eventing
- **NATS Topics**: `pose.stream`, `gloss.decode`, `sem.translate`, `rag.enrich`, `sign.synthesize`, `asr.decode`, `tts.speak`, `export.make`.  
- **Redis Streams**: conversation state, progress updates.  

### Data Layer
- **Postgres 16 + pgvector**: orgs, users, sessions, segments, termbanks, terms, avatars, exports.  
- **Redis**: session state, caches, throttling.  
- **S3/R2**: avatar rigs, session exports.  
- **Encryption**: Cloud KMS, per-tenant envelopes.  

### Observability & Security
- **Tracing**: OpenTelemetry.  
- **Metrics**: Prometheus + Grafana.  
- **Errors**: Sentry.  
- **Security**: TLS/HSTS, SSO/OIDC, privacy modes (local-only, blur).  
- **Compliance**: GDPR delete/export, audit logs, dataset governance.  

### DevOps & Deployment
- **Frontend**: Vercel with edge runtime for TTS/ASR proxy.  
- **Backend**: Render/Fly/GKE, autoscaling pools per worker type.  
- **CI/CD**: GitHub Actions (lint, typecheck, integration, deploy).  
- **Storage/CDN**: S3/R2 + CDN for model and avatar asset delivery.  
