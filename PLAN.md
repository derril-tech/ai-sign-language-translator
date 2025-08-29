# PLAN.md

## Product: AI Sign Language Translator (Real-Time Video → Text/Voice + Reverse Avatar)

### Vision & Goals
Enable **real-time, context-aware translation** between sign languages (ASL/BSL/ISL, etc.) and spoken/written languages. Provide accurate streaming captions and TTS speech, while supporting reverse mode (text/voice → signed avatar). Prioritize **accessibility, inclusivity, and privacy**.

### Key Objectives
- Translate sign→text/voice with semantic accuracy, NMM integration, and contextual awareness.
- Reverse translation: text/voice→3D signing avatar with grammar fidelity.
- Support conversation features: turn-taking, partial hypotheses, confidence bands.
- Export transcripts, gloss streams, and session analytics.
- Provide on-device inference by default with optional cloud fallback.

### Target Users
- Deaf/HoH individuals and their hearing counterparts.
- Educators, classrooms, and students for learning and captioning.
- Customer-facing services ensuring accessible interactions.
- Content creators adding sign captions or translations.

### High-Level Approach
1. **Frontend (Next.js 14 + React 18)**  
   - WebRTC for video ingestion.  
   - Pose overlay, gloss strip, confidence captions.  
   - 3D avatar rendering (WebGL/GLTF) for reverse translation.  
   - shadcn/ui + Tailwind for design; Framer Motion for micro-animations.  

2. **Backend (NestJS + Python Workers)**  
   - API Gateway (REST/gRPC, Zod validation, Casbin RBAC, RLS).  
   - Python workers: pose, gloss, semantic, RAG, avatar synth, ASR, TTS, exports.  
   - Event-driven with NATS + Redis Streams.  
   - Postgres + pgvector for sessions, vocab, termbanks.  
   - S3/R2 for avatar rigs, session exports.  

3. **DevOps & Deployment**  
   - Vercel (frontend), Render/Fly/GKE (backend).  
   - Managed Postgres, Redis, NATS.  
   - CI/CD with GitHub Actions.  
   - Monitoring via OpenTelemetry, Prometheus, Grafana, Sentry.  

### Success Criteria
- **Product KPIs**:  
  - ≥80% conversation task success.  
  - ≥4.5/5 comprehension rating both directions.  
  - Weekly active sessions/org ≥3 by week 4.  
  - Quiz/check answers in education ≥85% correctness.  

- **Engineering SLOs**:  
  - Gloss F1 ≥ 0.85, WER ≤ 18%.  
  - Fingerspelling accuracy ≥ 92%.  
  - Latency (sign→voice) ≤ 350 ms p95.  
  - Crash-free session rate ≥ 99.5%.  
