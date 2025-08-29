# TODO.md

## Development Roadmap

### Phase 1: Foundations & Infrastructure ✅
- [x] Initialize monorepo (frontend, backend, workers).  
- [x] Set up Next.js 14 project with Tailwind + shadcn/ui; React 18.  
- [x] Initialize NestJS API Gateway (REST/gRPC, Zod validation, RBAC, RLS).  
- [x] Configure Postgres + pgvector, Redis, NATS, and S3/R2 locally with Docker Compose.  
- [x] Implement authentication (SSO/OIDC, JWT, audience scoping).  
- [x] Establish GitHub Actions CI/CD (lint, typecheck, tests).  

### Phase 2: Core Translation Pipeline ✅
- [x] Implement pose-worker for body/hand/face keypoints with smoothing filters.  
- [x] Gloss-worker for CTC decoding, fingerspelling, spatial role labeling.  
- [x] Semantic translator (sem-worker) with NMM fusion, contextual memory.  
- [x] RAG-worker for terminology injection from termbanks.  
- [x] TTS-worker for low-latency streaming synthesis.  
- [x] ASR-worker for speech→text with diarization.  
- [x] Smoothing, confidence scoring, partial/commit hypotheses.  

### Phase 3: Frontend Experience & Reverse Mode ✅
- [x] Build live session page (CameraStage, CaptionRail, GlossStrip, TTSDock).  
- [x] Confidence visualization (color-coded bars).  
- [x] Implement AvatarPanel (WebGL/GLTF) for sign synthesis.  
- [x] Reverse mode pipeline: text→sign grammar planning + avatar curves.  
- [x] TermbankManager: upload/search domain terms.  
- [x] History & Export views: redact names, export PDF/SRT/JSON.  

### Phase 4: Conversation Features, Privacy & Exports ✅
- [x] Implement turn-taking, push-to-talk, clarify/repeat buttons.  
- [x] Conversation memory for co-reference continuity.  
- [x] Terminology manager integration with active packs.  
- [x] Export-worker: transcripts, vocabulary highlights, analytics.  
- [x] Privacy features: blur faces, local-only mode, timed auto-delete.  
- [x] Redaction & GDPR compliance with audit logs.  

### Phase 5: Testing, Accessibility & Deployment ✅
- [x] Unit tests: keypoint normalization, NMM detectors, gloss decoder.  
- [x] Model evaluation: gloss F1, WER, fairness across demographics.  
- [x] Integration tests: sign stream→captions→TTS; reverse mode→avatar.  
- [x] VR/AR simulations for occlusion & low-light.  
- [x] Human-in-loop review panels with Deaf signers.  
- [x] Load/chaos testing: 500 concurrent sessions, jitter, packet loss.  
- [x] Accessibility: captions customization, screen-reader support, dyslexia-friendly fonts.  
- [x] Deploy frontend to Vercel; backend API+workers to GKE/Fly/Render.  
- [x] Configure monitoring dashboards and alerting.  
