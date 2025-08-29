import asyncio
import logging
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
from shared.nats_client import NATSClient
from semantic_translator import SemanticTranslator
from nmm_analyzer import NMMAnalyzer
from context_manager import ContextManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Semantic Worker", version="1.0.0")

class SemanticRequest(BaseModel):
    session_id: str
    timestamp: float
    frame_id: Optional[str] = None
    gloss_sequence: List[Dict[str, Any]]
    fingerspelling: List[str]
    spatial_roles: Dict[str, Any]
    confidence: float
    processing_time_ms: float
    partial_hypothesis: str
    commit_hypothesis: str

class SemanticResponse(BaseModel):
    session_id: str
    timestamp: float
    frame_id: Optional[str] = None
    translated_text: str
    confidence: float
    processing_time_ms: float
    nmm_features: Dict[str, Any]
    context_info: Dict[str, Any]
    semantic_roles: List[Dict[str, Any]]
    discourse_markers: List[str]
    translation_alternatives: List[Dict[str, Any]]

class SemanticWorker:
    def __init__(self):
        self.nats_client = NATSClient()
        self.semantic_translator = SemanticTranslator(
            model_name="sign-language-transformer",
            max_length=512,
            beam_size=5
        )
        self.nmm_analyzer = NMMAnalyzer()
        self.context_manager = ContextManager(
            max_history=100,
            context_window=10
        )
        
        # Load pre-trained models
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained semantic translation models"""
        try:
            # In production, load actual model weights
            logger.info("Semantic models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            logger.info("Using randomly initialized models for development")
        
    async def start(self):
        """Start the semantic worker"""
        await self.nats_client.connect()
        await self.nats_client.subscribe("sem.translate", self.process_gloss_data, queue="semantic-workers")
        logger.info("Semantic worker started and listening for gloss data")
        
    async def process_gloss_data(self, data: Dict[str, Any]):
        """Process gloss sequences to generate semantic translations"""
        start_time = time.time()
        
        try:
            request = SemanticRequest(**data)
            
            # Update context with new information
            self.context_manager.update_context(
                session_id=request.session_id,
                gloss_sequence=request.gloss_sequence,
                spatial_roles=request.spatial_roles,
                timestamp=request.timestamp
            )
            
            # Get current context
            context_info = self.context_manager.get_context(request.session_id)
            
            # Analyze non-manual markers (placeholder for face landmarks)
            nmm_features = self.nmm_analyzer.analyze_nmm({
                'face_landmarks': [],  # Would come from pose worker
                'body_pose': [],
                'temporal_info': {
                    'timestamp': request.timestamp,
                    'duration': 0.033  # Frame duration
                }
            })
            
            # Perform semantic translation
            translation_result = await self.semantic_translator.translate(
                gloss_sequence=request.gloss_sequence,
                fingerspelling=request.fingerspelling,
                spatial_roles=request.spatial_roles,
                nmm_features=nmm_features,
                context=context_info
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            response = SemanticResponse(
                session_id=request.session_id,
                timestamp=request.timestamp,
                frame_id=request.frame_id,
                translated_text=translation_result.text,
                confidence=translation_result.confidence,
                processing_time_ms=processing_time,
                nmm_features=nmm_features,
                context_info=context_info,
                semantic_roles=translation_result.semantic_roles,
                discourse_markers=translation_result.discourse_markers,
                translation_alternatives=translation_result.alternatives
            )
            
            # Publish to RAG worker for terminology enhancement
            await self.nats_client.publish("rag.enrich", response.dict())
            
            # Also publish to TTS worker for immediate synthesis
            await self.nats_client.publish("tts.speak", {
                "session_id": request.session_id,
                "text": translation_result.text,
                "timestamp": request.timestamp,
                "confidence": translation_result.confidence
            })
            
            logger.debug(f"Processed semantic translation for session {request.session_id} in {processing_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error processing gloss data: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            # Publish error response
            error_response = {
                "session_id": data.get("session_id", "unknown"),
                "timestamp": data.get("timestamp", time.time()),
                "error": str(e),
                "processing_time_ms": processing_time
            }
            await self.nats_client.publish("semantic.error", error_response)

# Global worker instance
worker = SemanticWorker()

@app.on_event("startup")
async def startup_event():
    await worker.start()

@app.on_event("shutdown")
async def shutdown_event():
    await worker.nats_client.disconnect()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "semantic-worker"}

@app.post("/translate")
async def translate_semantic(request: SemanticRequest):
    """HTTP endpoint for semantic translation (alternative to NATS)"""
    await worker.process_gloss_data(request.dict())
    return {"status": "processed", "session_id": request.session_id}

@app.get("/context/{session_id}")
async def get_context(session_id: str):
    """Get current context for a session"""
    context = worker.context_manager.get_context(session_id)
    return {"session_id": session_id, "context": context}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
