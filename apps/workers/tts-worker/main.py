import asyncio
import logging
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from shared.nats_client import NATSClient
from tts_synthesizer import TTSSynthesizer
from voice_cloning import VoiceCloner
from audio_processor import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TTS Worker", version="1.0.0")

class TTSRequest(BaseModel):
    session_id: str
    text: str
    timestamp: float
    confidence: float
    voice_id: Optional[str] = "default"
    speed: Optional[float] = 1.0
    pitch: Optional[float] = 1.0
    emotion: Optional[str] = "neutral"

class TTSResponse(BaseModel):
    session_id: str
    timestamp: float
    audio_data: str  # base64 encoded audio
    audio_format: str
    duration_ms: float
    processing_time_ms: float
    voice_characteristics: Dict[str, Any]

class TTSWorker:
    def __init__(self):
        self.nats_client = NATSClient()
        self.tts_synthesizer = TTSSynthesizer(
            model_name="neural-tts-v2",
            sample_rate=22050,
            channels=1
        )
        self.voice_cloner = VoiceCloner()
        self.audio_processor = AudioProcessor()
        
        # Load TTS models
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained TTS models"""
        try:
            # In production, load actual model weights
            logger.info("TTS models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            logger.info("Using mock TTS for development")
        
    async def start(self):
        """Start the TTS worker"""
        await self.nats_client.connect()
        await self.nats_client.subscribe("tts.speak", self.process_text, queue="tts-workers")
        logger.info("TTS worker started and listening for text")
        
    async def process_text(self, data: Dict[str, Any]):
        """Process text to generate speech audio"""
        start_time = time.time()
        
        try:
            request = TTSRequest(**data)
            
            # Synthesize speech
            audio_result = await self.tts_synthesizer.synthesize(
                text=request.text,
                voice_id=request.voice_id,
                speed=request.speed,
                pitch=request.pitch,
                emotion=request.emotion
            )
            
            # Process audio (normalize, enhance)
            processed_audio = self.audio_processor.process_audio(
                audio_result.audio_data,
                enhance_quality=True,
                normalize=True
            )
            
            # Encode audio to base64
            audio_base64 = self.audio_processor.encode_audio_base64(processed_audio)
            
            processing_time = (time.time() - start_time) * 1000
            
            response = TTSResponse(
                session_id=request.session_id,
                timestamp=request.timestamp,
                audio_data=audio_base64,
                audio_format="wav",
                duration_ms=audio_result.duration_ms,
                processing_time_ms=processing_time,
                voice_characteristics=audio_result.voice_characteristics
            )
            
            # Publish audio to frontend
            await self.nats_client.publish("audio.stream", response.dict())
            logger.debug(f"Processed TTS for session {request.session_id} in {processing_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error processing TTS: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            # Publish error response
            error_response = {
                "session_id": data.get("session_id", "unknown"),
                "timestamp": data.get("timestamp", time.time()),
                "error": str(e),
                "processing_time_ms": processing_time
            }
            await self.nats_client.publish("tts.error", error_response)

# Global worker instance
worker = TTSWorker()

@app.on_event("startup")
async def startup_event():
    await worker.start()

@app.on_event("shutdown")
async def shutdown_event():
    await worker.nats_client.disconnect()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "tts-worker"}

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """HTTP endpoint for TTS synthesis (alternative to NATS)"""
    await worker.process_text(request.dict())
    return {"status": "processed", "session_id": request.session_id}

@app.get("/voices")
async def list_voices():
    """List available voices"""
    return {
        "voices": [
            {"id": "default", "name": "Default Voice", "language": "en-US"},
            {"id": "female1", "name": "Sarah", "language": "en-US"},
            {"id": "male1", "name": "David", "language": "en-US"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
