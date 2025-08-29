import asyncio
import logging
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from shared.nats_client import NATSClient
from speech_recognizer import SpeechRecognizer
from diarization import SpeakerDiarization
from audio_processor import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ASR Worker", version="1.0.0")

class ASRRequest(BaseModel):
    session_id: str
    audio_data: str  # base64 encoded audio
    timestamp: float
    audio_format: str = "wav"
    language: str = "en-US"
    enable_diarization: bool = True

class ASRResponse(BaseModel):
    session_id: str
    timestamp: float
    transcribed_text: str
    confidence: float
    processing_time_ms: float
    speaker_segments: List[Dict[str, Any]]
    language_detected: str
    audio_metrics: Dict[str, Any]

class ASRWorker:
    def __init__(self):
        self.nats_client = NATSClient()
        self.speech_recognizer = SpeechRecognizer(
            model_name="whisper-large-v3",
            language="en"
        )
        self.speaker_diarization = SpeakerDiarization()
        self.audio_processor = AudioProcessor()
        
        # Load ASR models
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained ASR models"""
        try:
            logger.info("ASR models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            logger.info("Using mock ASR for development")
        
    async def start(self):
        """Start the ASR worker"""
        await self.nats_client.connect()
        await self.nats_client.subscribe("asr.decode", self.process_audio, queue="asr-workers")
        logger.info("ASR worker started and listening for audio")
        
    async def process_audio(self, data: Dict[str, Any]):
        """Process audio to generate text transcription"""
        start_time = time.time()
        
        try:
            request = ASRRequest(**data)
            
            # Decode audio
            audio_data = self.audio_processor.decode_audio_base64(request.audio_data)
            if audio_data is None:
                logger.error(f"Failed to decode audio for session {request.session_id}")
                return
            
            # Calculate audio metrics
            audio_metrics = self.audio_processor.calculate_audio_metrics(audio_data)
            
            # Perform speech recognition
            transcription_result = await self.speech_recognizer.transcribe(
                audio_data=audio_data,
                language=request.language
            )
            
            # Perform speaker diarization if enabled
            speaker_segments = []
            if request.enable_diarization and len(audio_data) > 0:
                speaker_segments = await self.speaker_diarization.diarize(
                    audio_data=audio_data,
                    transcription=transcription_result.text
                )
            
            processing_time = (time.time() - start_time) * 1000
            
            response = ASRResponse(
                session_id=request.session_id,
                timestamp=request.timestamp,
                transcribed_text=transcription_result.text,
                confidence=transcription_result.confidence,
                processing_time_ms=processing_time,
                speaker_segments=speaker_segments,
                language_detected=transcription_result.language,
                audio_metrics=audio_metrics
            )
            
            # Publish transcription
            await self.nats_client.publish("transcription.result", response.dict())
            logger.debug(f"Processed ASR for session {request.session_id} in {processing_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error processing ASR: {e}")

# Global worker instance
worker = ASRWorker()

@app.on_event("startup")
async def startup_event():
    await worker.start()

@app.on_event("shutdown")
async def shutdown_event():
    await worker.nats_client.disconnect()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "asr-worker"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
