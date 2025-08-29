import asyncio
import logging
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from shared.nats_client import NATSClient
from pose_detector import PoseDetector, PoseLandmarks
from frame_processor import FrameProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pose Worker", version="1.0.0")

class PoseRequest(BaseModel):
    session_id: str
    frame_data: str  # base64 encoded frame
    timestamp: float
    frame_id: Optional[str] = None

class PoseResponse(BaseModel):
    session_id: str
    timestamp: float
    frame_id: Optional[str] = None
    pose_landmarks: Dict[str, Any]
    hand_landmarks: Dict[str, Any]
    face_landmarks: Dict[str, Any]
    confidence: float
    processing_time_ms: float
    frame_metrics: Dict[str, Any]

class PoseWorker:
    def __init__(self):
        self.nats_client = NATSClient()
        self.pose_detector = PoseDetector(
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            enable_smoothing=True
        )
        self.frame_processor = FrameProcessor(
            target_width=640,
            target_height=480,
            quality=85
        )
        
    async def start(self):
        """Start the pose worker"""
        await self.nats_client.connect()
        await self.nats_client.subscribe("pose.stream", self.process_frame, queue="pose-workers")
        logger.info("Pose worker started and listening for frames")
        
    async def process_frame(self, data: Dict[str, Any]):
        """Process incoming frame for pose detection"""
        start_time = time.time()
        
        try:
            request = PoseRequest(**data)
            
            # Decode base64 frame
            frame = self.frame_processor.decode_base64_frame(request.frame_data)
            if frame is None:
                logger.error(f"Failed to decode frame for session {request.session_id}")
                return
            
            # Preprocess frame
            processed_frame = self.frame_processor.preprocess_frame(frame)
            
            # Calculate frame metrics
            frame_metrics = self.frame_processor.calculate_frame_metrics(processed_frame)
            
            # Detect pose landmarks
            landmarks = self.pose_detector.detect_pose(processed_frame, request.timestamp)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            if landmarks:
                # Create response with detected landmarks
                response = PoseResponse(
                    session_id=request.session_id,
                    timestamp=request.timestamp,
                    frame_id=request.frame_id,
                    pose_landmarks={
                        "body": landmarks.body,
                        "confidence": landmarks.confidence
                    },
                    hand_landmarks={
                        "left": landmarks.left_hand,
                        "right": landmarks.right_hand
                    },
                    face_landmarks={
                        "landmarks": landmarks.face
                    },
                    confidence=landmarks.confidence,
                    processing_time_ms=processing_time,
                    frame_metrics=frame_metrics
                )
                
                # Publish processed pose data to gloss worker
                await self.nats_client.publish("gloss.decode", response.dict())
                logger.debug(f"Processed pose for session {request.session_id} in {processing_time:.1f}ms")
                
            else:
                # No landmarks detected
                logger.warning(f"No pose landmarks detected for session {request.session_id}")
                
                # Still publish a response with empty landmarks
                response = PoseResponse(
                    session_id=request.session_id,
                    timestamp=request.timestamp,
                    frame_id=request.frame_id,
                    pose_landmarks={"body": [], "confidence": 0.0},
                    hand_landmarks={"left": [], "right": []},
                    face_landmarks={"landmarks": []},
                    confidence=0.0,
                    processing_time_ms=processing_time,
                    frame_metrics=frame_metrics
                )
                
                await self.nats_client.publish("gloss.decode", response.dict())
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            # Publish error response
            error_response = {
                "session_id": data.get("session_id", "unknown"),
                "timestamp": data.get("timestamp", time.time()),
                "error": str(e),
                "processing_time_ms": processing_time
            }
            await self.nats_client.publish("pose.error", error_response)

# Global worker instance
worker = PoseWorker()

@app.on_event("startup")
async def startup_event():
    await worker.start()

@app.on_event("shutdown")
async def shutdown_event():
    await worker.nats_client.disconnect()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "pose-worker"}

@app.post("/process")
async def process_pose(request: PoseRequest):
    """HTTP endpoint for pose processing (alternative to NATS)"""
    await worker.process_frame(request.dict())
    return {"status": "processed", "session_id": request.session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
