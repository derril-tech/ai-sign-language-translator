import asyncio
import logging
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from shared.nats_client import NATSClient
from gloss_decoder import GlossDecoder, GlossSequence
from fingerspelling_detector import FingerspellingDetector
from spatial_analyzer import SpatialAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gloss Worker", version="1.0.0")

class GlossRequest(BaseModel):
    session_id: str
    timestamp: float
    frame_id: Optional[str] = None
    pose_landmarks: Dict[str, Any]
    hand_landmarks: Dict[str, Any]
    face_landmarks: Dict[str, Any]
    confidence: float
    processing_time_ms: float
    frame_metrics: Dict[str, Any]

class GlossResponse(BaseModel):
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

class GlossWorker:
    def __init__(self):
        self.nats_client = NATSClient()
        self.gloss_decoder = GlossDecoder(
            vocab_size=5000,
            hidden_dim=512,
            num_layers=3,
            dropout=0.1
        )
        self.fingerspelling_detector = FingerspellingDetector()
        self.spatial_analyzer = SpatialAnalyzer()
        
        # Load pre-trained models (placeholder paths)
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models"""
        try:
            # In production, load actual model weights
            # self.gloss_decoder.load_state_dict(torch.load('models/gloss_decoder.pth'))
            logger.info("Gloss models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            logger.info("Using randomly initialized models for development")
        
    async def start(self):
        """Start the gloss worker"""
        await self.nats_client.connect()
        await self.nats_client.subscribe("gloss.decode", self.process_pose_data, queue="gloss-workers")
        logger.info("Gloss worker started and listening for pose data")
        
    async def process_pose_data(self, data: Dict[str, Any]):
        """Process pose landmarks to extract gloss sequences"""
        start_time = time.time()
        
        try:
            request = GlossRequest(**data)
            
            # Extract features from pose landmarks
            features = self._extract_features(request)
            
            # Decode gloss sequence
            gloss_sequence = await self.gloss_decoder.decode_sequence(features)
            
            # Detect fingerspelling
            fingerspelling = self.fingerspelling_detector.detect_letters(
                request.hand_landmarks
            )
            
            # Analyze spatial roles
            spatial_roles = self.spatial_analyzer.analyze_spatial_grammar(
                request.pose_landmarks,
                request.hand_landmarks
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Generate partial and commit hypotheses
            partial_hypothesis, commit_hypothesis = self._generate_hypotheses(
                gloss_sequence, fingerspelling
            )
            
            response = GlossResponse(
                session_id=request.session_id,
                timestamp=request.timestamp,
                frame_id=request.frame_id,
                gloss_sequence=gloss_sequence.to_dict() if gloss_sequence else [],
                fingerspelling=fingerspelling,
                spatial_roles=spatial_roles,
                confidence=gloss_sequence.confidence if gloss_sequence else 0.0,
                processing_time_ms=processing_time,
                partial_hypothesis=partial_hypothesis,
                commit_hypothesis=commit_hypothesis
            )
            
            # Publish to semantic worker
            await self.nats_client.publish("sem.translate", response.dict())
            logger.debug(f"Processed gloss for session {request.session_id} in {processing_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error processing pose data: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            # Publish error response
            error_response = {
                "session_id": data.get("session_id", "unknown"),
                "timestamp": data.get("timestamp", time.time()),
                "error": str(e),
                "processing_time_ms": processing_time
            }
            await self.nats_client.publish("gloss.error", error_response)
    
    def _extract_features(self, request: GlossRequest) -> torch.Tensor:
        """Extract features from pose landmarks for gloss decoding"""
        try:
            features = []
            
            # Extract body pose features
            if request.pose_landmarks.get("body"):
                body_features = self._extract_body_features(request.pose_landmarks["body"])
                features.extend(body_features)
            
            # Extract hand features
            if request.hand_landmarks.get("left"):
                left_hand_features = self._extract_hand_features(request.hand_landmarks["left"])
                features.extend(left_hand_features)
            
            if request.hand_landmarks.get("right"):
                right_hand_features = self._extract_hand_features(request.hand_landmarks["right"])
                features.extend(right_hand_features)
            
            # Extract face features (for non-manual markers)
            if request.face_landmarks.get("landmarks"):
                face_features = self._extract_face_features(request.face_landmarks["landmarks"])
                features.extend(face_features)
            
            # Convert to tensor
            if features:
                feature_tensor = torch.tensor(features, dtype=torch.float32)
                return feature_tensor.unsqueeze(0)  # Add batch dimension
            else:
                # Return zero tensor if no features
                return torch.zeros(1, 300)  # Placeholder feature size
                
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return torch.zeros(1, 300)
    
    def _extract_body_features(self, body_landmarks: List[Dict]) -> List[float]:
        """Extract features from body landmarks"""
        features = []
        
        # Key body points for sign language
        key_points = [11, 12, 13, 14, 15, 16, 23, 24]  # Shoulders, elbows, wrists, hips
        
        for i, landmark in enumerate(body_landmarks):
            if i in key_points:
                features.extend([
                    landmark.get('x', 0.0),
                    landmark.get('y', 0.0),
                    landmark.get('z', 0.0),
                    landmark.get('visibility', 0.0)
                ])
        
        # Pad to fixed size
        while len(features) < 32:  # 8 points * 4 features
            features.append(0.0)
            
        return features[:32]
    
    def _extract_hand_features(self, hand_landmarks: List[Dict]) -> List[float]:
        """Extract features from hand landmarks"""
        features = []
        
        for landmark in hand_landmarks:
            features.extend([
                landmark.get('x', 0.0),
                landmark.get('y', 0.0),
                landmark.get('z', 0.0)
            ])
        
        # Pad to fixed size (21 landmarks * 3 features = 63)
        while len(features) < 63:
            features.append(0.0)
            
        return features[:63]
    
    def _extract_face_features(self, face_landmarks: List[Dict]) -> List[float]:
        """Extract features from face landmarks (for NMM)"""
        features = []
        
        # Key facial points for non-manual markers
        # Eyes, eyebrows, mouth, cheeks
        key_points = list(range(0, 468, 10))  # Sample every 10th point
        
        for i, landmark in enumerate(face_landmarks):
            if i in key_points:
                features.extend([
                    landmark.get('x', 0.0),
                    landmark.get('y', 0.0),
                    landmark.get('z', 0.0)
                ])
        
        # Pad to fixed size
        while len(features) < 141:  # 47 points * 3 features
            features.append(0.0)
            
        return features[:141]
    
    def _generate_hypotheses(self, gloss_sequence: Optional[GlossSequence], 
                           fingerspelling: List[str]) -> Tuple[str, str]:
        """Generate partial and commit hypotheses"""
        try:
            partial_tokens = []
            commit_tokens = []
            
            if gloss_sequence:
                for gloss in gloss_sequence.glosses:
                    if gloss.confidence > 0.8:
                        commit_tokens.append(gloss.token)
                    elif gloss.confidence > 0.5:
                        partial_tokens.append(f"[{gloss.token}]")
            
            # Add fingerspelling
            if fingerspelling:
                fs_string = "".join(fingerspelling)
                if len(fs_string) > 2:  # Only add if substantial fingerspelling
                    partial_tokens.append(f"#{fs_string}#")
            
            partial_hypothesis = " ".join(partial_tokens)
            commit_hypothesis = " ".join(commit_tokens)
            
            return partial_hypothesis, commit_hypothesis
            
        except Exception as e:
            logger.error(f"Error generating hypotheses: {e}")
            return "", ""

# Global worker instance
worker = GlossWorker()

@app.on_event("startup")
async def startup_event():
    await worker.start()

@app.on_event("shutdown")
async def shutdown_event():
    await worker.nats_client.disconnect()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "gloss-worker"}

@app.post("/decode")
async def decode_gloss(request: GlossRequest):
    """HTTP endpoint for gloss decoding (alternative to NATS)"""
    await worker.process_pose_data(request.dict())
    return {"status": "processed", "session_id": request.session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
