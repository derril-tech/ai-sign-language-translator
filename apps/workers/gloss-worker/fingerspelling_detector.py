import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FingerSpellingFrame:
    """Single frame of fingerspelling detection"""
    letter: str
    confidence: float
    hand_landmarks: List[Dict[str, float]]
    timestamp: float

class HandShapeClassifier(nn.Module):
    """Neural network for classifying hand shapes into letters"""
    
    def __init__(self, input_dim: int = 63, num_classes: int = 26):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

class FingerspellingDetector:
    """Detects fingerspelling from hand landmarks"""
    
    def __init__(self):
        # Initialize classifier for both hands
        self.left_hand_classifier = HandShapeClassifier()
        self.right_hand_classifier = HandShapeClassifier()
        
        # Letter mapping
        self.letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        
        # Temporal smoothing buffer
        self.left_hand_buffer = []
        self.right_hand_buffer = []
        self.buffer_size = 5
        
        # Confidence thresholds
        self.min_confidence = 0.7
        self.stability_threshold = 3  # Frames to confirm letter
        
        logger.info("FingerspellingDetector initialized")
    
    def detect_letters(self, hand_landmarks: Dict[str, List[Dict]]) -> List[str]:
        """Detect fingerspelled letters from hand landmarks"""
        try:
            detected_letters = []
            
            # Process left hand
            if hand_landmarks.get("left"):
                left_letter = self._detect_letter_from_hand(
                    hand_landmarks["left"], 
                    self.left_hand_classifier,
                    self.left_hand_buffer
                )
                if left_letter:
                    detected_letters.append(left_letter)
            
            # Process right hand
            if hand_landmarks.get("right"):
                right_letter = self._detect_letter_from_hand(
                    hand_landmarks["right"],
                    self.right_hand_classifier,
                    self.right_hand_buffer
                )
                if right_letter:
                    detected_letters.append(right_letter)
            
            return detected_letters
            
        except Exception as e:
            logger.error(f"Error detecting fingerspelling: {e}")
            return []
    
    def _detect_letter_from_hand(self, 
                                landmarks: List[Dict], 
                                classifier: HandShapeClassifier,
                                buffer: List[str]) -> Optional[str]:
        """Detect letter from single hand landmarks"""
        try:
            if not landmarks or len(landmarks) != 21:
                return None
            
            # Extract features
            features = self._extract_hand_features(landmarks)
            if features is None:
                return None
            
            # Classify
            with torch.no_grad():
                logits = classifier(features)
                probabilities = torch.softmax(logits, dim=-1)
                confidence, predicted_class = torch.max(probabilities, dim=-1)
                
                confidence_val = confidence.item()
                letter_idx = predicted_class.item()
                
                if confidence_val > self.min_confidence:
                    letter = self.letters[letter_idx]
                    
                    # Add to temporal buffer
                    buffer.append(letter)
                    if len(buffer) > self.buffer_size:
                        buffer.pop(0)
                    
                    # Check for stability
                    if len(buffer) >= self.stability_threshold:
                        # Check if recent frames are consistent
                        recent_letters = buffer[-self.stability_threshold:]
                        if all(l == recent_letters[0] for l in recent_letters):
                            return recent_letters[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting letter from hand: {e}")
            return None
    
    def _extract_hand_features(self, landmarks: List[Dict]) -> Optional[torch.Tensor]:
        """Extract features from hand landmarks for classification"""
        try:
            # Convert landmarks to feature vector
            features = []
            
            for landmark in landmarks:
                features.extend([
                    landmark.get('x', 0.0),
                    landmark.get('y', 0.0),
                    landmark.get('z', 0.0)
                ])
            
            # Normalize features
            features = np.array(features, dtype=np.float32)
            
            # Calculate relative positions (important for hand shape)
            if len(features) == 63:  # 21 landmarks * 3 coordinates
                features = self._calculate_relative_features(features)
            
            return torch.tensor(features).unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            logger.error(f"Error extracting hand features: {e}")
            return None
    
    def _calculate_relative_features(self, landmarks: np.ndarray) -> np.ndarray:
        """Calculate relative positions and angles for better hand shape recognition"""
        try:
            # Reshape to (21, 3)
            points = landmarks.reshape(21, 3)
            
            # Use wrist (point 0) as reference
            wrist = points[0]
            
            # Calculate relative positions
            relative_points = points - wrist
            
            # Calculate distances between key points
            distances = []
            
            # Finger tip to base distances
            finger_tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
            finger_bases = [2, 5, 9, 13, 17]  # Corresponding bases
            
            for tip, base in zip(finger_tips, finger_bases):
                dist = np.linalg.norm(relative_points[tip] - relative_points[base])
                distances.append(dist)
            
            # Calculate angles between fingers
            angles = []
            for i in range(len(finger_tips) - 1):
                v1 = relative_points[finger_tips[i]]
                v2 = relative_points[finger_tips[i + 1]]
                
                # Calculate angle
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(angle)
            
            # Combine features
            features = np.concatenate([
                relative_points.flatten(),  # 63 features
                distances,                  # 5 features
                angles                      # 4 features
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating relative features: {e}")
            return landmarks
    
    def load_models(self, left_hand_path: str, right_hand_path: str):
        """Load pre-trained fingerspelling models"""
        try:
            self.left_hand_classifier.load_state_dict(torch.load(left_hand_path))
            self.right_hand_classifier.load_state_dict(torch.load(right_hand_path))
            logger.info("Loaded fingerspelling models")
        except Exception as e:
            logger.warning(f"Could not load fingerspelling models: {e}")
    
    def reset_buffers(self):
        """Reset temporal buffers"""
        self.left_hand_buffer.clear()
        self.right_hand_buffer.clear()
    
    def get_fingerspelling_confidence(self, landmarks: List[Dict]) -> float:
        """Calculate confidence that current pose is fingerspelling"""
        try:
            if not landmarks or len(landmarks) != 21:
                return 0.0
            
            # Heuristics for fingerspelling detection:
            # 1. Hand should be relatively stable
            # 2. Hand should be in signing space (chest/face level)
            # 3. Fingers should be in clear configuration
            
            # Calculate hand position
            palm_center = np.mean([[lm.get('x', 0), lm.get('y', 0)] for lm in landmarks[:4]], axis=0)
            
            # Check if in signing space (rough heuristic)
            if 0.3 <= palm_center[0] <= 0.7 and 0.2 <= palm_center[1] <= 0.8:
                return 0.8
            
            return 0.3
            
        except Exception as e:
            logger.error(f"Error calculating fingerspelling confidence: {e}")
            return 0.0
