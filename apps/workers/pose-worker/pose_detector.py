import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

@dataclass
class PoseLandmarks:
    """Structured pose landmarks with confidence scores"""
    body: List[Dict[str, float]]
    left_hand: List[Dict[str, float]]
    right_hand: List[Dict[str, float]]
    face: List[Dict[str, float]]
    confidence: float
    timestamp: float

class SmoothingFilter:
    """Temporal smoothing filter for pose landmarks"""
    
    def __init__(self, window_size: int = 5, poly_order: int = 2):
        self.window_size = window_size
        self.poly_order = poly_order
        self.history: Dict[str, List[np.ndarray]] = {
            'body': [],
            'left_hand': [],
            'right_hand': [],
            'face': []
        }
        
    def smooth_landmarks(self, landmarks: PoseLandmarks) -> PoseLandmarks:
        """Apply temporal smoothing to landmarks"""
        # Add current landmarks to history
        self._add_to_history('body', landmarks.body)
        self._add_to_history('left_hand', landmarks.left_hand)
        self._add_to_history('right_hand', landmarks.right_hand)
        self._add_to_history('face', landmarks.face)
        
        # Apply smoothing if we have enough history
        smoothed_landmarks = PoseLandmarks(
            body=self._smooth_landmark_group('body'),
            left_hand=self._smooth_landmark_group('left_hand'),
            right_hand=self._smooth_landmark_group('right_hand'),
            face=self._smooth_landmark_group('face'),
            confidence=landmarks.confidence,
            timestamp=landmarks.timestamp
        )
        
        return smoothed_landmarks
    
    def _add_to_history(self, group: str, landmarks: List[Dict[str, float]]):
        """Add landmarks to history buffer"""
        if landmarks:
            # Convert to numpy array for easier processing
            landmark_array = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
            self.history[group].append(landmark_array)
            
            # Keep only recent history
            if len(self.history[group]) > self.window_size:
                self.history[group].pop(0)
    
    def _smooth_landmark_group(self, group: str) -> List[Dict[str, float]]:
        """Apply Savitzky-Golay smoothing to landmark group"""
        if len(self.history[group]) < 3:
            # Not enough history, return latest
            if self.history[group]:
                latest = self.history[group][-1]
                return [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in latest]
            return []
        
        # Stack history for smoothing
        history_stack = np.stack(self.history[group], axis=0)  # (time, landmarks, xyz)
        
        # Apply smoothing along time axis
        window_len = min(len(self.history[group]), self.window_size)
        if window_len >= self.poly_order + 1:
            smoothed = savgol_filter(history_stack, window_len, self.poly_order, axis=0)
            # Return the most recent smoothed frame
            latest_smoothed = smoothed[-1]
            return [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in latest_smoothed]
        else:
            # Fallback to simple averaging
            averaged = np.mean(history_stack, axis=0)
            return [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in averaged]

class PoseDetector:
    """Advanced pose detection with MediaPipe and smoothing"""
    
    def __init__(self, 
                 model_complexity: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 enable_smoothing: bool = True):
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize smoothing filter
        self.smoothing_filter = SmoothingFilter() if enable_smoothing else None
        
        logger.info(f"PoseDetector initialized with complexity={model_complexity}")
    
    def detect_pose(self, frame: np.ndarray, timestamp: float) -> Optional[PoseLandmarks]:
        """Detect pose landmarks from video frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.holistic.process(rgb_frame)
            
            # Extract landmarks
            landmarks = self._extract_landmarks(results, timestamp)
            
            if landmarks and self.smoothing_filter:
                # Apply temporal smoothing
                landmarks = self.smoothing_filter.smooth_landmarks(landmarks)
            
            return landmarks
            
        except Exception as e:
            logger.error(f"Error in pose detection: {e}")
            return None
    
    def _extract_landmarks(self, results, timestamp: float) -> Optional[PoseLandmarks]:
        """Extract structured landmarks from MediaPipe results"""
        try:
            # Calculate overall confidence
            confidences = []
            
            # Extract body landmarks
            body_landmarks = []
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    body_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                confidences.append(np.mean([lm['visibility'] for lm in body_landmarks]))
            
            # Extract hand landmarks
            left_hand_landmarks = []
            if results.left_hand_landmarks:
                for landmark in results.left_hand_landmarks.landmark:
                    left_hand_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                confidences.append(0.9)  # Hand landmarks don't have visibility
            
            right_hand_landmarks = []
            if results.right_hand_landmarks:
                for landmark in results.right_hand_landmarks.landmark:
                    right_hand_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                confidences.append(0.9)
            
            # Extract face landmarks
            face_landmarks = []
            if results.face_landmarks:
                for landmark in results.face_landmarks.landmark:
                    face_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                confidences.append(0.8)  # Face landmarks confidence estimate
            
            # Calculate overall confidence
            overall_confidence = np.mean(confidences) if confidences else 0.0
            
            # Only return if we have at least body or hands
            if body_landmarks or left_hand_landmarks or right_hand_landmarks:
                return PoseLandmarks(
                    body=body_landmarks,
                    left_hand=left_hand_landmarks,
                    right_hand=right_hand_landmarks,
                    face=face_landmarks,
                    confidence=overall_confidence,
                    timestamp=timestamp
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: PoseLandmarks) -> np.ndarray:
        """Draw pose landmarks on frame for visualization"""
        try:
            # This would require converting back to MediaPipe format
            # For now, return frame as-is
            return frame
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")
            return frame
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'holistic'):
            self.holistic.close()
