import base64
import cv2
import numpy as np
from typing import Optional, Tuple
import logging
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

class FrameProcessor:
    """Handles frame decoding, preprocessing, and encoding"""
    
    def __init__(self, 
                 target_width: int = 640,
                 target_height: int = 480,
                 quality: int = 85):
        self.target_width = target_width
        self.target_height = target_height
        self.quality = quality
        
    def decode_base64_frame(self, base64_data: str) -> Optional[np.ndarray]:
        """Decode base64 encoded frame to OpenCV format"""
        try:
            # Remove data URL prefix if present
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_data)
            
            # Convert to PIL Image
            pil_image = Image.open(BytesIO(image_data))
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to OpenCV format (BGR)
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error decoding base64 frame: {e}")
            return None
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for pose detection"""
        try:
            # Resize frame to target dimensions
            if frame.shape[:2] != (self.target_height, self.target_width):
                frame = cv2.resize(frame, (self.target_width, self.target_height))
            
            # Apply noise reduction
            frame = cv2.bilateralFilter(frame, 9, 75, 75)
            
            # Enhance contrast
            frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return frame
    
    def encode_frame_to_base64(self, frame: np.ndarray) -> str:
        """Encode OpenCV frame to base64 string"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Encode to JPEG
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG', quality=self.quality)
            
            # Encode to base64
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/jpeg;base64,{base64_data}"
            
        except Exception as e:
            logger.error(f"Error encoding frame to base64: {e}")
            return ""
    
    def extract_roi(self, frame: np.ndarray, landmarks) -> Tuple[np.ndarray, np.ndarray]:
        """Extract regions of interest (hands, face) from frame"""
        try:
            # Extract hand regions based on landmarks
            left_hand_roi = self._extract_hand_roi(frame, landmarks.left_hand, 'left')
            right_hand_roi = self._extract_hand_roi(frame, landmarks.right_hand, 'right')
            
            return left_hand_roi, right_hand_roi
            
        except Exception as e:
            logger.error(f"Error extracting ROI: {e}")
            return frame, frame
    
    def _extract_hand_roi(self, frame: np.ndarray, hand_landmarks: list, hand_type: str) -> np.ndarray:
        """Extract hand region of interest"""
        try:
            if not hand_landmarks:
                return np.zeros((64, 64, 3), dtype=np.uint8)
            
            # Calculate bounding box
            x_coords = [lm['x'] for lm in hand_landmarks]
            y_coords = [lm['y'] for lm in hand_landmarks]
            
            # Convert normalized coordinates to pixel coordinates
            h, w = frame.shape[:2]
            x_coords = [int(x * w) for x in x_coords]
            y_coords = [int(y * h) for y in y_coords]
            
            # Add padding
            padding = 20
            x_min = max(0, min(x_coords) - padding)
            x_max = min(w, max(x_coords) + padding)
            y_min = max(0, min(y_coords) - padding)
            y_max = min(h, max(y_coords) + padding)
            
            # Extract ROI
            roi = frame[y_min:y_max, x_min:x_max]
            
            # Resize to standard size
            roi = cv2.resize(roi, (64, 64))
            
            return roi
            
        except Exception as e:
            logger.error(f"Error extracting {hand_type} hand ROI: {e}")
            return np.zeros((64, 64, 3), dtype=np.uint8)
    
    def calculate_frame_metrics(self, frame: np.ndarray) -> dict:
        """Calculate frame quality metrics"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray)
            
            return {
                'sharpness': float(sharpness),
                'brightness': float(brightness),
                'contrast': float(contrast),
                'width': frame.shape[1],
                'height': frame.shape[0]
            }
            
        except Exception as e:
            logger.error(f"Error calculating frame metrics: {e}")
            return {}
