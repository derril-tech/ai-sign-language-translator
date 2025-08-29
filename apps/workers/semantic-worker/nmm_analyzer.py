import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import cv2

logger = logging.getLogger(__name__)

class NMMType(Enum):
    """Types of non-manual markers"""
    FACIAL_EXPRESSION = "facial_expression"
    HEAD_MOVEMENT = "head_movement"
    BODY_POSTURE = "body_posture"
    EYE_GAZE = "eye_gaze"
    MOUTH_SHAPE = "mouth_shape"

@dataclass
class NMMFeature:
    """Non-manual marker feature"""
    type: NMMType
    name: str
    intensity: float
    confidence: float
    temporal_info: Dict[str, Any]
    spatial_info: Optional[Dict[str, Any]] = None

class FacialExpressionAnalyzer:
    """Analyzes facial expressions for linguistic meaning"""
    
    def __init__(self):
        # Key facial landmarks for expressions
        self.eyebrow_landmarks = list(range(17, 27))  # Eyebrow region
        self.eye_landmarks = list(range(36, 48))      # Eye region
        self.mouth_landmarks = list(range(48, 68))    # Mouth region
        self.nose_landmarks = list(range(27, 36))     # Nose region
        
        # Expression templates (simplified)
        self.expression_templates = {
            'eyebrow_raise': {'eyebrow_height_threshold': 0.02},
            'eye_squint': {'eye_openness_threshold': 0.3},
            'mouth_open': {'mouth_openness_threshold': 0.01},
            'cheek_puff': {'cheek_expansion_threshold': 0.015}
        }
        
    def analyze_expressions(self, face_landmarks: List[Dict]) -> Dict[str, Any]:
        """Analyze facial expressions from landmarks"""
        try:
            if not face_landmarks or len(face_landmarks) < 468:
                return {}
            
            expressions = {}
            
            # Analyze eyebrow raise (question marker)
            eyebrow_raise = self._detect_eyebrow_raise(face_landmarks)
            if eyebrow_raise['detected']:
                expressions['eyebrow_raise'] = {
                    'intensity': eyebrow_raise['intensity'],
                    'confidence': eyebrow_raise['confidence'],
                    'linguistic_meaning': 'question_marker'
                }
            
            # Analyze eye gaze direction
            eye_gaze = self._detect_eye_gaze(face_landmarks)
            if eye_gaze['detected']:
                expressions['eye_gaze'] = {
                    'direction': eye_gaze['direction'],
                    'confidence': eye_gaze['confidence'],
                    'linguistic_meaning': 'reference_shift'
                }
            
            # Analyze mouth shapes (adverbial markers)
            mouth_shape = self._detect_mouth_shape(face_landmarks)
            if mouth_shape['detected']:
                expressions['mouth_shape'] = {
                    'type': mouth_shape['type'],
                    'intensity': mouth_shape['intensity'],
                    'confidence': mouth_shape['confidence'],
                    'linguistic_meaning': mouth_shape['meaning']
                }
            
            # Analyze cheek puff (size/intensity marker)
            cheek_puff = self._detect_cheek_puff(face_landmarks)
            if cheek_puff['detected']:
                expressions['cheek_puff'] = {
                    'intensity': cheek_puff['intensity'],
                    'confidence': cheek_puff['confidence'],
                    'linguistic_meaning': 'intensity_marker'
                }
            
            return expressions
            
        except Exception as e:
            logger.error(f"Error analyzing facial expressions: {e}")
            return {}
    
    def _detect_eyebrow_raise(self, landmarks: List[Dict]) -> Dict[str, Any]:
        """Detect eyebrow raise (question marker)"""
        try:
            # Calculate eyebrow height relative to eyes
            eyebrow_points = [landmarks[i] for i in self.eyebrow_landmarks]
            eye_points = [landmarks[i] for i in self.eye_landmarks]
            
            avg_eyebrow_y = np.mean([p['y'] for p in eyebrow_points])
            avg_eye_y = np.mean([p['y'] for p in eye_points])
            
            eyebrow_height = avg_eye_y - avg_eyebrow_y  # Higher eyebrows = lower y values
            
            threshold = self.expression_templates['eyebrow_raise']['eyebrow_height_threshold']
            
            if eyebrow_height > threshold:
                intensity = min(eyebrow_height / (threshold * 2), 1.0)
                return {
                    'detected': True,
                    'intensity': intensity,
                    'confidence': 0.8 if intensity > 0.5 else 0.6
                }
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"Error detecting eyebrow raise: {e}")
            return {'detected': False}
    
    def _detect_eye_gaze(self, landmarks: List[Dict]) -> Dict[str, Any]:
        """Detect eye gaze direction"""
        try:
            # Simplified gaze detection using eye landmarks
            left_eye = [landmarks[i] for i in range(36, 42)]
            right_eye = [landmarks[i] for i in range(42, 48)]
            
            # Calculate eye center and iris position (simplified)
            left_center = np.mean([[p['x'], p['y']] for p in left_eye], axis=0)
            right_center = np.mean([[p['x'], p['y']] for p in right_eye], axis=0)
            
            # Detect gaze direction based on eye shape (simplified heuristic)
            eye_aspect_ratio = self._calculate_eye_aspect_ratio(left_eye, right_eye)
            
            if eye_aspect_ratio < 0.25:  # Eyes more closed
                return {
                    'detected': True,
                    'direction': 'focused',
                    'confidence': 0.7
                }
            
            # Check for lateral gaze (simplified)
            if abs(left_center[0] - right_center[0]) > 0.1:
                direction = 'left' if left_center[0] < right_center[0] else 'right'
                return {
                    'detected': True,
                    'direction': direction,
                    'confidence': 0.6
                }
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"Error detecting eye gaze: {e}")
            return {'detected': False}
    
    def _detect_mouth_shape(self, landmarks: List[Dict]) -> Dict[str, Any]:
        """Detect mouth shapes for adverbial markers"""
        try:
            mouth_points = [landmarks[i] for i in self.mouth_landmarks]
            
            # Calculate mouth dimensions
            mouth_width = abs(mouth_points[6]['x'] - mouth_points[0]['x'])  # Corner to corner
            mouth_height = abs(mouth_points[3]['y'] - mouth_points[9]['y'])  # Top to bottom
            
            mouth_ratio = mouth_height / (mouth_width + 1e-8)
            
            # Classify mouth shape
            if mouth_ratio > 0.5:  # Tall mouth
                return {
                    'detected': True,
                    'type': 'open',
                    'intensity': min(mouth_ratio, 1.0),
                    'confidence': 0.7,
                    'meaning': 'surprise_marker'
                }
            elif mouth_ratio < 0.2:  # Wide mouth
                return {
                    'detected': True,
                    'type': 'wide',
                    'intensity': 1.0 - mouth_ratio * 5,
                    'confidence': 0.6,
                    'meaning': 'emphasis_marker'
                }
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"Error detecting mouth shape: {e}")
            return {'detected': False}
    
    def _detect_cheek_puff(self, landmarks: List[Dict]) -> Dict[str, Any]:
        """Detect cheek puff (intensity/size marker)"""
        try:
            # Use face contour to detect cheek expansion
            left_cheek_points = [landmarks[i] for i in range(1, 8)]
            right_cheek_points = [landmarks[i] for i in range(9, 16)]
            
            # Calculate face width at cheek level
            left_cheek_x = np.mean([p['x'] for p in left_cheek_points])
            right_cheek_x = np.mean([p['x'] for p in right_cheek_points])
            
            face_width = abs(right_cheek_x - left_cheek_x)
            
            # Compare to normal face width (heuristic)
            normal_width_ratio = 0.3  # Approximate normal face width ratio
            
            if face_width > normal_width_ratio * 1.1:  # 10% wider than normal
                intensity = (face_width - normal_width_ratio) / normal_width_ratio
                return {
                    'detected': True,
                    'intensity': min(intensity, 1.0),
                    'confidence': 0.6
                }
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"Error detecting cheek puff: {e}")
            return {'detected': False}
    
    def _calculate_eye_aspect_ratio(self, left_eye: List[Dict], right_eye: List[Dict]) -> float:
        """Calculate eye aspect ratio"""
        try:
            # Calculate for left eye
            left_vertical = abs(left_eye[1]['y'] - left_eye[5]['y']) + abs(left_eye[2]['y'] - left_eye[4]['y'])
            left_horizontal = abs(left_eye[0]['x'] - left_eye[3]['x'])
            left_ear = left_vertical / (2.0 * left_horizontal + 1e-8)
            
            # Calculate for right eye
            right_vertical = abs(right_eye[1]['y'] - right_eye[5]['y']) + abs(right_eye[2]['y'] - right_eye[4]['y'])
            right_horizontal = abs(right_eye[0]['x'] - right_eye[3]['x'])
            right_ear = right_vertical / (2.0 * right_horizontal + 1e-8)
            
            return (left_ear + right_ear) / 2.0
            
        except Exception as e:
            logger.error(f"Error calculating eye aspect ratio: {e}")
            return 0.3  # Default value

class HeadMovementAnalyzer:
    """Analyzes head movements for grammatical markers"""
    
    def __init__(self):
        self.movement_history = []
        self.history_size = 10
        
    def analyze_head_movements(self, pose_landmarks: List[Dict]) -> Dict[str, Any]:
        """Analyze head movements from pose landmarks"""
        try:
            if not pose_landmarks:
                return {}
            
            movements = {}
            
            # Get head pose (nose landmark)
            nose_landmark = pose_landmarks[0] if pose_landmarks else None
            if nose_landmark:
                current_position = (nose_landmark['x'], nose_landmark['y'], nose_landmark['z'])
                
                # Add to history
                self.movement_history.append(current_position)
                if len(self.movement_history) > self.history_size:
                    self.movement_history.pop(0)
                
                # Analyze movements if we have enough history
                if len(self.movement_history) >= 5:
                    nod_detected = self._detect_head_nod()
                    if nod_detected['detected']:
                        movements['nod'] = {
                            'intensity': nod_detected['intensity'],
                            'confidence': nod_detected['confidence'],
                            'linguistic_meaning': 'affirmation'
                        }
                    
                    shake_detected = self._detect_head_shake()
                    if shake_detected['detected']:
                        movements['shake'] = {
                            'intensity': shake_detected['intensity'],
                            'confidence': shake_detected['confidence'],
                            'linguistic_meaning': 'negation'
                        }
            
            return movements
            
        except Exception as e:
            logger.error(f"Error analyzing head movements: {e}")
            return {}
    
    def _detect_head_nod(self) -> Dict[str, Any]:
        """Detect head nod (up-down movement)"""
        try:
            if len(self.movement_history) < 5:
                return {'detected': False}
            
            # Calculate vertical movement
            y_positions = [pos[1] for pos in self.movement_history[-5:]]
            y_diff = max(y_positions) - min(y_positions)
            
            # Check for oscillation pattern
            if y_diff > 0.02:  # Threshold for significant movement
                # Simple oscillation detection
                direction_changes = 0
                for i in range(1, len(y_positions) - 1):
                    if (y_positions[i] > y_positions[i-1] and y_positions[i] > y_positions[i+1]) or \
                       (y_positions[i] < y_positions[i-1] and y_positions[i] < y_positions[i+1]):
                        direction_changes += 1
                
                if direction_changes >= 1:  # At least one peak/valley
                    intensity = min(y_diff / 0.05, 1.0)
                    return {
                        'detected': True,
                        'intensity': intensity,
                        'confidence': 0.7
                    }
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"Error detecting head nod: {e}")
            return {'detected': False}
    
    def _detect_head_shake(self) -> Dict[str, Any]:
        """Detect head shake (left-right movement)"""
        try:
            if len(self.movement_history) < 5:
                return {'detected': False}
            
            # Calculate horizontal movement
            x_positions = [pos[0] for pos in self.movement_history[-5:]]
            x_diff = max(x_positions) - min(x_positions)
            
            # Check for oscillation pattern
            if x_diff > 0.02:  # Threshold for significant movement
                # Simple oscillation detection
                direction_changes = 0
                for i in range(1, len(x_positions) - 1):
                    if (x_positions[i] > x_positions[i-1] and x_positions[i] > x_positions[i+1]) or \
                       (x_positions[i] < x_positions[i-1] and x_positions[i] < x_positions[i+1]):
                        direction_changes += 1
                
                if direction_changes >= 1:  # At least one peak/valley
                    intensity = min(x_diff / 0.05, 1.0)
                    return {
                        'detected': True,
                        'intensity': intensity,
                        'confidence': 0.7
                    }
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"Error detecting head shake: {e}")
            return {'detected': False}

class BodyPostureAnalyzer:
    """Analyzes body posture for discourse markers"""
    
    def analyze_body_posture(self, pose_landmarks: List[Dict]) -> Dict[str, Any]:
        """Analyze body posture from pose landmarks"""
        try:
            if not pose_landmarks or len(pose_landmarks) < 25:
                return {}
            
            posture = {}
            
            # Analyze shoulder position
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            
            # Detect shoulder shift
            shoulder_shift = self._detect_shoulder_shift(left_shoulder, right_shoulder)
            if shoulder_shift['detected']:
                posture['shoulder_shift'] = {
                    'direction': shoulder_shift['direction'],
                    'intensity': shoulder_shift['intensity'],
                    'confidence': shoulder_shift['confidence'],
                    'linguistic_meaning': 'role_shift'
                }
            
            # Analyze torso lean
            torso_lean = self._detect_torso_lean(pose_landmarks)
            if torso_lean['detected']:
                posture['torso_lean'] = {
                    'direction': torso_lean['direction'],
                    'intensity': torso_lean['intensity'],
                    'confidence': torso_lean['confidence'],
                    'linguistic_meaning': 'emphasis_marker'
                }
            
            return posture
            
        except Exception as e:
            logger.error(f"Error analyzing body posture: {e}")
            return {}
    
    def _detect_shoulder_shift(self, left_shoulder: Dict, right_shoulder: Dict) -> Dict[str, Any]:
        """Detect shoulder shift for role shifting"""
        try:
            # Calculate shoulder height difference
            height_diff = abs(left_shoulder['y'] - right_shoulder['y'])
            
            if height_diff > 0.02:  # Threshold for significant shift
                direction = 'left_up' if left_shoulder['y'] < right_shoulder['y'] else 'right_up'
                intensity = min(height_diff / 0.05, 1.0)
                
                return {
                    'detected': True,
                    'direction': direction,
                    'intensity': intensity,
                    'confidence': 0.6
                }
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"Error detecting shoulder shift: {e}")
            return {'detected': False}
    
    def _detect_torso_lean(self, pose_landmarks: List[Dict]) -> Dict[str, Any]:
        """Detect torso lean for emphasis"""
        try:
            # Use shoulder and hip landmarks to calculate torso angle
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            left_hip = pose_landmarks[23]
            right_hip = pose_landmarks[24]
            
            # Calculate torso center points
            shoulder_center = ((left_shoulder['x'] + right_shoulder['x']) / 2,
                             (left_shoulder['y'] + right_shoulder['y']) / 2)
            hip_center = ((left_hip['x'] + right_hip['x']) / 2,
                         (left_hip['y'] + right_hip['y']) / 2)
            
            # Calculate lean angle
            lean_angle = abs(shoulder_center[0] - hip_center[0])
            
            if lean_angle > 0.03:  # Threshold for significant lean
                direction = 'forward' if shoulder_center[0] > hip_center[0] else 'backward'
                intensity = min(lean_angle / 0.1, 1.0)
                
                return {
                    'detected': True,
                    'direction': direction,
                    'intensity': intensity,
                    'confidence': 0.5
                }
            
            return {'detected': False}
            
        except Exception as e:
            logger.error(f"Error detecting torso lean: {e}")
            return {'detected': False}

class NMMAnalyzer:
    """Main non-manual marker analyzer"""
    
    def __init__(self):
        self.facial_analyzer = FacialExpressionAnalyzer()
        self.head_analyzer = HeadMovementAnalyzer()
        self.body_analyzer = BodyPostureAnalyzer()
        
        logger.info("NMMAnalyzer initialized")
    
    def analyze_nmm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze all non-manual markers"""
        try:
            nmm_features = {
                'facial_expressions': {},
                'head_movements': {},
                'body_posture': {},
                'temporal_info': input_data.get('temporal_info', {}),
                'overall_confidence': 0.0
            }
            
            # Analyze facial expressions
            face_landmarks = input_data.get('face_landmarks', [])
            if face_landmarks:
                nmm_features['facial_expressions'] = self.facial_analyzer.analyze_expressions(face_landmarks)
            
            # Analyze head movements
            body_pose = input_data.get('body_pose', [])
            if body_pose:
                nmm_features['head_movements'] = self.head_analyzer.analyze_head_movements(body_pose)
                nmm_features['body_posture'] = self.body_analyzer.analyze_body_posture(body_pose)
            
            # Calculate overall confidence
            confidences = []
            for category in ['facial_expressions', 'head_movements', 'body_posture']:
                for feature_name, feature_data in nmm_features[category].items():
                    if isinstance(feature_data, dict) and 'confidence' in feature_data:
                        confidences.append(feature_data['confidence'])
            
            nmm_features['overall_confidence'] = np.mean(confidences) if confidences else 0.0
            
            return nmm_features
            
        except Exception as e:
            logger.error(f"Error in NMM analysis: {e}")
            return {
                'facial_expressions': {},
                'head_movements': {},
                'body_posture': {},
                'temporal_info': {},
                'overall_confidence': 0.0
            }
    
    def reset_temporal_state(self):
        """Reset temporal state for new session"""
        self.head_analyzer.movement_history.clear()
        logger.info("NMM temporal state reset")
