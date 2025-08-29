import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SpatialRole(Enum):
    """Spatial role types in sign language"""
    SUBJECT = "subject"
    OBJECT = "object"
    LOCATION = "location"
    CLASSIFIER = "classifier"
    REFERENCE = "reference"

@dataclass
class SpatialEntity:
    """Spatial entity with role and location"""
    role: SpatialRole
    location: Tuple[float, float, float]  # 3D position
    confidence: float
    frame_range: Tuple[int, int]
    description: str

class SpatialAnalyzer:
    """Analyzes spatial grammar and role labeling in sign language"""
    
    def __init__(self):
        # Spatial zones in signing space
        self.signing_space = {
            'center': (0.5, 0.5, 0.0),
            'left': (0.3, 0.5, 0.0),
            'right': (0.7, 0.5, 0.0),
            'upper': (0.5, 0.3, 0.0),
            'lower': (0.5, 0.7, 0.0)
        }
        
        # Tracking spatial references
        self.spatial_references = {}
        self.entity_history = []
        
        logger.info("SpatialAnalyzer initialized")
    
    def analyze_spatial_grammar(self, 
                              pose_landmarks: Dict[str, Any],
                              hand_landmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spatial grammar from pose and hand landmarks"""
        try:
            analysis = {
                'spatial_roles': [],
                'spatial_zones': {},
                'reference_tracking': {},
                'classifier_usage': {},
                'spatial_relationships': []
            }
            
            # Extract hand positions
            left_hand_pos = self._get_hand_position(hand_landmarks.get('left', []))
            right_hand_pos = self._get_hand_position(hand_landmarks.get('right', []))
            
            # Analyze spatial zones
            analysis['spatial_zones'] = self._analyze_spatial_zones(
                left_hand_pos, right_hand_pos
            )
            
            # Detect spatial roles
            analysis['spatial_roles'] = self._detect_spatial_roles(
                pose_landmarks, left_hand_pos, right_hand_pos
            )
            
            # Track spatial references
            analysis['reference_tracking'] = self._track_spatial_references(
                left_hand_pos, right_hand_pos
            )
            
            # Detect classifier usage
            analysis['classifier_usage'] = self._detect_classifier_usage(
                hand_landmarks
            )
            
            # Analyze spatial relationships
            analysis['spatial_relationships'] = self._analyze_spatial_relationships(
                left_hand_pos, right_hand_pos
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in spatial analysis: {e}")
            return {}
    
    def _get_hand_position(self, hand_landmarks: List[Dict]) -> Optional[Tuple[float, float, float]]:
        """Get 3D position of hand center"""
        try:
            if not hand_landmarks or len(hand_landmarks) < 21:
                return None
            
            # Use wrist position (landmark 0) as hand center
            wrist = hand_landmarks[0]
            return (
                wrist.get('x', 0.0),
                wrist.get('y', 0.0),
                wrist.get('z', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error getting hand position: {e}")
            return None
    
    def _analyze_spatial_zones(self, 
                             left_hand_pos: Optional[Tuple[float, float, float]],
                             right_hand_pos: Optional[Tuple[float, float, float]]) -> Dict[str, Any]:
        """Analyze which spatial zones are being used"""
        try:
            zones = {
                'left_hand_zone': None,
                'right_hand_zone': None,
                'active_zones': [],
                'zone_transitions': []
            }
            
            # Classify left hand zone
            if left_hand_pos:
                zones['left_hand_zone'] = self._classify_spatial_zone(left_hand_pos)
                zones['active_zones'].append(zones['left_hand_zone'])
            
            # Classify right hand zone
            if right_hand_pos:
                zones['right_hand_zone'] = self._classify_spatial_zone(right_hand_pos)
                zones['active_zones'].append(zones['right_hand_zone'])
            
            return zones
            
        except Exception as e:
            logger.error(f"Error analyzing spatial zones: {e}")
            return {}
    
    def _classify_spatial_zone(self, position: Tuple[float, float, float]) -> str:
        """Classify position into spatial zone"""
        try:
            x, y, z = position
            
            # Horizontal classification
            if x < 0.4:
                h_zone = "left"
            elif x > 0.6:
                h_zone = "right"
            else:
                h_zone = "center"
            
            # Vertical classification
            if y < 0.4:
                v_zone = "upper"
            elif y > 0.6:
                v_zone = "lower"
            else:
                v_zone = "middle"
            
            # Depth classification
            if z > 0.1:
                d_zone = "forward"
            elif z < -0.1:
                d_zone = "back"
            else:
                d_zone = "neutral"
            
            return f"{h_zone}_{v_zone}_{d_zone}"
            
        except Exception as e:
            logger.error(f"Error classifying spatial zone: {e}")
            return "unknown"
    
    def _detect_spatial_roles(self, 
                            pose_landmarks: Dict[str, Any],
                            left_hand_pos: Optional[Tuple[float, float, float]],
                            right_hand_pos: Optional[Tuple[float, float, float]]) -> List[Dict[str, Any]]:
        """Detect spatial roles being established"""
        try:
            roles = []
            
            # Analyze hand configurations for role establishment
            if left_hand_pos and right_hand_pos:
                # Check for subject-object relationships
                distance = np.linalg.norm(np.array(left_hand_pos) - np.array(right_hand_pos))
                
                if distance > 0.3:  # Hands far apart - potential role establishment
                    # Left hand could be subject
                    roles.append({
                        'role': SpatialRole.SUBJECT.value,
                        'hand': 'left',
                        'position': left_hand_pos,
                        'confidence': 0.7
                    })
                    
                    # Right hand could be object
                    roles.append({
                        'role': SpatialRole.OBJECT.value,
                        'hand': 'right',
                        'position': right_hand_pos,
                        'confidence': 0.7
                    })
            
            # Check for location establishment
            if left_hand_pos:
                # High positions often indicate locations
                if left_hand_pos[1] < 0.3:  # Upper area
                    roles.append({
                        'role': SpatialRole.LOCATION.value,
                        'hand': 'left',
                        'position': left_hand_pos,
                        'confidence': 0.6
                    })
            
            return roles
            
        except Exception as e:
            logger.error(f"Error detecting spatial roles: {e}")
            return []
    
    def _track_spatial_references(self, 
                                left_hand_pos: Optional[Tuple[float, float, float]],
                                right_hand_pos: Optional[Tuple[float, float, float]]) -> Dict[str, Any]:
        """Track spatial references across time"""
        try:
            tracking = {
                'established_references': [],
                'active_references': [],
                'reference_consistency': 0.0
            }
            
            # Update spatial reference history
            current_refs = []
            
            if left_hand_pos:
                zone = self._classify_spatial_zone(left_hand_pos)
                current_refs.append(('left', zone, left_hand_pos))
            
            if right_hand_pos:
                zone = self._classify_spatial_zone(right_hand_pos)
                current_refs.append(('right', zone, right_hand_pos))
            
            # Check for established references
            for hand, zone, pos in current_refs:
                ref_key = f"{hand}_{zone}"
                
                if ref_key in self.spatial_references:
                    # Update existing reference
                    self.spatial_references[ref_key]['count'] += 1
                    self.spatial_references[ref_key]['last_seen'] = pos
                else:
                    # New reference
                    self.spatial_references[ref_key] = {
                        'position': pos,
                        'count': 1,
                        'first_seen': pos,
                        'last_seen': pos
                    }
            
            # Filter established references (seen multiple times)
            for ref_key, ref_data in self.spatial_references.items():
                if ref_data['count'] >= 3:  # Threshold for establishment
                    tracking['established_references'].append({
                        'reference': ref_key,
                        'position': ref_data['position'],
                        'stability': min(ref_data['count'] / 10.0, 1.0)
                    })
            
            tracking['active_references'] = current_refs
            
            return tracking
            
        except Exception as e:
            logger.error(f"Error tracking spatial references: {e}")
            return {}
    
    def _detect_classifier_usage(self, hand_landmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Detect classifier handshape usage"""
        try:
            classifiers = {
                'detected_classifiers': [],
                'classifier_confidence': 0.0
            }
            
            # Analyze hand shapes for common classifiers
            for hand_type in ['left', 'right']:
                landmarks = hand_landmarks.get(hand_type, [])
                if landmarks and len(landmarks) == 21:
                    classifier_type = self._classify_handshape(landmarks)
                    if classifier_type:
                        classifiers['detected_classifiers'].append({
                            'hand': hand_type,
                            'classifier': classifier_type,
                            'confidence': 0.8
                        })
            
            if classifiers['detected_classifiers']:
                classifiers['classifier_confidence'] = 0.8
            
            return classifiers
            
        except Exception as e:
            logger.error(f"Error detecting classifiers: {e}")
            return {}
    
    def _classify_handshape(self, landmarks: List[Dict]) -> Optional[str]:
        """Classify handshape for classifier detection"""
        try:
            # Simple heuristics for common classifiers
            # In production, this would use a trained classifier
            
            # Calculate finger extensions
            finger_tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
            finger_mcp = [2, 5, 9, 13, 17]    # MCP joints
            
            extensions = []
            for tip, mcp in zip(finger_tips, finger_mcp):
                tip_pos = np.array([landmarks[tip]['y']])
                mcp_pos = np.array([landmarks[mcp]['y']])
                extensions.append(tip_pos[0] < mcp_pos[0])  # Extended if tip is higher
            
            # Classify based on extension pattern
            if extensions == [False, True, False, False, False]:  # Index extended
                return "CL:1"  # Person classifier
            elif extensions == [False, True, True, False, False]:  # Index and middle
                return "CL:V"  # Two-person classifier
            elif all(extensions[1:]):  # All fingers extended (except thumb)
                return "CL:5"  # Flat hand classifier
            
            return None
            
        except Exception as e:
            logger.error(f"Error classifying handshape: {e}")
            return None
    
    def _analyze_spatial_relationships(self, 
                                     left_hand_pos: Optional[Tuple[float, float, float]],
                                     right_hand_pos: Optional[Tuple[float, float, float]]) -> List[Dict[str, Any]]:
        """Analyze spatial relationships between entities"""
        try:
            relationships = []
            
            if left_hand_pos and right_hand_pos:
                # Calculate relative positions
                left_pos = np.array(left_hand_pos)
                right_pos = np.array(right_hand_pos)
                
                # Distance relationship
                distance = np.linalg.norm(left_pos - right_pos)
                
                if distance < 0.2:
                    relationships.append({
                        'type': 'proximity',
                        'entities': ['left_hand', 'right_hand'],
                        'description': 'close',
                        'confidence': 0.8
                    })
                elif distance > 0.5:
                    relationships.append({
                        'type': 'separation',
                        'entities': ['left_hand', 'right_hand'],
                        'description': 'far',
                        'confidence': 0.8
                    })
                
                # Vertical relationship
                if abs(left_pos[1] - right_pos[1]) > 0.2:
                    if left_pos[1] < right_pos[1]:
                        relationships.append({
                            'type': 'vertical',
                            'entities': ['left_hand', 'right_hand'],
                            'description': 'left_above_right',
                            'confidence': 0.7
                        })
                    else:
                        relationships.append({
                            'type': 'vertical',
                            'entities': ['left_hand', 'right_hand'],
                            'description': 'right_above_left',
                            'confidence': 0.7
                        })
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error analyzing spatial relationships: {e}")
            return []
    
    def reset_tracking(self):
        """Reset spatial reference tracking"""
        self.spatial_references.clear()
        self.entity_history.clear()
        logger.info("Spatial tracking reset")
