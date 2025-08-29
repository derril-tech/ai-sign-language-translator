import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceFrame:
    """Single frame confidence data"""
    timestamp: float
    pose_confidence: float
    gloss_confidence: float
    semantic_confidence: float
    overall_confidence: float
    frame_id: Optional[str] = None

@dataclass
class Hypothesis:
    """Translation hypothesis with confidence"""
    text: str
    confidence: float
    timestamp: float
    is_partial: bool
    source_components: Dict[str, float]  # confidence from each component

class ConfidenceManager:
    """Manages confidence scoring and hypothesis generation across the pipeline"""
    
    def __init__(self, 
                 history_size: int = 20,
                 commit_threshold: float = 0.8,
                 partial_threshold: float = 0.5):
        
        self.history_size = history_size
        self.commit_threshold = commit_threshold
        self.partial_threshold = partial_threshold
        
        # Session-specific confidence tracking
        self.session_confidence: Dict[str, deque] = {}
        self.session_hypotheses: Dict[str, List[Hypothesis]] = {}
        
        # Component weights for overall confidence
        self.component_weights = {
            'pose': 0.25,
            'gloss': 0.30,
            'semantic': 0.35,
            'rag': 0.10
        }
        
        logger.info(f"ConfidenceManager initialized with commit_threshold={commit_threshold}")
    
    def update_confidence(self, 
                         session_id: str,
                         timestamp: float,
                         pose_confidence: float = 0.0,
                         gloss_confidence: float = 0.0,
                         semantic_confidence: float = 0.0,
                         rag_confidence: float = 0.0,
                         frame_id: Optional[str] = None) -> float:
        """Update confidence for a session and return smoothed overall confidence"""
        try:
            # Calculate overall confidence
            overall_confidence = (
                pose_confidence * self.component_weights['pose'] +
                gloss_confidence * self.component_weights['gloss'] +
                semantic_confidence * self.component_weights['semantic'] +
                rag_confidence * self.component_weights['rag']
            )
            
            # Create confidence frame
            frame = ConfidenceFrame(
                timestamp=timestamp,
                pose_confidence=pose_confidence,
                gloss_confidence=gloss_confidence,
                semantic_confidence=semantic_confidence,
                overall_confidence=overall_confidence,
                frame_id=frame_id
            )
            
            # Initialize session if needed
            if session_id not in self.session_confidence:
                self.session_confidence[session_id] = deque(maxlen=self.history_size)
                self.session_hypotheses[session_id] = []
            
            # Add to history
            self.session_confidence[session_id].append(frame)
            
            # Return smoothed confidence
            return self._calculate_smoothed_confidence(session_id)
            
        except Exception as e:
            logger.error(f"Error updating confidence: {e}")
            return 0.0
    
    def generate_hypothesis(self, 
                          session_id: str,
                          text: str,
                          timestamp: float,
                          component_confidences: Dict[str, float]) -> Hypothesis:
        """Generate hypothesis with confidence assessment"""
        try:
            # Calculate overall confidence
            overall_confidence = sum(
                component_confidences.get(comp, 0.0) * weight
                for comp, weight in self.component_weights.items()
            )
            
            # Determine if partial or commit
            is_partial = overall_confidence < self.commit_threshold
            
            # Create hypothesis
            hypothesis = Hypothesis(
                text=text,
                confidence=overall_confidence,
                timestamp=timestamp,
                is_partial=is_partial,
                source_components=component_confidences.copy()
            )
            
            # Add to session hypotheses
            if session_id not in self.session_hypotheses:
                self.session_hypotheses[session_id] = []
            
            self.session_hypotheses[session_id].append(hypothesis)
            
            # Limit hypothesis history
            if len(self.session_hypotheses[session_id]) > 50:
                self.session_hypotheses[session_id] = self.session_hypotheses[session_id][-50:]
            
            return hypothesis
            
        except Exception as e:
            logger.error(f"Error generating hypothesis: {e}")
            return Hypothesis(
                text=text,
                confidence=0.0,
                timestamp=timestamp,
                is_partial=True,
                source_components={}
            )
    
    def get_partial_hypothesis(self, session_id: str) -> Optional[str]:
        """Get current partial hypothesis"""
        try:
            if session_id not in self.session_hypotheses:
                return None
            
            hypotheses = self.session_hypotheses[session_id]
            if not hypotheses:
                return None
            
            # Get recent partial hypotheses
            recent_partials = [
                h for h in hypotheses[-10:] 
                if h.is_partial and h.confidence >= self.partial_threshold
            ]
            
            if recent_partials:
                # Return the most confident recent partial
                best_partial = max(recent_partials, key=lambda h: h.confidence)
                return best_partial.text
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting partial hypothesis: {e}")
            return None
    
    def get_commit_hypothesis(self, session_id: str) -> Optional[str]:
        """Get current commit hypothesis"""
        try:
            if session_id not in self.session_hypotheses:
                return None
            
            hypotheses = self.session_hypotheses[session_id]
            if not hypotheses:
                return None
            
            # Get recent commit hypotheses
            recent_commits = [
                h for h in hypotheses[-5:] 
                if not h.is_partial and h.confidence >= self.commit_threshold
            ]
            
            if recent_commits:
                # Return the most recent commit
                return recent_commits[-1].text
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting commit hypothesis: {e}")
            return None
    
    def _calculate_smoothed_confidence(self, session_id: str) -> float:
        """Calculate smoothed confidence using temporal filtering"""
        try:
            if session_id not in self.session_confidence:
                return 0.0
            
            frames = list(self.session_confidence[session_id])
            if not frames:
                return 0.0
            
            # Simple exponential smoothing
            alpha = 0.3  # Smoothing factor
            smoothed = frames[0].overall_confidence
            
            for frame in frames[1:]:
                smoothed = alpha * frame.overall_confidence + (1 - alpha) * smoothed
            
            return smoothed
            
        except Exception as e:
            logger.error(f"Error calculating smoothed confidence: {e}")
            return 0.0
    
    def get_confidence_trend(self, session_id: str, window_size: int = 10) -> str:
        """Get confidence trend (improving, declining, stable)"""
        try:
            if session_id not in self.session_confidence:
                return "unknown"
            
            frames = list(self.session_confidence[session_id])
            if len(frames) < window_size:
                return "insufficient_data"
            
            recent_frames = frames[-window_size:]
            confidences = [f.overall_confidence for f in recent_frames]
            
            # Calculate trend using linear regression slope
            x = np.arange(len(confidences))
            slope = np.polyfit(x, confidences, 1)[0]
            
            if slope > 0.01:
                return "improving"
            elif slope < -0.01:
                return "declining"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating confidence trend: {e}")
            return "unknown"
    
    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        try:
            if session_id not in self.session_confidence:
                return {}
            
            frames = list(self.session_confidence[session_id])
            hypotheses = self.session_hypotheses.get(session_id, [])
            
            if not frames:
                return {}
            
            # Calculate statistics
            confidences = [f.overall_confidence for f in frames]
            
            stats = {
                "session_id": session_id,
                "total_frames": len(frames),
                "total_hypotheses": len(hypotheses),
                "avg_confidence": np.mean(confidences),
                "min_confidence": np.min(confidences),
                "max_confidence": np.max(confidences),
                "confidence_std": np.std(confidences),
                "trend": self.get_confidence_trend(session_id),
                "partial_hypotheses": sum(1 for h in hypotheses if h.is_partial),
                "commit_hypotheses": sum(1 for h in hypotheses if not h.is_partial),
                "current_smoothed_confidence": self._calculate_smoothed_confidence(session_id)
            }
            
            # Component-specific statistics
            for component in ['pose', 'gloss', 'semantic']:
                component_confidences = [getattr(f, f"{component}_confidence") for f in frames]
                stats[f"{component}_avg_confidence"] = np.mean(component_confidences)
                stats[f"{component}_std_confidence"] = np.std(component_confidences)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting session statistics: {e}")
            return {}
    
    def reset_session(self, session_id: str):
        """Reset confidence tracking for a session"""
        try:
            if session_id in self.session_confidence:
                del self.session_confidence[session_id]
            if session_id in self.session_hypotheses:
                del self.session_hypotheses[session_id]
            
            logger.info(f"Reset confidence tracking for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error resetting session: {e}")
    
    def adjust_thresholds(self, 
                         commit_threshold: Optional[float] = None,
                         partial_threshold: Optional[float] = None):
        """Adjust confidence thresholds"""
        try:
            if commit_threshold is not None:
                self.commit_threshold = max(0.0, min(1.0, commit_threshold))
                logger.info(f"Updated commit threshold to {self.commit_threshold}")
            
            if partial_threshold is not None:
                self.partial_threshold = max(0.0, min(1.0, partial_threshold))
                logger.info(f"Updated partial threshold to {self.partial_threshold}")
                
        except Exception as e:
            logger.error(f"Error adjusting thresholds: {e}")
    
    def get_confidence_distribution(self, session_id: str) -> Dict[str, int]:
        """Get confidence distribution for analysis"""
        try:
            if session_id not in self.session_confidence:
                return {}
            
            frames = list(self.session_confidence[session_id])
            confidences = [f.overall_confidence for f in frames]
            
            # Create bins
            bins = {
                "very_low": 0,    # 0.0 - 0.2
                "low": 0,         # 0.2 - 0.4
                "medium": 0,      # 0.4 - 0.6
                "high": 0,        # 0.6 - 0.8
                "very_high": 0    # 0.8 - 1.0
            }
            
            for conf in confidences:
                if conf < 0.2:
                    bins["very_low"] += 1
                elif conf < 0.4:
                    bins["low"] += 1
                elif conf < 0.6:
                    bins["medium"] += 1
                elif conf < 0.8:
                    bins["high"] += 1
                else:
                    bins["very_high"] += 1
            
            return bins
            
        except Exception as e:
            logger.error(f"Error getting confidence distribution: {e}")
            return {}
