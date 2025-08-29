from typing import Dict, List, Any, Optional, Tuple
import time
import logging
from dataclasses import dataclass, asdict
from collections import deque
import json

logger = logging.getLogger(__name__)

@dataclass
class ContextFrame:
    """Single frame of context information"""
    timestamp: float
    gloss_sequence: List[Dict[str, Any]]
    spatial_roles: Dict[str, Any]
    semantic_content: Optional[str] = None
    confidence: float = 0.0
    frame_id: Optional[str] = None

@dataclass
class ConversationTurn:
    """A turn in the conversation"""
    turn_id: str
    start_time: float
    end_time: Optional[float]
    frames: List[ContextFrame]
    summary: Optional[str] = None
    speaker_role: str = "signer"

@dataclass
class EntityReference:
    """Spatial entity reference tracking"""
    entity_id: str
    spatial_location: Tuple[float, float, float]
    first_mentioned: float
    last_mentioned: float
    reference_count: int
    semantic_role: Optional[str] = None
    description: Optional[str] = None

class ContextManager:
    """Manages conversational context and memory"""
    
    def __init__(self, max_history: int = 100, context_window: int = 10):
        self.max_history = max_history
        self.context_window = context_window
        
        # Session-specific context storage
        self.session_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Global entity tracking
        self.entity_references: Dict[str, EntityReference] = {}
        
        logger.info(f"ContextManager initialized with max_history={max_history}, context_window={context_window}")
    
    def update_context(self, 
                      session_id: str,
                      gloss_sequence: List[Dict[str, Any]],
                      spatial_roles: Dict[str, Any],
                      timestamp: float,
                      frame_id: Optional[str] = None,
                      semantic_content: Optional[str] = None,
                      confidence: float = 0.0) -> None:
        """Update context with new frame information"""
        try:
            # Initialize session context if not exists
            if session_id not in self.session_contexts:
                self.session_contexts[session_id] = {
                    'frames': deque(maxlen=self.max_history),
                    'turns': [],
                    'current_turn': None,
                    'discourse_state': {},
                    'topic_tracking': {},
                    'entity_references': {},
                    'session_start': timestamp
                }
            
            context = self.session_contexts[session_id]
            
            # Create new context frame
            frame = ContextFrame(
                timestamp=timestamp,
                gloss_sequence=gloss_sequence,
                spatial_roles=spatial_roles,
                semantic_content=semantic_content,
                confidence=confidence,
                frame_id=frame_id
            )
            
            # Add frame to history
            context['frames'].append(frame)
            
            # Update current turn
            self._update_current_turn(session_id, frame)
            
            # Update entity references
            self._update_entity_references(session_id, spatial_roles, timestamp)
            
            # Update discourse state
            self._update_discourse_state(session_id, gloss_sequence, spatial_roles)
            
            # Update topic tracking
            self._update_topic_tracking(session_id, gloss_sequence, semantic_content)
            
            logger.debug(f"Updated context for session {session_id} at timestamp {timestamp}")
            
        except Exception as e:
            logger.error(f"Error updating context: {e}")
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get current context for a session"""
        try:
            if session_id not in self.session_contexts:
                return self._create_empty_context()
            
            context = self.session_contexts[session_id]
            
            # Get recent frames within context window
            recent_frames = list(context['frames'])[-self.context_window:]
            
            # Build context summary
            context_summary = {
                'session_id': session_id,
                'frame_count': len(context['frames']),
                'recent_frames': [asdict(frame) for frame in recent_frames],
                'current_turn': context.get('current_turn'),
                'discourse_state': context.get('discourse_state', {}),
                'topic_tracking': context.get('topic_tracking', {}),
                'entity_references': context.get('entity_references', {}),
                'session_duration': self._calculate_session_duration(session_id),
                'conversation_flow': self._analyze_conversation_flow(recent_frames),
                'semantic_coherence': self._calculate_semantic_coherence(recent_frames)
            }
            
            return context_summary
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return self._create_empty_context()
    
    def _update_current_turn(self, session_id: str, frame: ContextFrame) -> None:
        """Update the current conversation turn"""
        try:
            context = self.session_contexts[session_id]
            
            # Check if we need to start a new turn
            if context['current_turn'] is None:
                # Start new turn
                turn = ConversationTurn(
                    turn_id=f"turn_{len(context['turns']) + 1}",
                    start_time=frame.timestamp,
                    end_time=None,
                    frames=[frame],
                    speaker_role="signer"
                )
                context['current_turn'] = turn
            else:
                # Add frame to current turn
                current_turn = context['current_turn']
                current_turn.frames.append(frame)
                
                # Check if turn should end (pause detection)
                if self._detect_turn_boundary(current_turn.frames):
                    # End current turn
                    current_turn.end_time = frame.timestamp
                    current_turn.summary = self._generate_turn_summary(current_turn.frames)
                    
                    # Add to turns history
                    context['turns'].append(current_turn)
                    
                    # Start new turn if there's more content
                    if frame.gloss_sequence:
                        context['current_turn'] = ConversationTurn(
                            turn_id=f"turn_{len(context['turns']) + 1}",
                            start_time=frame.timestamp,
                            end_time=None,
                            frames=[frame],
                            speaker_role="signer"
                        )
                    else:
                        context['current_turn'] = None
            
        except Exception as e:
            logger.error(f"Error updating current turn: {e}")
    
    def _update_entity_references(self, 
                                session_id: str,
                                spatial_roles: Dict[str, Any],
                                timestamp: float) -> None:
        """Update spatial entity references"""
        try:
            context = self.session_contexts[session_id]
            
            # Process spatial roles
            roles = spatial_roles.get('spatial_roles', [])
            for role in roles:
                if isinstance(role, dict) and 'position' in role:
                    # Create entity ID based on spatial location
                    pos = role['position']
                    entity_id = f"entity_{pos[0]:.2f}_{pos[1]:.2f}"
                    
                    if entity_id in context['entity_references']:
                        # Update existing reference
                        ref = context['entity_references'][entity_id]
                        ref.last_mentioned = timestamp
                        ref.reference_count += 1
                    else:
                        # Create new reference
                        ref = EntityReference(
                            entity_id=entity_id,
                            spatial_location=tuple(pos),
                            first_mentioned=timestamp,
                            last_mentioned=timestamp,
                            reference_count=1,
                            semantic_role=role.get('role')
                        )
                        context['entity_references'][entity_id] = ref
            
        except Exception as e:
            logger.error(f"Error updating entity references: {e}")
    
    def _update_discourse_state(self, 
                              session_id: str,
                              gloss_sequence: List[Dict[str, Any]],
                              spatial_roles: Dict[str, Any]) -> None:
        """Update discourse state tracking"""
        try:
            context = self.session_contexts[session_id]
            discourse_state = context.setdefault('discourse_state', {})
            
            # Track question-answer patterns
            has_question_marker = any(
                gloss.get('token', '').endswith('?') or 'WH-' in gloss.get('token', '')
                for gloss in gloss_sequence
                if isinstance(gloss, dict)
            )
            
            if has_question_marker:
                discourse_state['last_question_time'] = gloss_sequence[0].get('start_time', 0)
                discourse_state['expecting_answer'] = True
            
            # Track negation
            has_negation = any(
                gloss.get('token', '') in ['NOT', 'NO', 'NEVER', 'NOTHING']
                for gloss in gloss_sequence
                if isinstance(gloss, dict)
            )
            
            if has_negation:
                discourse_state['negation_context'] = True
                discourse_state['negation_scope'] = len(gloss_sequence)
            
            # Track spatial reference establishment
            if spatial_roles.get('spatial_roles'):
                discourse_state['active_spatial_frame'] = True
                discourse_state['spatial_complexity'] = len(spatial_roles.get('spatial_roles', []))
            
        except Exception as e:
            logger.error(f"Error updating discourse state: {e}")
    
    def _update_topic_tracking(self, 
                             session_id: str,
                             gloss_sequence: List[Dict[str, Any]],
                             semantic_content: Optional[str]) -> None:
        """Update topic tracking"""
        try:
            context = self.session_contexts[session_id]
            topic_tracking = context.setdefault('topic_tracking', {
                'current_topics': [],
                'topic_transitions': [],
                'semantic_fields': {}
            })
            
            # Extract potential topics from gloss sequence
            topics = []
            for gloss in gloss_sequence:
                if isinstance(gloss, dict):
                    token = gloss.get('token', '')
                    # Simple topic detection based on noun-like glosses
                    if token and token.isupper() and len(token) > 2:
                        topics.append(token)
            
            # Update current topics
            for topic in topics:
                if topic not in topic_tracking['current_topics']:
                    topic_tracking['current_topics'].append(topic)
                    
                    # Track topic transitions
                    if len(topic_tracking['current_topics']) > 1:
                        transition = {
                            'from': topic_tracking['current_topics'][-2],
                            'to': topic,
                            'timestamp': gloss_sequence[0].get('start_time', 0) if gloss_sequence else 0
                        }
                        topic_tracking['topic_transitions'].append(transition)
            
            # Limit topic history
            if len(topic_tracking['current_topics']) > 5:
                topic_tracking['current_topics'] = topic_tracking['current_topics'][-5:]
            
        except Exception as e:
            logger.error(f"Error updating topic tracking: {e}")
    
    def _detect_turn_boundary(self, frames: List[ContextFrame]) -> bool:
        """Detect if a turn boundary should be inserted"""
        try:
            if len(frames) < 2:
                return False
            
            # Check for temporal gap
            last_frame = frames[-1]
            prev_frame = frames[-2]
            
            time_gap = last_frame.timestamp - prev_frame.timestamp
            
            # Turn boundary if gap > 2 seconds
            if time_gap > 2.0:
                return True
            
            # Turn boundary if confidence drops significantly
            if (prev_frame.confidence > 0.7 and 
                last_frame.confidence < 0.3):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting turn boundary: {e}")
            return False
    
    def _generate_turn_summary(self, frames: List[ContextFrame]) -> str:
        """Generate summary for a conversation turn"""
        try:
            if not frames:
                return ""
            
            # Collect all gloss tokens
            all_tokens = []
            for frame in frames:
                for gloss in frame.gloss_sequence:
                    if isinstance(gloss, dict) and 'token' in gloss:
                        all_tokens.append(gloss['token'])
            
            # Simple summary - join unique tokens
            unique_tokens = list(dict.fromkeys(all_tokens))  # Preserve order, remove duplicates
            summary = " ".join(unique_tokens[:10])  # Limit to 10 tokens
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating turn summary: {e}")
            return ""
    
    def _calculate_session_duration(self, session_id: str) -> float:
        """Calculate session duration in seconds"""
        try:
            context = self.session_contexts[session_id]
            if not context['frames']:
                return 0.0
            
            start_time = context['session_start']
            last_frame = context['frames'][-1]
            
            return last_frame.timestamp - start_time
            
        except Exception as e:
            logger.error(f"Error calculating session duration: {e}")
            return 0.0
    
    def _analyze_conversation_flow(self, frames: List[ContextFrame]) -> Dict[str, Any]:
        """Analyze conversation flow patterns"""
        try:
            if not frames:
                return {}
            
            flow_analysis = {
                'frame_count': len(frames),
                'temporal_consistency': 0.0,
                'content_continuity': 0.0,
                'confidence_trend': 'stable'
            }
            
            # Calculate temporal consistency
            if len(frames) > 1:
                time_gaps = []
                for i in range(1, len(frames)):
                    gap = frames[i].timestamp - frames[i-1].timestamp
                    time_gaps.append(gap)
                
                # Temporal consistency based on gap variance
                if time_gaps:
                    avg_gap = sum(time_gaps) / len(time_gaps)
                    variance = sum((gap - avg_gap) ** 2 for gap in time_gaps) / len(time_gaps)
                    flow_analysis['temporal_consistency'] = max(0.0, 1.0 - variance)
            
            # Calculate confidence trend
            confidences = [frame.confidence for frame in frames if frame.confidence > 0]
            if len(confidences) > 1:
                trend = confidences[-1] - confidences[0]
                if trend > 0.1:
                    flow_analysis['confidence_trend'] = 'improving'
                elif trend < -0.1:
                    flow_analysis['confidence_trend'] = 'declining'
            
            return flow_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing conversation flow: {e}")
            return {}
    
    def _calculate_semantic_coherence(self, frames: List[ContextFrame]) -> float:
        """Calculate semantic coherence across frames"""
        try:
            if len(frames) < 2:
                return 1.0
            
            # Simple coherence based on token overlap
            all_tokens = set()
            frame_tokens = []
            
            for frame in frames:
                tokens = set()
                for gloss in frame.gloss_sequence:
                    if isinstance(gloss, dict) and 'token' in gloss:
                        token = gloss['token']
                        tokens.add(token)
                        all_tokens.add(token)
                frame_tokens.append(tokens)
            
            # Calculate pairwise overlap
            overlaps = []
            for i in range(len(frame_tokens) - 1):
                overlap = len(frame_tokens[i] & frame_tokens[i + 1])
                total = len(frame_tokens[i] | frame_tokens[i + 1])
                if total > 0:
                    overlaps.append(overlap / total)
            
            return sum(overlaps) / len(overlaps) if overlaps else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating semantic coherence: {e}")
            return 0.0
    
    def _create_empty_context(self) -> Dict[str, Any]:
        """Create empty context structure"""
        return {
            'session_id': '',
            'frame_count': 0,
            'recent_frames': [],
            'current_turn': None,
            'discourse_state': {},
            'topic_tracking': {},
            'entity_references': {},
            'session_duration': 0.0,
            'conversation_flow': {},
            'semantic_coherence': 0.0
        }
    
    def clear_session_context(self, session_id: str) -> None:
        """Clear context for a specific session"""
        if session_id in self.session_contexts:
            del self.session_contexts[session_id]
            logger.info(f"Cleared context for session {session_id}")
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        try:
            if session_id not in self.session_contexts:
                return {}
            
            context = self.session_contexts[session_id]
            
            summary = {
                'session_id': session_id,
                'total_frames': len(context['frames']),
                'total_turns': len(context['turns']),
                'session_duration': self._calculate_session_duration(session_id),
                'entity_count': len(context.get('entity_references', {})),
                'topic_count': len(context.get('topic_tracking', {}).get('current_topics', [])),
                'discourse_markers': context.get('discourse_state', {}),
                'conversation_quality': self._assess_conversation_quality(context)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            return {}
    
    def _assess_conversation_quality(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall conversation quality"""
        try:
            frames = list(context['frames'])
            
            if not frames:
                return {'overall_score': 0.0}
            
            # Calculate various quality metrics
            avg_confidence = sum(f.confidence for f in frames) / len(frames)
            
            # Temporal consistency
            temporal_score = 1.0
            if len(frames) > 1:
                gaps = [frames[i].timestamp - frames[i-1].timestamp for i in range(1, len(frames))]
                avg_gap = sum(gaps) / len(gaps)
                temporal_score = max(0.0, 1.0 - (max(gaps) - avg_gap) / avg_gap) if avg_gap > 0 else 1.0
            
            # Content richness
            total_glosses = sum(len(f.gloss_sequence) for f in frames)
            content_score = min(1.0, total_glosses / (len(frames) * 3))  # Expect ~3 glosses per frame
            
            overall_score = (avg_confidence + temporal_score + content_score) / 3
            
            return {
                'overall_score': overall_score,
                'confidence_score': avg_confidence,
                'temporal_score': temporal_score,
                'content_score': content_score
            }
            
        except Exception as e:
            logger.error(f"Error assessing conversation quality: {e}")
            return {'overall_score': 0.0}
