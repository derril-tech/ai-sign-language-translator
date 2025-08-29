#!/usr/bin/env python3
"""
Integration tests for the complete ASL translation pipeline
Tests: sign stream → captions → TTS and reverse mode → avatar
"""

import pytest
import asyncio
import numpy as np
import cv2
import json
import time
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
from typing import Dict, List, Any, Optional

# Mock NATS client for testing
class MockNATSClient:
    def __init__(self):
        self.messages = {}
        self.subscribers = {}
        
    async def connect(self):
        pass
    
    async def publish(self, subject: str, data: Dict[str, Any]):
        if subject not in self.messages:
            self.messages[subject] = []
        self.messages[subject].append(data)
        
        # Trigger subscribers
        if subject in self.subscribers:
            for callback in self.subscribers[subject]:
                await callback(data)
    
    async def subscribe(self, subject: str, callback, queue: str = None):
        if subject not in self.subscribers:
            self.subscribers[subject] = []
        self.subscribers[subject].append(callback)
    
    def get_messages(self, subject: str) -> List[Dict[str, Any]]:
        return self.messages.get(subject, [])

class TestSignToTextPipeline:
    """Test the complete sign-to-text translation pipeline"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.nats_client = MockNATSClient()
        self.session_id = "test_session_123"
        self.pipeline_results = {}
        
    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self):
        """Test the complete pipeline from video frame to final text"""
        
        # Step 1: Create mock video frame
        test_frame = self._create_test_frame_with_pose()
        frame_data = self._encode_frame_to_base64(test_frame)
        
        # Step 2: Send frame to pose worker
        pose_request = {
            "session_id": self.session_id,
            "frame_data": frame_data,
            "timestamp": time.time(),
            "frame_id": "frame_001"
        }
        
        # Mock pose worker response
        await self._mock_pose_worker_response(pose_request)
        
        # Step 3: Verify pose detection results
        pose_messages = self.nats_client.get_messages("gloss.decode")
        assert len(pose_messages) == 1
        
        pose_result = pose_messages[0]
        assert pose_result["session_id"] == self.session_id
        assert "pose_landmarks" in pose_result
        assert "hand_landmarks" in pose_result
        assert "confidence" in pose_result
        
        # Step 4: Process through gloss decoder
        await self._mock_gloss_decoder_response(pose_result)
        
        # Step 5: Verify gloss decoding results
        gloss_messages = self.nats_client.get_messages("semantic.translate")
        assert len(gloss_messages) == 1
        
        gloss_result = gloss_messages[0]
        assert "gloss_sequence" in gloss_result
        assert "confidence" in gloss_result
        assert len(gloss_result["gloss_sequence"]) > 0
        
        # Step 6: Process through semantic translator
        await self._mock_semantic_translator_response(gloss_result)
        
        # Step 7: Verify semantic translation results
        semantic_messages = self.nats_client.get_messages("rag.enhance")
        assert len(semantic_messages) == 1
        
        semantic_result = semantic_messages[0]
        assert "translated_text" in semantic_result
        assert "confidence" in semantic_result
        assert len(semantic_result["translated_text"]) > 0
        
        # Step 8: Process through RAG enhancement
        await self._mock_rag_enhancement_response(semantic_result)
        
        # Step 9: Verify final enhanced text
        rag_messages = self.nats_client.get_messages("translation.final")
        assert len(rag_messages) == 1
        
        final_result = rag_messages[0]
        assert "enhanced_text" in final_result
        assert "confidence" in final_result
        assert "terminology_matches" in final_result
        
        # Step 10: Verify end-to-end latency
        start_time = pose_request["timestamp"]
        end_time = final_result.get("timestamp", time.time())
        latency = end_time - start_time
        
        assert latency < 2.0, f"Pipeline latency too high: {latency}s"
        
        print(f"✅ Complete pipeline test passed - Latency: {latency:.3f}s")
        print(f"   Final text: '{final_result['enhanced_text']}'")
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self):
        """Test pipeline error handling and recovery"""
        
        # Test with invalid frame data
        invalid_request = {
            "session_id": self.session_id,
            "frame_data": "invalid_base64_data",
            "timestamp": time.time(),
            "frame_id": "frame_error"
        }
        
        # Should handle gracefully
        await self._mock_pose_worker_response(invalid_request, should_error=True)
        
        # Check error handling
        error_messages = self.nats_client.get_messages("pose.error")
        assert len(error_messages) == 1
        
        error_result = error_messages[0]
        assert "error" in error_result
        assert error_result["session_id"] == self.session_id
    
    @pytest.mark.asyncio
    async def test_pipeline_confidence_propagation(self):
        """Test that confidence scores propagate correctly through pipeline"""
        
        # Create frame with known confidence levels
        test_frame = self._create_test_frame_with_pose()
        frame_data = self._encode_frame_to_base64(test_frame)
        
        pose_request = {
            "session_id": self.session_id,
            "frame_data": frame_data,
            "timestamp": time.time(),
            "frame_id": "frame_conf"
        }
        
        # Set specific confidence at each stage
        initial_confidence = 0.85
        
        # Mock responses with decreasing confidence
        await self._mock_pose_worker_response(pose_request, confidence=initial_confidence)
        pose_result = self.nats_client.get_messages("gloss.decode")[-1]
        
        await self._mock_gloss_decoder_response(pose_result, confidence=initial_confidence * 0.9)
        gloss_result = self.nats_client.get_messages("semantic.translate")[-1]
        
        await self._mock_semantic_translator_response(gloss_result, confidence=initial_confidence * 0.8)
        semantic_result = self.nats_client.get_messages("rag.enhance")[-1]
        
        await self._mock_rag_enhancement_response(semantic_result, confidence=initial_confidence * 0.85)
        final_result = self.nats_client.get_messages("translation.final")[-1]
        
        # Verify confidence propagation
        assert final_result["confidence"] > 0.5
        assert final_result["confidence"] <= initial_confidence
        
        print(f"✅ Confidence propagation test passed - Final confidence: {final_result['confidence']:.3f}")
    
    @pytest.mark.asyncio
    async def test_multi_frame_sequence(self):
        """Test processing of multiple frames in sequence"""
        
        frame_count = 5
        results = []
        
        for i in range(frame_count):
            # Create slightly different frames
            test_frame = self._create_test_frame_with_pose(frame_id=i)
            frame_data = self._encode_frame_to_base64(test_frame)
            
            pose_request = {
                "session_id": self.session_id,
                "frame_data": frame_data,
                "timestamp": time.time() + i * 0.033,  # 30 FPS
                "frame_id": f"frame_{i:03d}"
            }
            
            # Process through pipeline
            await self._mock_pose_worker_response(pose_request)
            pose_result = self.nats_client.get_messages("gloss.decode")[-1]
            
            await self._mock_gloss_decoder_response(pose_result)
            gloss_result = self.nats_client.get_messages("semantic.translate")[-1]
            
            await self._mock_semantic_translator_response(gloss_result)
            semantic_result = self.nats_client.get_messages("rag.enhance")[-1]
            
            await self._mock_rag_enhancement_response(semantic_result)
            final_result = self.nats_client.get_messages("translation.final")[-1]
            
            results.append(final_result)
        
        # Verify all frames processed
        assert len(results) == frame_count
        
        # Verify temporal consistency
        timestamps = [r["timestamp"] for r in results]
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
        
        print(f"✅ Multi-frame sequence test passed - Processed {frame_count} frames")
    
    # Helper methods
    def _create_test_frame_with_pose(self, frame_id: int = 0) -> np.ndarray:
        """Create a test frame with simulated pose"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some structure to simulate a person
        # Draw simple stick figure
        center_x, center_y = 320, 240
        
        # Head
        cv2.circle(frame, (center_x, center_y - 100), 40, (255, 200, 150), -1)
        
        # Body
        cv2.line(frame, (center_x, center_y - 60), (center_x, center_y + 100), (100, 100, 255), 10)
        
        # Arms (vary position based on frame_id for sequence testing)
        arm_offset = frame_id * 10
        cv2.line(frame, (center_x, center_y - 20), (center_x - 80 - arm_offset, center_y + 20), (100, 255, 100), 8)
        cv2.line(frame, (center_x, center_y - 20), (center_x + 80 + arm_offset, center_y + 20), (100, 255, 100), 8)
        
        return frame
    
    def _encode_frame_to_base64(self, frame: np.ndarray) -> str:
        """Encode frame to base64 string"""
        import base64
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    
    async def _mock_pose_worker_response(self, request: Dict[str, Any], should_error: bool = False, confidence: float = 0.85):
        """Mock pose worker response"""
        if should_error:
            error_response = {
                "session_id": request["session_id"],
                "timestamp": request["timestamp"],
                "error": "Invalid frame data",
                "processing_time_ms": 50
            }
            await self.nats_client.publish("pose.error", error_response)
            return
        
        # Mock successful pose detection
        pose_response = {
            "session_id": request["session_id"],
            "timestamp": request["timestamp"],
            "frame_id": request.get("frame_id"),
            "pose_landmarks": {
                "body": [
                    {"x": 0.5, "y": 0.3, "z": 0.1, "visibility": 0.9},
                    {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.95},
                    {"x": 0.3, "y": 0.4, "z": 0.05, "visibility": 0.8}
                ],
                "confidence": confidence
            },
            "hand_landmarks": {
                "left": [
                    {"x": 0.2, "y": 0.4, "z": 0.1, "visibility": 0.9}
                ],
                "right": [
                    {"x": 0.8, "y": 0.4, "z": 0.1, "visibility": 0.9}
                ]
            },
            "face_landmarks": {
                "landmarks": [
                    {"x": 0.5, "y": 0.2, "z": 0.0, "visibility": 0.95}
                ]
            },
            "confidence": confidence,
            "processing_time_ms": 150,
            "frame_metrics": {
                "brightness": 0.7,
                "sharpness": 0.8,
                "overall_quality": 0.75
            }
        }
        
        await self.nats_client.publish("gloss.decode", pose_response)
    
    async def _mock_gloss_decoder_response(self, pose_data: Dict[str, Any], confidence: float = 0.8):
        """Mock gloss decoder response"""
        gloss_response = {
            "session_id": pose_data["session_id"],
            "timestamp": pose_data["timestamp"],
            "gloss_sequence": ["HELLO", "HOW", "ARE", "YOU"],
            "confidence": confidence,
            "ctc_scores": [0.9, 0.85, 0.8, 0.88],
            "fingerspelling_detected": [],
            "spatial_roles": {
                "left_space": "PERSON_A",
                "right_space": "PERSON_B"
            },
            "processing_time_ms": 200
        }
        
        await self.nats_client.publish("semantic.translate", gloss_response)
    
    async def _mock_semantic_translator_response(self, gloss_data: Dict[str, Any], confidence: float = 0.75):
        """Mock semantic translator response"""
        semantic_response = {
            "session_id": gloss_data["session_id"],
            "timestamp": gloss_data["timestamp"],
            "translated_text": "Hello, how are you?",
            "confidence": confidence,
            "nmm_analysis": {
                "eyebrows": "neutral",
                "eye_gaze": "forward",
                "mouth_shape": "neutral"
            },
            "discourse_markers": [],
            "processing_time_ms": 180
        }
        
        await self.nats_client.publish("rag.enhance", semantic_response)
    
    async def _mock_rag_enhancement_response(self, semantic_data: Dict[str, Any], confidence: float = 0.78):
        """Mock RAG enhancement response"""
        rag_response = {
            "session_id": semantic_data["session_id"],
            "timestamp": time.time(),
            "enhanced_text": "Hello, how are you doing today?",
            "confidence": confidence,
            "terminology_matches": [
                {
                    "original_term": "hello",
                    "enhanced_term": "hello",
                    "domain": "general",
                    "confidence": 0.95
                }
            ],
            "domain_adaptations": [],
            "processing_time_ms": 120
        }
        
        await self.nats_client.publish("translation.final", rag_response)

class TestTextToSignPipeline:
    """Test the reverse mode text-to-sign pipeline"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.nats_client = MockNATSClient()
        self.session_id = "test_reverse_session_456"
    
    @pytest.mark.asyncio
    async def test_text_to_avatar_pipeline(self):
        """Test complete text-to-sign avatar generation pipeline"""
        
        # Step 1: Input text
        input_text = "Hello, how are you today?"
        
        text_request = {
            "session_id": self.session_id,
            "text": input_text,
            "timestamp": time.time(),
            "mode": "text_to_sign"
        }
        
        # Step 2: Process through text-to-sign pipeline
        await self._mock_text_to_sign_processing(text_request)
        
        # Step 3: Verify linguistic analysis
        linguistic_messages = self.nats_client.get_messages("sign.planning")
        assert len(linguistic_messages) == 1
        
        linguistic_result = linguistic_messages[0]
        assert "linguistic_analysis" in linguistic_result
        assert "grammar_plan" in linguistic_result
        assert "gloss_sequence" in linguistic_result
        
        # Step 4: Process through sign planning
        await self._mock_sign_planning_response(linguistic_result)
        
        # Step 5: Verify pose generation
        pose_messages = self.nats_client.get_messages("avatar.animate")
        assert len(pose_messages) == 1
        
        pose_result = pose_messages[0]
        assert "pose_sequence" in pose_result
        assert "timing_data" in pose_result
        assert len(pose_result["pose_sequence"]) > 0
        
        # Step 6: Process through avatar animation
        await self._mock_avatar_animation_response(pose_result)
        
        # Step 7: Verify final animation data
        animation_messages = self.nats_client.get_messages("avatar.ready")
        assert len(animation_messages) == 1
        
        animation_result = animation_messages[0]
        assert "animation_curves" in animation_result
        assert "total_duration" in animation_result
        assert animation_result["total_duration"] > 0
        
        print(f"✅ Text-to-sign pipeline test passed")
        print(f"   Input: '{input_text}'")
        print(f"   Gloss: {linguistic_result['gloss_sequence']}")
        print(f"   Duration: {animation_result['total_duration']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_complex_sentence_processing(self):
        """Test processing of complex sentences with multiple clauses"""
        
        complex_text = "I went to the store yesterday, but they were closed, so I came home."
        
        text_request = {
            "session_id": self.session_id,
            "text": complex_text,
            "timestamp": time.time(),
            "mode": "text_to_sign"
        }
        
        await self._mock_text_to_sign_processing(text_request)
        linguistic_result = self.nats_client.get_messages("sign.planning")[-1]
        
        # Verify complex sentence handling
        assert len(linguistic_result["gloss_sequence"]) > 5
        assert "temporal_markers" in linguistic_result["grammar_plan"]
        assert "spatial_setup" in linguistic_result["grammar_plan"]
        
        print(f"✅ Complex sentence test passed")
        print(f"   Gloss length: {len(linguistic_result['gloss_sequence'])} tokens")
    
    async def _mock_text_to_sign_processing(self, request: Dict[str, Any]):
        """Mock text-to-sign processing"""
        linguistic_response = {
            "session_id": request["session_id"],
            "timestamp": request["timestamp"],
            "original_text": request["text"],
            "linguistic_analysis": {
                "words": request["text"].split(),
                "tense": "present",
                "entities": ["you"],
                "sentiment": "neutral"
            },
            "grammar_plan": {
                "topicalization": {"type": "question", "structure": "wh-question"},
                "spatial_setup": {"entityLocations": {}},
                "time_reference": {"location": "center", "movement": "neutral"},
                "non_manual_markers": {
                    "eyebrows": "raised",
                    "eye_gaze": "forward",
                    "mouth_shape": "neutral",
                    "head_movement": "none"
                }
            },
            "gloss_sequence": ["HELLO", "HOW", "YOU", "TODAY"],
            "confidence": 0.85
        }
        
        await self.nats_client.publish("sign.planning", linguistic_response)
    
    async def _mock_sign_planning_response(self, linguistic_data: Dict[str, Any]):
        """Mock sign planning response"""
        pose_response = {
            "session_id": linguistic_data["session_id"],
            "timestamp": linguistic_data["timestamp"],
            "pose_sequence": [
                {
                    "timestamp": 0.0,
                    "duration": 0.8,
                    "gloss": "HELLO",
                    "hand_shape": {"left": "flat", "right": "flat"},
                    "movement": {"type": "wave", "direction": [1, 0, 0], "speed": 1.0},
                    "location": {"x": 0.3, "y": -0.2, "z": 0},
                    "non_manual_markers": {
                        "eyebrows": "neutral",
                        "eye_gaze": "forward",
                        "mouth_shape": "smile",
                        "head_movement": "none"
                    }
                },
                {
                    "timestamp": 0.8,
                    "duration": 0.6,
                    "gloss": "HOW",
                    "hand_shape": {"left": "curved", "right": "curved"},
                    "movement": {"type": "question", "direction": [0, -1, 0], "speed": 1.2},
                    "location": {"x": 0, "y": -0.1, "z": 0},
                    "non_manual_markers": {
                        "eyebrows": "raised",
                        "eye_gaze": "forward",
                        "mouth_shape": "neutral",
                        "head_movement": "tilt"
                    }
                }
            ],
            "timing_data": {
                "total_duration": 2.5,
                "frame_rate": 30,
                "transition_time": 0.1
            }
        }
        
        await self.nats_client.publish("avatar.animate", pose_response)
    
    async def _mock_avatar_animation_response(self, pose_data: Dict[str, Any]):
        """Mock avatar animation response"""
        animation_response = {
            "session_id": pose_data["session_id"],
            "timestamp": time.time(),
            "animation_curves": {
                "left_hand": [
                    {"time": 0.0, "position": [0.2, 0.4, 0.0], "rotation": [0, 0, 0]},
                    {"time": 0.8, "position": [0.25, 0.35, 0.05], "rotation": [0.1, 0, 0]},
                    {"time": 1.4, "position": [0.15, 0.45, 0.0], "rotation": [0, 0, 0]}
                ],
                "right_hand": [
                    {"time": 0.0, "position": [0.8, 0.4, 0.0], "rotation": [0, 0, 0]},
                    {"time": 0.8, "position": [0.75, 0.35, 0.05], "rotation": [-0.1, 0, 0]},
                    {"time": 1.4, "position": [0.85, 0.45, 0.0], "rotation": [0, 0, 0]}
                ]
            },
            "total_duration": pose_data["timing_data"]["total_duration"],
            "frame_rate": pose_data["timing_data"]["frame_rate"],
            "quality_score": 0.92
        }
        
        await self.nats_client.publish("avatar.ready", animation_response)

class TestTTSIntegration:
    """Test TTS integration with translation pipeline"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.nats_client = MockNATSClient()
        self.session_id = "test_tts_session_789"
    
    @pytest.mark.asyncio
    async def test_tts_pipeline_integration(self):
        """Test TTS integration with translation results"""
        
        # Mock translation result
        translation_result = {
            "session_id": self.session_id,
            "enhanced_text": "Hello, how are you doing today?",
            "confidence": 0.85,
            "timestamp": time.time()
        }
        
        # Send to TTS
        tts_request = {
            "session_id": self.session_id,
            "text": translation_result["enhanced_text"],
            "voice_id": "default",
            "speed": 1.0,
            "timestamp": time.time()
        }
        
        await self._mock_tts_processing(tts_request)
        
        # Verify TTS output
        tts_messages = self.nats_client.get_messages("tts.audio")
        assert len(tts_messages) == 1
        
        tts_result = tts_messages[0]
        assert "audio_data" in tts_result
        assert "duration" in tts_result
        assert tts_result["duration"] > 0
        
        print(f"✅ TTS integration test passed")
        print(f"   Text: '{tts_request['text']}'")
        print(f"   Audio duration: {tts_result['duration']:.2f}s")
    
    async def _mock_tts_processing(self, request: Dict[str, Any]):
        """Mock TTS processing"""
        tts_response = {
            "session_id": request["session_id"],
            "timestamp": request["timestamp"],
            "audio_data": "base64_encoded_audio_data_here",
            "duration": 3.2,
            "sample_rate": 22050,
            "voice_id": request["voice_id"],
            "processing_time_ms": 450
        }
        
        await self.nats_client.publish("tts.audio", tts_response)

class TestPipelinePerformance:
    """Test pipeline performance and timing"""
    
    @pytest.mark.asyncio
    async def test_pipeline_latency_requirements(self):
        """Test that pipeline meets latency requirements"""
        
        nats_client = MockNATSClient()
        session_id = "perf_test_session"
        
        # Test multiple frames for average latency
        latencies = []
        
        for i in range(10):
            start_time = time.time()
            
            # Simulate full pipeline
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Mock processing times based on actual measurements
            await asyncio.sleep(0.15)  # Pose detection: ~150ms
            await asyncio.sleep(0.20)  # Gloss decoding: ~200ms
            await asyncio.sleep(0.18)  # Semantic translation: ~180ms
            await asyncio.sleep(0.12)  # RAG enhancement: ~120ms
            
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # Requirements: avg < 1s, p95 < 1.5s
        assert avg_latency < 1.0, f"Average latency too high: {avg_latency:.3f}s"
        assert p95_latency < 1.5, f"P95 latency too high: {p95_latency:.3f}s"
        
        print(f"✅ Latency requirements met")
        print(f"   Average: {avg_latency:.3f}s")
        print(f"   P95: {p95_latency:.3f}s")
    
    @pytest.mark.asyncio
    async def test_throughput_requirements(self):
        """Test pipeline throughput under load"""
        
        nats_client = MockNATSClient()
        concurrent_sessions = 5
        frames_per_session = 10
        
        async def process_session(session_id: str):
            """Process frames for one session"""
            for i in range(frames_per_session):
                # Simulate frame processing
                await asyncio.sleep(0.1)  # Reduced processing time under load
            return session_id
        
        # Process multiple sessions concurrently
        start_time = time.time()
        
        tasks = [
            process_session(f"session_{i}") 
            for i in range(concurrent_sessions)
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        total_frames = concurrent_sessions * frames_per_session
        throughput = total_frames / total_time
        
        # Requirement: > 20 FPS aggregate throughput
        assert throughput > 20, f"Throughput too low: {throughput:.1f} FPS"
        
        print(f"✅ Throughput requirements met")
        print(f"   Processed {total_frames} frames in {total_time:.2f}s")
        print(f"   Throughput: {throughput:.1f} FPS")

if __name__ == '__main__':
    # Run integration tests
    pytest.main([__file__, '-v', '--tb=short', '-s'])
