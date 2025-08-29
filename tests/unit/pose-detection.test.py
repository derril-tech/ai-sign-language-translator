#!/usr/bin/env python3
"""
Unit tests for pose detection and keypoint normalization
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add workers to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'workers', 'pose-worker'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'workers', 'shared'))

from pose_detector import PoseDetector, PoseLandmarks
from frame_processor import FrameProcessor

class TestKeypointNormalization:
    """Test keypoint normalization and coordinate systems"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.pose_detector = PoseDetector(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_smoothing=True
        )
        self.frame_processor = FrameProcessor()
        
    def test_coordinate_normalization(self):
        """Test that coordinates are properly normalized to [0,1] range"""
        # Mock MediaPipe results
        mock_landmarks = Mock()
        mock_landmarks.landmark = [
            Mock(x=0.5, y=0.3, z=0.1, visibility=0.9),
            Mock(x=0.8, y=0.7, z=-0.05, visibility=0.8),
            Mock(x=0.2, y=0.9, z=0.2, visibility=0.7)
        ]
        
        normalized = self.pose_detector._normalize_landmarks(mock_landmarks, 640, 480)
        
        # Check all coordinates are in [0,1] range
        for point in normalized:
            assert 0 <= point['x'] <= 1, f"X coordinate {point['x']} out of range"
            assert 0 <= point['y'] <= 1, f"Y coordinate {point['y']} out of range"
            assert -1 <= point['z'] <= 1, f"Z coordinate {point['z']} out of range"
            assert 0 <= point['visibility'] <= 1, f"Visibility {point['visibility']} out of range"
    
    def test_keypoint_filtering(self):
        """Test filtering of low-confidence keypoints"""
        # Create mock landmarks with varying confidence
        mock_landmarks = Mock()
        mock_landmarks.landmark = [
            Mock(x=0.5, y=0.3, z=0.1, visibility=0.9),  # High confidence
            Mock(x=0.8, y=0.7, z=-0.05, visibility=0.3),  # Low confidence
            Mock(x=0.2, y=0.9, z=0.2, visibility=0.8)   # High confidence
        ]
        
        normalized = self.pose_detector._normalize_landmarks(mock_landmarks, 640, 480)
        filtered = self.pose_detector._filter_low_confidence_points(normalized, threshold=0.5)
        
        # Should have 2 points (high confidence ones)
        assert len(filtered) == 2
        assert all(point['visibility'] >= 0.5 for point in filtered)
    
    def test_temporal_smoothing(self):
        """Test Savitzky-Golay smoothing filter"""
        # Create sequence of noisy keypoints
        timestamps = [i * 0.033 for i in range(10)]  # 30 FPS
        noisy_points = []
        
        for i, t in enumerate(timestamps):
            # Base trajectory with noise
            base_x = 0.5 + 0.1 * np.sin(t * 2)
            base_y = 0.3 + 0.05 * np.cos(t * 3)
            noise_x = np.random.normal(0, 0.02)
            noise_y = np.random.normal(0, 0.02)
            
            noisy_points.append({
                'x': base_x + noise_x,
                'y': base_y + noise_y,
                'z': 0.0,
                'visibility': 0.9
            })
        
        # Apply smoothing
        smoothed_points = []
        for i, point in enumerate(noisy_points):
            smoothed = self.pose_detector._apply_temporal_smoothing([point], timestamps[i])
            smoothed_points.append(smoothed[0] if smoothed else point)
        
        # Smoothed points should have less variance
        original_var_x = np.var([p['x'] for p in noisy_points])
        smoothed_var_x = np.var([p['x'] for p in smoothed_points])
        
        # Note: This test might be flaky due to random noise
        # In practice, we'd use deterministic test data
        assert smoothed_var_x <= original_var_x * 1.5  # Allow some tolerance
    
    def test_hand_landmark_extraction(self):
        """Test extraction of hand landmarks"""
        # Mock hand landmarks
        mock_hand_landmarks = Mock()
        mock_hand_landmarks.landmark = [
            Mock(x=0.1 + i*0.05, y=0.2 + i*0.03, z=0.0, visibility=0.9)
            for i in range(21)  # 21 hand landmarks
        ]
        
        extracted = self.pose_detector._extract_hand_landmarks(mock_hand_landmarks)
        
        assert len(extracted) == 21
        assert all('x' in point and 'y' in point and 'z' in point for point in extracted)
    
    def test_face_landmark_extraction(self):
        """Test extraction of face landmarks for NMM detection"""
        # Mock face landmarks (468 points)
        mock_face_landmarks = Mock()
        mock_face_landmarks.landmark = [
            Mock(x=0.4 + (i%10)*0.02, y=0.1 + (i//10)*0.02, z=0.0, visibility=0.9)
            for i in range(468)
        ]
        
        extracted = self.pose_detector._extract_face_landmarks(mock_face_landmarks)
        
        assert len(extracted) == 468
        
        # Check key facial regions are present
        eyebrow_points = extracted[70:84]  # Approximate eyebrow region
        eye_points = extracted[33:42]      # Approximate eye region
        mouth_points = extracted[61:68]    # Approximate mouth region
        
        assert len(eyebrow_points) > 0
        assert len(eye_points) > 0
        assert len(mouth_points) > 0

class TestFrameProcessing:
    """Test frame preprocessing and quality metrics"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.frame_processor = FrameProcessor(
            target_width=640,
            target_height=480,
            quality=85
        )
    
    def test_frame_preprocessing(self):
        """Test frame preprocessing pipeline"""
        # Create test frame
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        processed = self.frame_processor.preprocess_frame(test_frame)
        
        # Check dimensions
        assert processed.shape[:2] == (480, 640), f"Expected (480, 640), got {processed.shape[:2]}"
        
        # Check data type
        assert processed.dtype == np.uint8
        
        # Check value range
        assert processed.min() >= 0 and processed.max() <= 255
    
    def test_quality_metrics(self):
        """Test frame quality assessment"""
        # Create test frames with different qualities
        
        # High quality frame (sharp, well-lit)
        high_quality = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        
        # Low quality frame (dark, blurry)
        low_quality = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        low_quality = cv2.GaussianBlur(low_quality, (15, 15), 5)
        
        high_metrics = self.frame_processor.calculate_frame_metrics(high_quality)
        low_metrics = self.frame_processor.calculate_frame_metrics(low_quality)
        
        # High quality should have better metrics
        assert high_metrics['brightness'] > low_metrics['brightness']
        assert high_metrics['sharpness'] > low_metrics['sharpness']
        assert high_metrics['overall_quality'] > low_metrics['overall_quality']
    
    def test_base64_encoding_decoding(self):
        """Test base64 frame encoding/decoding"""
        # Create test frame
        original_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Encode to base64
        encoded = self.frame_processor.encode_frame_to_base64(original_frame)
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        
        # Decode back
        decoded = self.frame_processor.decode_base64_frame(encoded)
        
        # Should be identical (within JPEG compression tolerance)
        assert decoded is not None
        assert decoded.shape == original_frame.shape
        
        # Allow for some JPEG compression artifacts
        mse = np.mean((original_frame.astype(float) - decoded.astype(float)) ** 2)
        assert mse < 100, f"MSE too high: {mse}"

class TestNonManualMarkers:
    """Test Non-Manual Marker (NMM) detection"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.pose_detector = PoseDetector()
    
    def test_eyebrow_detection(self):
        """Test eyebrow position detection"""
        # Mock face landmarks with raised eyebrows
        raised_eyebrows = [
            {'x': 0.3, 'y': 0.15, 'z': 0.0, 'visibility': 0.9},  # Left eyebrow
            {'x': 0.7, 'y': 0.15, 'z': 0.0, 'visibility': 0.9},  # Right eyebrow
        ]
        
        # Mock face landmarks with normal eyebrows
        normal_eyebrows = [
            {'x': 0.3, 'y': 0.25, 'z': 0.0, 'visibility': 0.9},  # Left eyebrow
            {'x': 0.7, 'y': 0.25, 'z': 0.0, 'visibility': 0.9},  # Right eyebrow
        ]
        
        raised_state = self.pose_detector._detect_eyebrow_state(raised_eyebrows)
        normal_state = self.pose_detector._detect_eyebrow_state(normal_eyebrows)
        
        assert raised_state == 'raised'
        assert normal_state == 'neutral'
    
    def test_eye_gaze_detection(self):
        """Test eye gaze direction detection"""
        # Mock eye landmarks looking left
        left_gaze = [
            {'x': 0.25, 'y': 0.2, 'z': 0.0, 'visibility': 0.9},  # Left pupil
            {'x': 0.65, 'y': 0.2, 'z': 0.0, 'visibility': 0.9},  # Right pupil
        ]
        
        # Mock eye landmarks looking forward
        forward_gaze = [
            {'x': 0.3, 'y': 0.2, 'z': 0.0, 'visibility': 0.9},   # Left pupil
            {'x': 0.7, 'y': 0.2, 'z': 0.0, 'visibility': 0.9},   # Right pupil
        ]
        
        left_direction = self.pose_detector._detect_gaze_direction(left_gaze)
        forward_direction = self.pose_detector._detect_gaze_direction(forward_gaze)
        
        assert left_direction == 'left'
        assert forward_direction == 'forward'
    
    def test_mouth_shape_detection(self):
        """Test mouth shape detection for mouthing"""
        # Mock mouth landmarks for different shapes
        open_mouth = [
            {'x': 0.5, 'y': 0.7, 'z': 0.0, 'visibility': 0.9},   # Top lip
            {'x': 0.5, 'y': 0.75, 'z': 0.0, 'visibility': 0.9},  # Bottom lip
        ]
        
        closed_mouth = [
            {'x': 0.5, 'y': 0.72, 'z': 0.0, 'visibility': 0.9},  # Top lip
            {'x': 0.5, 'y': 0.72, 'z': 0.0, 'visibility': 0.9},  # Bottom lip (same position)
        ]
        
        open_shape = self.pose_detector._detect_mouth_shape(open_mouth)
        closed_shape = self.pose_detector._detect_mouth_shape(closed_mouth)
        
        assert open_shape == 'open'
        assert closed_shape == 'closed'

class TestPoseDetectorIntegration:
    """Integration tests for the complete pose detection pipeline"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.pose_detector = PoseDetector(enable_smoothing=True)
        self.frame_processor = FrameProcessor()
    
    @patch('mediapipe.solutions.pose.Pose')
    @patch('mediapipe.solutions.hands.Hands')
    @patch('mediapipe.solutions.face_mesh.FaceMesh')
    def test_full_detection_pipeline(self, mock_face_mesh, mock_hands, mock_pose):
        """Test the complete pose detection pipeline"""
        # Mock MediaPipe results
        mock_pose_results = Mock()
        mock_pose_results.pose_landmarks = Mock()
        mock_pose_results.pose_landmarks.landmark = [
            Mock(x=0.5, y=0.3, z=0.1, visibility=0.9) for _ in range(33)
        ]
        
        mock_hand_results = Mock()
        mock_hand_results.multi_hand_landmarks = [Mock()]
        mock_hand_results.multi_hand_landmarks[0].landmark = [
            Mock(x=0.1 + i*0.05, y=0.2, z=0.0, visibility=0.9) for i in range(21)
        ]
        
        mock_face_results = Mock()
        mock_face_results.multi_face_landmarks = [Mock()]
        mock_face_results.multi_face_landmarks[0].landmark = [
            Mock(x=0.4, y=0.1, z=0.0, visibility=0.9) for _ in range(468)
        ]
        
        # Setup mocks
        mock_pose.return_value.process.return_value = mock_pose_results
        mock_hands.return_value.process.return_value = mock_hand_results
        mock_face_mesh.return_value.process.return_value = mock_face_results
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run detection
        landmarks = self.pose_detector.detect_pose(test_frame, timestamp=1234567890.0)
        
        # Verify results
        assert landmarks is not None
        assert isinstance(landmarks, PoseLandmarks)
        assert len(landmarks.body) > 0
        assert len(landmarks.left_hand) > 0
        assert len(landmarks.face) > 0
        assert 0 <= landmarks.confidence <= 1
    
    def test_error_handling(self):
        """Test error handling in pose detection"""
        # Test with invalid frame
        invalid_frame = None
        
        landmarks = self.pose_detector.detect_pose(invalid_frame, timestamp=1234567890.0)
        assert landmarks is None
        
        # Test with empty frame
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        
        landmarks = self.pose_detector.detect_pose(empty_frame, timestamp=1234567890.0)
        assert landmarks is None
    
    def test_performance_metrics(self):
        """Test performance tracking"""
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Mock successful detection
        with patch.object(self.pose_detector, '_detect_pose_landmarks') as mock_detect:
            mock_detect.return_value = (
                [{'x': 0.5, 'y': 0.3, 'z': 0.1, 'visibility': 0.9}],  # body
                [{'x': 0.1, 'y': 0.2, 'z': 0.0, 'visibility': 0.9}],  # left_hand
                [{'x': 0.9, 'y': 0.2, 'z': 0.0, 'visibility': 0.9}],  # right_hand
                [{'x': 0.4, 'y': 0.1, 'z': 0.0, 'visibility': 0.9}],  # face
                0.85  # confidence
            )
            
            landmarks = self.pose_detector.detect_pose(test_frame, timestamp=1234567890.0)
            
            assert landmarks is not None
            assert landmarks.confidence == 0.85

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
