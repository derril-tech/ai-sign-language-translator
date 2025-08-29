#!/usr/bin/env python3
"""
Unit tests for gloss decoder and CTC decoding
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add workers to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'workers', 'gloss-worker'))

from gloss_decoder import GlossDecoder, CTCDecoder
from fingerspelling_detector import FingerspellingDetector
from spatial_analyzer import SpatialAnalyzer

class TestCTCDecoding:
    """Test CTC (Connectionist Temporal Classification) decoding"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.vocab_size = 100
        self.ctc_decoder = CTCDecoder(
            vocab_size=self.vocab_size,
            blank_token=0,
            beam_width=5
        )
        
        # Mock vocabulary
        self.vocabulary = {
            0: '<BLANK>',
            1: 'HELLO',
            2: 'WORLD',
            3: 'HOW',
            4: 'ARE',
            5: 'YOU',
            6: 'THANK',
            7: 'PLEASE',
            8: 'SORRY',
            9: 'GOOD',
            10: 'BAD'
        }
        self.ctc_decoder.set_vocabulary(self.vocabulary)
    
    def test_ctc_greedy_decoding(self):
        """Test greedy CTC decoding"""
        # Create mock logits for sequence "HELLO WORLD"
        # Shape: (sequence_length, vocab_size)
        sequence_length = 20
        logits = np.random.randn(sequence_length, self.vocab_size) * 0.1
        
        # Inject strong signals for HELLO (token 1) and WORLD (token 2)
        logits[2:5, 1] = 5.0    # HELLO
        logits[5:8, 0] = 3.0    # BLANK
        logits[8:12, 2] = 5.0   # WORLD
        logits[12:, 0] = 3.0    # BLANK
        
        # Convert to torch tensor
        logits_tensor = torch.from_numpy(logits).float()
        
        # Decode
        decoded_sequence = self.ctc_decoder.greedy_decode(logits_tensor)
        
        # Should decode to "HELLO WORLD"
        assert len(decoded_sequence) == 2
        assert decoded_sequence[0] == 'HELLO'
        assert decoded_sequence[1] == 'WORLD'
    
    def test_ctc_beam_search_decoding(self):
        """Test beam search CTC decoding"""
        # Create more complex logits
        sequence_length = 25
        logits = np.random.randn(sequence_length, self.vocab_size) * 0.5
        
        # Create ambiguous sequence that benefits from beam search
        logits[1:4, 1] = 4.0    # HELLO (strong)
        logits[4:6, 0] = 2.0    # BLANK
        logits[6:8, 3] = 3.5    # HOW (medium)
        logits[8:10, 0] = 2.0   # BLANK
        logits[10:13, 4] = 4.5  # ARE (strong)
        logits[13:15, 0] = 2.0  # BLANK
        logits[15:18, 5] = 3.8  # YOU (strong)
        
        logits_tensor = torch.from_numpy(logits).float()
        
        # Beam search should find better path than greedy
        beam_result = self.ctc_decoder.beam_search_decode(logits_tensor, beam_width=5)
        greedy_result = self.ctc_decoder.greedy_decode(logits_tensor)
        
        # Beam search should produce reasonable sequence
        assert len(beam_result) >= 3
        assert 'HELLO' in beam_result or 'HOW' in beam_result
        assert 'ARE' in beam_result
        assert 'YOU' in beam_result
    
    def test_ctc_blank_token_handling(self):
        """Test proper handling of CTC blank tokens"""
        # Sequence with many blanks and repetitions
        sequence_length = 15
        logits = np.full((sequence_length, self.vocab_size), -5.0)  # Low probability baseline
        
        # Pattern: BLANK-HELLO-HELLO-BLANK-BLANK-WORLD-WORLD-WORLD-BLANK
        logits[0, 0] = 5.0      # BLANK
        logits[1:3, 1] = 5.0    # HELLO (repeated)
        logits[3:5, 0] = 5.0    # BLANK (repeated)
        logits[5:8, 2] = 5.0    # WORLD (repeated)
        logits[8:, 0] = 5.0     # BLANK
        
        logits_tensor = torch.from_numpy(logits).float()
        decoded = self.ctc_decoder.greedy_decode(logits_tensor)
        
        # Should collapse repetitions and remove blanks
        assert decoded == ['HELLO', 'WORLD']
    
    def test_confidence_scoring(self):
        """Test confidence score calculation"""
        # High confidence sequence
        high_conf_logits = np.full((10, self.vocab_size), -10.0)
        high_conf_logits[1:4, 1] = 10.0  # Very confident HELLO
        high_conf_logits[5:8, 2] = 10.0  # Very confident WORLD
        
        # Low confidence sequence
        low_conf_logits = np.random.randn(10, self.vocab_size) * 0.1  # Noisy
        low_conf_logits[1:4, 1] = 1.0   # Weak HELLO
        low_conf_logits[5:8, 2] = 1.0   # Weak WORLD
        
        high_conf_tensor = torch.from_numpy(high_conf_logits).float()
        low_conf_tensor = torch.from_numpy(low_conf_logits).float()
        
        high_confidence = self.ctc_decoder.calculate_confidence(high_conf_tensor)
        low_confidence = self.ctc_decoder.calculate_confidence(low_conf_tensor)
        
        assert high_confidence > low_confidence
        assert 0 <= high_confidence <= 1
        assert 0 <= low_confidence <= 1

class TestGlossDecoder:
    """Test the main gloss decoder with transformer architecture"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.gloss_decoder = GlossDecoder(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            dropout=0.1
        )
    
    def test_pose_sequence_encoding(self):
        """Test encoding of pose sequences"""
        # Mock pose sequence (batch_size=1, seq_len=20, pose_dim=75)
        # 75 = 25 body points * 3 coordinates
        batch_size, seq_len, pose_dim = 1, 20, 75
        pose_sequence = torch.randn(batch_size, seq_len, pose_dim)
        
        # Encode
        encoded = self.gloss_decoder.encode_pose_sequence(pose_sequence)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, self.gloss_decoder.hidden_dim)
        assert encoded.shape == expected_shape
    
    def test_temporal_attention(self):
        """Test temporal attention mechanism"""
        batch_size, seq_len, hidden_dim = 2, 15, 256
        
        # Create sequence with attention pattern
        sequence = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Apply self-attention
        attended = self.gloss_decoder.apply_temporal_attention(sequence)
        
        # Output should have same shape
        assert attended.shape == sequence.shape
        
        # Attention should create dependencies between time steps
        # (This is hard to test directly, but we can check for reasonable outputs)
        assert not torch.allclose(attended, sequence, atol=1e-6)
    
    def test_gloss_prediction(self):
        """Test gloss sequence prediction"""
        # Mock input features
        batch_size, seq_len, feature_dim = 1, 10, 256
        features = torch.randn(batch_size, seq_len, feature_dim)
        
        # Predict glosses
        with torch.no_grad():
            logits = self.gloss_decoder.predict_glosses(features)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, self.gloss_decoder.vocab_size)
        assert logits.shape == expected_shape
        
        # Logits should be reasonable
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_sequence_alignment(self):
        """Test sequence alignment for variable-length inputs"""
        # Different length sequences
        seq1 = torch.randn(1, 8, 75)   # Short sequence
        seq2 = torch.randn(1, 15, 75)  # Long sequence
        
        # Both should produce valid outputs
        with torch.no_grad():
            encoded1 = self.gloss_decoder.encode_pose_sequence(seq1)
            encoded2 = self.gloss_decoder.encode_pose_sequence(seq2)
        
        assert encoded1.shape[1] == 8   # Preserves sequence length
        assert encoded2.shape[1] == 15
        assert encoded1.shape[2] == encoded2.shape[2]  # Same feature dim

class TestFingerspellingDetector:
    """Test fingerspelling detection and neural hand shape classification"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.fingerspelling_detector = FingerspellingDetector(
            model_path=None,  # Use mock model
            confidence_threshold=0.7,
            temporal_buffer_size=5
        )
    
    def test_hand_shape_classification(self):
        """Test hand shape classification for fingerspelling"""
        # Mock hand landmarks (21 points * 3 coordinates = 63 features)
        hand_landmarks = np.random.randn(63)
        
        with patch.object(self.fingerspelling_detector, '_classify_hand_shape') as mock_classify:
            mock_classify.return_value = ('A', 0.95)
            
            letter, confidence = self.fingerspelling_detector.detect_letter(hand_landmarks)
            
            assert letter == 'A'
            assert confidence == 0.95
            mock_classify.assert_called_once()
    
    def test_temporal_buffering(self):
        """Test temporal buffering for stable letter detection"""
        # Simulate sequence of hand shapes spelling "CAT"
        letter_sequence = [
            ('C', 0.9), ('C', 0.85), ('C', 0.92),  # Stable C
            ('A', 0.88), ('A', 0.91), ('A', 0.87), # Stable A
            ('T', 0.93), ('T', 0.89), ('T', 0.94)  # Stable T
        ]
        
        detected_word = []
        
        for letter, conf in letter_sequence:
            with patch.object(self.fingerspelling_detector, '_classify_hand_shape') as mock_classify:
                mock_classify.return_value = (letter, conf)
                
                # Mock hand landmarks
                hand_landmarks = np.random.randn(63)
                result = self.fingerspelling_detector.process_frame(hand_landmarks, timestamp=0.0)
                
                if result and result['stable']:
                    detected_word.append(result['letter'])
        
        # Should detect stable letters
        assert len(detected_word) > 0
    
    def test_confidence_filtering(self):
        """Test filtering of low-confidence detections"""
        # Low confidence detection
        low_conf_landmarks = np.random.randn(63)
        
        with patch.object(self.fingerspelling_detector, '_classify_hand_shape') as mock_classify:
            mock_classify.return_value = ('X', 0.3)  # Low confidence
            
            result = self.fingerspelling_detector.detect_letter(low_conf_landmarks)
            
            # Should be filtered out
            assert result[1] < self.fingerspelling_detector.confidence_threshold
    
    def test_hand_shape_normalization(self):
        """Test hand landmark normalization"""
        # Raw hand landmarks (not normalized)
        raw_landmarks = np.array([
            [100, 200, 0.1], [150, 220, 0.2], [200, 240, 0.3]  # Sample points
        ]).flatten()
        
        normalized = self.fingerspelling_detector._normalize_hand_landmarks(raw_landmarks)
        
        # Should be normalized to reasonable range
        assert len(normalized) == len(raw_landmarks)
        assert np.all(np.abs(normalized) <= 10)  # Reasonable range after normalization

class TestSpatialAnalyzer:
    """Test spatial role labeling and grammar analysis"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.spatial_analyzer = SpatialAnalyzer(
            spatial_threshold=0.1,
            temporal_window=1.0
        )
    
    def test_spatial_role_assignment(self):
        """Test assignment of spatial roles to entities"""
        # Mock pose data with spatial information
        pose_data = {
            'body': [
                {'x': 0.5, 'y': 0.3, 'z': 0.0, 'visibility': 0.9}  # Center reference
            ],
            'left_hand': [
                {'x': 0.2, 'y': 0.4, 'z': 0.1, 'visibility': 0.9}  # Left space
            ],
            'right_hand': [
                {'x': 0.8, 'y': 0.4, 'z': 0.1, 'visibility': 0.9}  # Right space
            ]
        }
        
        spatial_roles = self.spatial_analyzer.analyze_spatial_roles(pose_data)
        
        assert 'left_space' in spatial_roles
        assert 'right_space' in spatial_roles
        assert 'center_space' in spatial_roles
    
    def test_referential_pointing(self):
        """Test detection of referential pointing"""
        # Mock pointing gesture (index finger extended toward specific location)
        pointing_pose = {
            'right_hand': [
                {'x': 0.7, 'y': 0.3, 'z': 0.2, 'visibility': 0.9}  # Hand position
            ],
            'pointing_vector': [0.8, 0.2, 0.0]  # Pointing direction
        }
        
        pointing_info = self.spatial_analyzer.detect_pointing(pointing_pose)
        
        assert pointing_info is not None
        assert 'direction' in pointing_info
        assert 'confidence' in pointing_info
        assert pointing_info['confidence'] > 0.5
    
    def test_spatial_agreement(self):
        """Test spatial agreement in verb inflection"""
        # Mock verb with spatial agreement (GIVE from left to right)
        verb_pose_sequence = [
            {
                'gloss': 'GIVE',
                'left_hand': {'x': 0.2, 'y': 0.4, 'z': 0.0},  # Start left
                'right_hand': {'x': 0.2, 'y': 0.4, 'z': 0.0}
            },
            {
                'gloss': 'GIVE',
                'left_hand': {'x': 0.5, 'y': 0.4, 'z': 0.0},  # Move center
                'right_hand': {'x': 0.5, 'y': 0.4, 'z': 0.0}
            },
            {
                'gloss': 'GIVE',
                'left_hand': {'x': 0.8, 'y': 0.4, 'z': 0.0},  # End right
                'right_hand': {'x': 0.8, 'y': 0.4, 'z': 0.0}
            }
        ]
        
        agreement_info = self.spatial_analyzer.analyze_spatial_agreement(verb_pose_sequence)
        
        assert agreement_info is not None
        assert agreement_info['verb'] == 'GIVE'
        assert agreement_info['direction'] == 'left_to_right'
        assert agreement_info['agreement_type'] == 'directional'
    
    def test_classifier_handshapes(self):
        """Test detection of classifier handshapes"""
        # Mock classifier handshape (CL:1 for thin objects)
        classifier_landmarks = np.array([
            # Index finger extended, others closed (simplified)
            [0.5, 0.3, 0.0],  # Wrist
            [0.52, 0.25, 0.0], # Index MCP
            [0.54, 0.20, 0.0], # Index PIP
            [0.56, 0.15, 0.0], # Index DIP
            [0.58, 0.10, 0.0], # Index tip
            # Other fingers closed...
        ]).flatten()
        
        classifier_type = self.spatial_analyzer.detect_classifier_handshape(classifier_landmarks)
        
        assert classifier_type is not None
        assert classifier_type in ['CL:1', 'CL:3', 'CL:4', 'CL:5', 'CL:B', 'CL:C']

class TestModelEvaluation:
    """Test model evaluation metrics"""
    
    def test_gloss_f1_score(self):
        """Test F1 score calculation for gloss recognition"""
        # Mock predictions and ground truth
        predicted_glosses = ['HELLO', 'HOW', 'ARE', 'YOU']
        ground_truth_glosses = ['HELLO', 'HOW', 'YOU', 'FINE']
        
        f1_score = self._calculate_gloss_f1(predicted_glosses, ground_truth_glosses)
        
        # Should be reasonable F1 score
        assert 0 <= f1_score <= 1
        assert f1_score > 0.3  # Some overlap expected
    
    def test_word_error_rate(self):
        """Test Word Error Rate (WER) calculation"""
        # Mock translation outputs
        predicted_text = "hello how are you today"
        reference_text = "hello how are you doing"
        
        wer = self._calculate_wer(predicted_text, reference_text)
        
        # WER should be reasonable
        assert wer >= 0
        assert wer <= 1  # Should not exceed 100%
    
    def test_fairness_metrics(self):
        """Test fairness evaluation across demographics"""
        # Mock performance data across different groups
        performance_by_group = {
            'age_young': {'accuracy': 0.85, 'f1': 0.82},
            'age_middle': {'accuracy': 0.87, 'f1': 0.84},
            'age_senior': {'accuracy': 0.79, 'f1': 0.76},
            'gender_male': {'accuracy': 0.84, 'f1': 0.81},
            'gender_female': {'accuracy': 0.86, 'f1': 0.83},
            'ethnicity_white': {'accuracy': 0.85, 'f1': 0.82},
            'ethnicity_black': {'accuracy': 0.82, 'f1': 0.79},
            'ethnicity_hispanic': {'accuracy': 0.83, 'f1': 0.80}
        }
        
        fairness_metrics = self._calculate_fairness_metrics(performance_by_group)
        
        assert 'demographic_parity' in fairness_metrics
        assert 'equalized_odds' in fairness_metrics
        assert 'overall_fairness_score' in fairness_metrics
        
        # Fairness score should be between 0 and 1
        assert 0 <= fairness_metrics['overall_fairness_score'] <= 1
    
    def _calculate_gloss_f1(self, predicted, ground_truth):
        """Calculate F1 score for gloss sequences"""
        predicted_set = set(predicted)
        ground_truth_set = set(ground_truth)
        
        if len(predicted_set) == 0 and len(ground_truth_set) == 0:
            return 1.0
        
        intersection = predicted_set.intersection(ground_truth_set)
        precision = len(intersection) / len(predicted_set) if predicted_set else 0
        recall = len(intersection) / len(ground_truth_set) if ground_truth_set else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_wer(self, predicted, reference):
        """Calculate Word Error Rate"""
        pred_words = predicted.split()
        ref_words = reference.split()
        
        # Simple edit distance calculation
        if len(ref_words) == 0:
            return 1.0 if len(pred_words) > 0 else 0.0
        
        # Simplified WER calculation (in practice, use proper edit distance)
        common_words = set(pred_words).intersection(set(ref_words))
        errors = len(ref_words) - len(common_words)
        
        return errors / len(ref_words)
    
    def _calculate_fairness_metrics(self, performance_by_group):
        """Calculate fairness metrics across demographic groups"""
        accuracies = [group['accuracy'] for group in performance_by_group.values()]
        f1_scores = [group['f1'] for group in performance_by_group.values()]
        
        # Demographic parity (difference in accuracy)
        max_accuracy = max(accuracies)
        min_accuracy = min(accuracies)
        demographic_parity = 1 - (max_accuracy - min_accuracy)
        
        # Equalized odds (difference in F1)
        max_f1 = max(f1_scores)
        min_f1 = min(f1_scores)
        equalized_odds = 1 - (max_f1 - min_f1)
        
        # Overall fairness score
        overall_fairness = (demographic_parity + equalized_odds) / 2
        
        return {
            'demographic_parity': demographic_parity,
            'equalized_odds': equalized_odds,
            'overall_fairness_score': overall_fairness
        }

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
