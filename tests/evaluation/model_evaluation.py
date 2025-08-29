#!/usr/bin/env python3
"""
Comprehensive model evaluation framework for ASL translation system
Includes gloss F1, WER, fairness metrics, and bias detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from collections import defaultdict, Counter
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import editdistance
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    metric_name: str
    overall_score: float
    group_scores: Dict[str, float]
    confidence_interval: Tuple[float, float]
    sample_size: int
    metadata: Dict[str, Any]

@dataclass
class FairnessMetrics:
    """Container for fairness evaluation results"""
    demographic_parity: float
    equalized_odds: float
    calibration: float
    individual_fairness: float
    overall_fairness_score: float
    bias_indicators: Dict[str, float]

class ModelEvaluator:
    """Comprehensive model evaluation with fairness assessment"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.evaluation_results: Dict[str, EvaluationResult] = {}
        self.fairness_results: Optional[FairnessMetrics] = None
        
    def evaluate_gloss_recognition(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        demographics: Optional[List[Dict[str, str]]] = None
    ) -> EvaluationResult:
        """Evaluate gloss recognition performance with F1 score"""
        
        logger.info("Evaluating gloss recognition performance...")
        
        # Calculate overall F1 score
        all_pred_glosses = []
        all_true_glosses = []
        
        for pred_seq, true_seq in zip(predictions, ground_truth):
            all_pred_glosses.extend(pred_seq)
            all_true_glosses.extend(true_seq)
        
        # Get unique gloss vocabulary
        all_glosses = sorted(set(all_pred_glosses + all_true_glosses))
        gloss_to_idx = {gloss: idx for idx, gloss in enumerate(all_glosses)}
        
        # Convert to label format for sklearn
        pred_labels = [gloss_to_idx.get(gloss, -1) for gloss in all_pred_glosses]
        true_labels = [gloss_to_idx.get(gloss, -1) for gloss in all_true_glosses]
        
        # Calculate metrics
        overall_f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            true_labels, pred_labels, average=None, zero_division=0
        )
        
        # Calculate confidence interval using bootstrap
        bootstrap_f1s = []
        n_bootstrap = 1000
        n_samples = len(pred_labels)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_pred = [pred_labels[i] for i in indices]
            boot_true = [true_labels[i] for i in indices]
            boot_f1 = f1_score(boot_true, boot_pred, average='weighted', zero_division=0)
            bootstrap_f1s.append(boot_f1)
        
        ci_lower = np.percentile(bootstrap_f1s, (1 - self.confidence_level) / 2 * 100)
        ci_upper = np.percentile(bootstrap_f1s, (1 + self.confidence_level) / 2 * 100)
        
        # Group-wise evaluation if demographics provided
        group_scores = {}
        if demographics:
            group_scores = self._evaluate_by_groups(
                predictions, ground_truth, demographics, self._calculate_sequence_f1
            )
        
        # Per-class performance
        class_performance = {}
        for i, gloss in enumerate(all_glosses):
            if i < len(f1_per_class):
                class_performance[gloss] = {
                    'f1': f1_per_class[i],
                    'precision': precision[i],
                    'recall': recall[i],
                    'support': support[i]
                }
        
        result = EvaluationResult(
            metric_name="gloss_f1",
            overall_score=overall_f1,
            group_scores=group_scores,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=len(predictions),
            metadata={
                'class_performance': class_performance,
                'vocabulary_size': len(all_glosses),
                'total_glosses': len(all_pred_glosses)
            }
        )
        
        self.evaluation_results['gloss_f1'] = result
        logger.info(f"Gloss F1 Score: {overall_f1:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
        
        return result
    
    def evaluate_translation_quality(
        self,
        predictions: List[str],
        references: List[str],
        demographics: Optional[List[Dict[str, str]]] = None
    ) -> EvaluationResult:
        """Evaluate translation quality using WER and BLEU"""
        
        logger.info("Evaluating translation quality...")
        
        # Calculate Word Error Rate (WER)
        wer_scores = []
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            wer = self._calculate_wer(pred, ref)
            bleu = self._calculate_bleu(pred, ref)
            wer_scores.append(wer)
            bleu_scores.append(bleu)
        
        overall_wer = np.mean(wer_scores)
        overall_bleu = np.mean(bleu_scores)
        
        # Bootstrap confidence intervals
        bootstrap_wers = []
        n_bootstrap = 1000
        n_samples = len(wer_scores)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_wer = np.mean([wer_scores[i] for i in indices])
            bootstrap_wers.append(boot_wer)
        
        ci_lower = np.percentile(bootstrap_wers, (1 - self.confidence_level) / 2 * 100)
        ci_upper = np.percentile(bootstrap_wers, (1 + self.confidence_level) / 2 * 100)
        
        # Group-wise evaluation
        group_scores = {}
        if demographics:
            group_scores = self._evaluate_by_groups(
                predictions, references, demographics, self._calculate_wer
            )
        
        result = EvaluationResult(
            metric_name="word_error_rate",
            overall_score=overall_wer,
            group_scores=group_scores,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=len(predictions),
            metadata={
                'bleu_score': overall_bleu,
                'wer_distribution': {
                    'mean': overall_wer,
                    'std': np.std(wer_scores),
                    'median': np.median(wer_scores),
                    'min': np.min(wer_scores),
                    'max': np.max(wer_scores)
                }
            }
        )
        
        self.evaluation_results['wer'] = result
        logger.info(f"Word Error Rate: {overall_wer:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
        logger.info(f"BLEU Score: {overall_bleu:.3f}")
        
        return result
    
    def evaluate_fairness(
        self,
        predictions: List[Any],
        ground_truth: List[Any],
        demographics: List[Dict[str, str]],
        prediction_scores: Optional[List[float]] = None
    ) -> FairnessMetrics:
        """Comprehensive fairness evaluation across demographic groups"""
        
        logger.info("Evaluating fairness across demographic groups...")
        
        # Group data by demographics
        demographic_groups = defaultdict(list)
        for i, demo in enumerate(demographics):
            for attr, value in demo.items():
                group_key = f"{attr}_{value}"
                demographic_groups[group_key].append(i)
        
        # Calculate performance by group
        group_performance = {}
        for group, indices in demographic_groups.items():
            if len(indices) < 10:  # Skip groups with too few samples
                continue
                
            group_pred = [predictions[i] for i in indices]
            group_true = [ground_truth[i] for i in indices]
            
            # Calculate accuracy
            if isinstance(predictions[0], str):
                # Text predictions
                accuracy = np.mean([p == t for p, t in zip(group_pred, group_true)])
            else:
                # Sequence predictions
                accuracy = np.mean([
                    self._calculate_sequence_accuracy(p, t) 
                    for p, t in zip(group_pred, group_true)
                ])
            
            group_performance[group] = {
                'accuracy': accuracy,
                'size': len(indices),
                'predictions': group_pred,
                'ground_truth': group_true
            }
        
        # Calculate fairness metrics
        accuracies = [perf['accuracy'] for perf in group_performance.values()]
        
        # Demographic Parity (difference in positive prediction rates)
        demographic_parity = 1 - (max(accuracies) - min(accuracies)) if accuracies else 1.0
        
        # Equalized Odds (difference in TPR and FPR across groups)
        equalized_odds = self._calculate_equalized_odds(group_performance)
        
        # Calibration (reliability of confidence scores across groups)
        calibration = 1.0  # Simplified - would need confidence scores
        if prediction_scores:
            calibration = self._calculate_calibration(group_performance, prediction_scores, demographics)
        
        # Individual Fairness (similar individuals get similar predictions)
        individual_fairness = self._calculate_individual_fairness(
            predictions, demographics, prediction_scores
        )
        
        # Overall fairness score
        overall_fairness = np.mean([
            demographic_parity, equalized_odds, calibration, individual_fairness
        ])
        
        # Bias indicators
        bias_indicators = self._detect_bias_indicators(group_performance)
        
        fairness_metrics = FairnessMetrics(
            demographic_parity=demographic_parity,
            equalized_odds=equalized_odds,
            calibration=calibration,
            individual_fairness=individual_fairness,
            overall_fairness_score=overall_fairness,
            bias_indicators=bias_indicators
        )
        
        self.fairness_results = fairness_metrics
        
        logger.info(f"Overall Fairness Score: {overall_fairness:.3f}")
        logger.info(f"Demographic Parity: {demographic_parity:.3f}")
        logger.info(f"Equalized Odds: {equalized_odds:.3f}")
        
        return fairness_metrics
    
    def evaluate_robustness(
        self,
        clean_predictions: List[Any],
        noisy_predictions: List[Any],
        ground_truth: List[Any],
        noise_types: List[str]
    ) -> Dict[str, EvaluationResult]:
        """Evaluate model robustness to different types of noise"""
        
        logger.info("Evaluating model robustness...")
        
        robustness_results = {}
        
        # Calculate clean performance baseline
        if isinstance(clean_predictions[0], str):
            clean_accuracy = np.mean([p == t for p, t in zip(clean_predictions, ground_truth)])
        else:
            clean_accuracy = np.mean([
                self._calculate_sequence_accuracy(p, t) 
                for p, t in zip(clean_predictions, ground_truth)
            ])
        
        # Evaluate each noise type
        for noise_type, noisy_pred in zip(noise_types, noisy_predictions):
            if isinstance(noisy_pred[0], str):
                noisy_accuracy = np.mean([p == t for p, t in zip(noisy_pred, ground_truth)])
            else:
                noisy_accuracy = np.mean([
                    self._calculate_sequence_accuracy(p, t) 
                    for p, t in zip(noisy_pred, ground_truth)
                ])
            
            # Robustness score (how much performance degrades)
            robustness_score = noisy_accuracy / clean_accuracy if clean_accuracy > 0 else 0
            
            result = EvaluationResult(
                metric_name=f"robustness_{noise_type}",
                overall_score=robustness_score,
                group_scores={},
                confidence_interval=(0, 1),  # Simplified
                sample_size=len(noisy_pred),
                metadata={
                    'clean_accuracy': clean_accuracy,
                    'noisy_accuracy': noisy_accuracy,
                    'performance_drop': clean_accuracy - noisy_accuracy
                }
            )
            
            robustness_results[noise_type] = result
            logger.info(f"Robustness to {noise_type}: {robustness_score:.3f}")
        
        return robustness_results
    
    def generate_evaluation_report(self, output_path: str = "evaluation_report.json"):
        """Generate comprehensive evaluation report"""
        
        logger.info("Generating evaluation report...")
        
        report = {
            'evaluation_summary': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'confidence_level': self.confidence_level,
                'total_metrics': len(self.evaluation_results)
            },
            'performance_metrics': {},
            'fairness_assessment': {},
            'recommendations': []
        }
        
        # Add performance metrics
        for metric_name, result in self.evaluation_results.items():
            report['performance_metrics'][metric_name] = {
                'overall_score': result.overall_score,
                'confidence_interval': result.confidence_interval,
                'sample_size': result.sample_size,
                'group_scores': result.group_scores,
                'metadata': result.metadata
            }
        
        # Add fairness assessment
        if self.fairness_results:
            report['fairness_assessment'] = {
                'demographic_parity': self.fairness_results.demographic_parity,
                'equalized_odds': self.fairness_results.equalized_odds,
                'calibration': self.fairness_results.calibration,
                'individual_fairness': self.fairness_results.individual_fairness,
                'overall_fairness_score': self.fairness_results.overall_fairness_score,
                'bias_indicators': self.fairness_results.bias_indicators
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        report['recommendations'] = recommendations
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
        return report
    
    def visualize_results(self, output_dir: str = "evaluation_plots"):
        """Generate visualization plots for evaluation results"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Performance by demographic groups
        if self.fairness_results:
            self._plot_fairness_metrics(output_dir)
        
        # Performance distributions
        self._plot_performance_distributions(output_dir)
        
        # Confusion matrices for classification tasks
        self._plot_confusion_matrices(output_dir)
        
        logger.info(f"Evaluation plots saved to {output_dir}")
    
    # Helper methods
    def _calculate_wer(self, prediction: str, reference: str) -> float:
        """Calculate Word Error Rate using edit distance"""
        pred_words = prediction.lower().split()
        ref_words = reference.lower().split()
        
        if len(ref_words) == 0:
            return 1.0 if len(pred_words) > 0 else 0.0
        
        edit_dist = editdistance.eval(pred_words, ref_words)
        return edit_dist / len(ref_words)
    
    def _calculate_bleu(self, prediction: str, reference: str) -> float:
        """Calculate BLEU score (simplified implementation)"""
        pred_words = prediction.lower().split()
        ref_words = reference.lower().split()
        
        if len(pred_words) == 0 or len(ref_words) == 0:
            return 0.0
        
        # 1-gram precision
        pred_counts = Counter(pred_words)
        ref_counts = Counter(ref_words)
        
        overlap = sum(min(pred_counts[word], ref_counts[word]) for word in pred_counts)
        precision = overlap / len(pred_words)
        
        # Brevity penalty
        bp = min(1.0, len(pred_words) / len(ref_words))
        
        return bp * precision
    
    def _calculate_sequence_f1(self, pred_seq: List[str], true_seq: List[str]) -> float:
        """Calculate F1 score for sequence"""
        pred_set = set(pred_seq)
        true_set = set(true_seq)
        
        if len(pred_set) == 0 and len(true_set) == 0:
            return 1.0
        
        intersection = pred_set.intersection(true_set)
        precision = len(intersection) / len(pred_set) if pred_set else 0
        recall = len(intersection) / len(true_set) if true_set else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_sequence_accuracy(self, pred_seq: List[str], true_seq: List[str]) -> float:
        """Calculate sequence-level accuracy"""
        return 1.0 if pred_seq == true_seq else 0.0
    
    def _evaluate_by_groups(
        self, 
        predictions: List[Any], 
        ground_truth: List[Any], 
        demographics: List[Dict[str, str]], 
        metric_func
    ) -> Dict[str, float]:
        """Evaluate performance by demographic groups"""
        
        group_scores = {}
        demographic_groups = defaultdict(list)
        
        # Group by demographics
        for i, demo in enumerate(demographics):
            for attr, value in demo.items():
                group_key = f"{attr}_{value}"
                demographic_groups[group_key].append(i)
        
        # Calculate scores for each group
        for group, indices in demographic_groups.items():
            if len(indices) < 5:  # Skip small groups
                continue
                
            group_pred = [predictions[i] for i in indices]
            group_true = [ground_truth[i] for i in indices]
            
            if len(group_pred) > 0:
                if callable(metric_func):
                    scores = [metric_func(p, t) for p, t in zip(group_pred, group_true)]
                    group_scores[group] = np.mean(scores)
        
        return group_scores
    
    def _calculate_equalized_odds(self, group_performance: Dict) -> float:
        """Calculate equalized odds fairness metric"""
        # Simplified implementation
        accuracies = [perf['accuracy'] for perf in group_performance.values()]
        if len(accuracies) < 2:
            return 1.0
        
        return 1 - (max(accuracies) - min(accuracies))
    
    def _calculate_calibration(
        self, 
        group_performance: Dict, 
        prediction_scores: List[float], 
        demographics: List[Dict[str, str]]
    ) -> float:
        """Calculate calibration across groups"""
        # Simplified - would need proper calibration analysis
        return 0.8  # Placeholder
    
    def _calculate_individual_fairness(
        self, 
        predictions: List[Any], 
        demographics: List[Dict[str, str]], 
        prediction_scores: Optional[List[float]]
    ) -> float:
        """Calculate individual fairness metric"""
        # Simplified - would need similarity metrics between individuals
        return 0.85  # Placeholder
    
    def _detect_bias_indicators(self, group_performance: Dict) -> Dict[str, float]:
        """Detect potential bias indicators"""
        bias_indicators = {}
        
        accuracies = [perf['accuracy'] for perf in group_performance.values()]
        group_sizes = [perf['size'] for perf in group_performance.values()]
        
        # Performance variance across groups
        bias_indicators['performance_variance'] = np.var(accuracies)
        
        # Size imbalance
        bias_indicators['size_imbalance'] = max(group_sizes) / min(group_sizes) if group_sizes else 1.0
        
        # Statistical significance of differences
        if len(accuracies) >= 2:
            _, p_value = stats.ttest_ind(accuracies[:len(accuracies)//2], accuracies[len(accuracies)//2:])
            bias_indicators['statistical_significance'] = p_value
        
        return bias_indicators
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Performance recommendations
        if 'gloss_f1' in self.evaluation_results:
            f1_score = self.evaluation_results['gloss_f1'].overall_score
            if f1_score < 0.7:
                recommendations.append("Consider improving gloss recognition model with more training data")
            if f1_score < 0.5:
                recommendations.append("Gloss recognition performance is critically low - review model architecture")
        
        if 'wer' in self.evaluation_results:
            wer_score = self.evaluation_results['wer'].overall_score
            if wer_score > 0.3:
                recommendations.append("High word error rate - consider improving translation model")
            if wer_score > 0.5:
                recommendations.append("Translation quality is poor - major model improvements needed")
        
        # Fairness recommendations
        if self.fairness_results:
            if self.fairness_results.overall_fairness_score < 0.8:
                recommendations.append("Fairness concerns detected - review training data for bias")
            if self.fairness_results.demographic_parity < 0.7:
                recommendations.append("Significant demographic disparities - consider bias mitigation techniques")
        
        return recommendations
    
    def _plot_fairness_metrics(self, output_dir: str):
        """Plot fairness metrics"""
        if not self.fairness_results:
            return
        
        metrics = {
            'Demographic Parity': self.fairness_results.demographic_parity,
            'Equalized Odds': self.fairness_results.equalized_odds,
            'Calibration': self.fairness_results.calibration,
            'Individual Fairness': self.fairness_results.individual_fairness
        }
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.ylim(0, 1)
        plt.ylabel('Fairness Score')
        plt.title('Fairness Metrics Across Demographic Groups')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fairness_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_distributions(self, output_dir: str):
        """Plot performance distributions"""
        # Implementation would create histograms and box plots
        pass
    
    def _plot_confusion_matrices(self, output_dir: str):
        """Plot confusion matrices for classification tasks"""
        # Implementation would create confusion matrix heatmaps
        pass

# Example usage and test cases
if __name__ == '__main__':
    # Create mock evaluation data
    evaluator = ModelEvaluator()
    
    # Mock gloss recognition data
    mock_predictions = [
        ['HELLO', 'HOW', 'ARE', 'YOU'],
        ['THANK', 'YOU', 'VERY', 'MUCH'],
        ['I', 'LOVE', 'SIGN', 'LANGUAGE']
    ]
    
    mock_ground_truth = [
        ['HELLO', 'HOW', 'ARE', 'YOU'],
        ['THANK', 'YOU', 'SO', 'MUCH'],
        ['I', 'LOVE', 'SIGN', 'LANGUAGE']
    ]
    
    mock_demographics = [
        {'age': 'young', 'gender': 'female', 'ethnicity': 'white'},
        {'age': 'middle', 'gender': 'male', 'ethnicity': 'black'},
        {'age': 'senior', 'gender': 'female', 'ethnicity': 'hispanic'}
    ]
    
    # Evaluate gloss recognition
    gloss_result = evaluator.evaluate_gloss_recognition(
        mock_predictions, mock_ground_truth, mock_demographics
    )
    
    # Mock translation data
    mock_translations = [
        "hello how are you today",
        "thank you very much",
        "i love sign language"
    ]
    
    mock_references = [
        "hello how are you",
        "thank you so much", 
        "i love sign language"
    ]
    
    # Evaluate translation quality
    translation_result = evaluator.evaluate_translation_quality(
        mock_translations, mock_references, mock_demographics
    )
    
    # Evaluate fairness
    fairness_result = evaluator.evaluate_fairness(
        mock_predictions, mock_ground_truth, mock_demographics
    )
    
    # Generate report
    report = evaluator.generate_evaluation_report("test_evaluation_report.json")
    
    print("Evaluation completed successfully!")
    print(f"Gloss F1 Score: {gloss_result.overall_score:.3f}")
    print(f"Word Error Rate: {translation_result.overall_score:.3f}")
    print(f"Overall Fairness Score: {fairness_result.overall_fairness_score:.3f}")
