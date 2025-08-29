#!/usr/bin/env python3
"""
Human-in-the-loop review system with Deaf signers
Facilitates expert review and feedback collection for ASL translation quality
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DISPUTED = "disputed"
    APPROVED = "approved"
    REJECTED = "rejected"

class ReviewType(Enum):
    TRANSLATION_QUALITY = "translation_quality"
    GLOSS_ACCURACY = "gloss_accuracy"
    CULTURAL_APPROPRIATENESS = "cultural_appropriateness"
    AVATAR_NATURALNESS = "avatar_naturalness"
    OVERALL_SYSTEM = "overall_system"

@dataclass
class ReviewItem:
    """Individual item for review"""
    id: str
    type: ReviewType
    content: Dict[str, Any]  # Original content (video, text, etc.)
    system_output: Dict[str, Any]  # System's translation/output
    metadata: Dict[str, Any]  # Additional context
    created_at: datetime
    priority: int  # 1-5, 5 being highest priority
    
@dataclass
class ReviewFeedback:
    """Feedback from a Deaf signer reviewer"""
    id: str
    review_item_id: str
    reviewer_id: str
    reviewer_profile: Dict[str, Any]  # Background, experience, etc.
    
    # Quality ratings (1-5 scale)
    accuracy_rating: int
    naturalness_rating: int
    cultural_rating: int
    overall_rating: int
    
    # Detailed feedback
    comments: str
    suggested_corrections: List[Dict[str, Any]]
    flagged_issues: List[str]
    
    # Confidence and consensus
    confidence_level: int  # 1-5
    agrees_with_system: bool
    
    # Metadata
    review_time_minutes: float
    submitted_at: datetime
    
@dataclass
class ReviewerProfile:
    """Profile of a Deaf signer reviewer"""
    id: str
    name: str
    email: str
    
    # ASL Background
    asl_fluency: str  # "native", "near_native", "fluent", "intermediate"
    years_signing: int
    deaf_since: str  # "birth", "early_childhood", "later_childhood", "adult"
    
    # Education & Experience
    education_level: str
    asl_teaching_experience: int
    translation_experience: int
    
    # Specializations
    specialization_domains: List[str]  # medical, legal, educational, etc.
    regional_variants: List[str]  # ASL regional variations familiar with
    
    # Review Statistics
    reviews_completed: int
    average_review_time: float
    reliability_score: float  # Based on consistency with other reviewers
    
    # Availability
    available_hours: Dict[str, List[str]]  # day_of_week -> [hour_ranges]
    timezone: str
    max_reviews_per_day: int

class DeafSignerReviewSystem:
    """System for managing human-in-the-loop reviews by Deaf signers"""
    
    def __init__(self, db_path: str = "review_system.db"):
        self.db_path = db_path
        self.setup_database()
        self.reviewers: Dict[str, ReviewerProfile] = {}
        self.pending_reviews: Dict[str, ReviewItem] = {}
        self.completed_reviews: Dict[str, List[ReviewFeedback]] = {}
        
    def setup_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviewers (
                id TEXT PRIMARY KEY,
                profile_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS review_items (
                id TEXT PRIMARY KEY,
                type TEXT,
                content TEXT,
                system_output TEXT,
                metadata TEXT,
                status TEXT,
                priority INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS review_feedback (
                id TEXT PRIMARY KEY,
                review_item_id TEXT,
                reviewer_id TEXT,
                feedback_data TEXT,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (review_item_id) REFERENCES review_items (id),
                FOREIGN KEY (reviewer_id) REFERENCES reviewers (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def register_reviewer(self, profile: ReviewerProfile) -> str:
        """Register a new Deaf signer reviewer"""
        logger.info(f"Registering new reviewer: {profile.name}")
        
        # Store in memory
        self.reviewers[profile.id] = profile
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO reviewers (id, profile_data) VALUES (?, ?)",
            (profile.id, json.dumps(asdict(profile), default=str))
        )
        conn.commit()
        conn.close()
        
        return profile.id
    
    def submit_for_review(
        self, 
        review_type: ReviewType,
        content: Dict[str, Any],
        system_output: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = 3
    ) -> str:
        """Submit content for human review"""
        
        review_item = ReviewItem(
            id=str(uuid.uuid4()),
            type=review_type,
            content=content,
            system_output=system_output,
            metadata=metadata or {},
            created_at=datetime.now(),
            priority=priority
        )
        
        # Store in memory
        self.pending_reviews[review_item.id] = review_item
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO review_items 
               (id, type, content, system_output, metadata, status, priority) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                review_item.id,
                review_type.value,
                json.dumps(content),
                json.dumps(system_output),
                json.dumps(metadata or {}),
                ReviewStatus.PENDING.value,
                priority
            )
        )
        conn.commit()
        conn.close()
        
        logger.info(f"Submitted review item {review_item.id} for {review_type.value}")
        
        # Assign to reviewers
        asyncio.create_task(self._assign_reviewers(review_item))
        
        return review_item.id
    
    async def _assign_reviewers(self, review_item: ReviewItem, num_reviewers: int = 3):
        """Assign review item to appropriate reviewers"""
        
        # Find suitable reviewers based on specialization and availability
        suitable_reviewers = []
        
        for reviewer in self.reviewers.values():
            # Check specialization match
            if review_item.type == ReviewType.TRANSLATION_QUALITY:
                if reviewer.translation_experience > 0:
                    suitable_reviewers.append(reviewer)
            elif review_item.type == ReviewType.CULTURAL_APPROPRIATENESS:
                if reviewer.asl_fluency in ["native", "near_native"]:
                    suitable_reviewers.append(reviewer)
            else:
                suitable_reviewers.append(reviewer)
        
        # Sort by reliability and availability
        suitable_reviewers.sort(key=lambda r: (-r.reliability_score, r.reviews_completed))
        
        # Assign to top reviewers
        assigned_reviewers = suitable_reviewers[:num_reviewers]
        
        for reviewer in assigned_reviewers:
            await self._notify_reviewer(reviewer, review_item)
        
        logger.info(f"Assigned review item {review_item.id} to {len(assigned_reviewers)} reviewers")
    
    async def _notify_reviewer(self, reviewer: ReviewerProfile, review_item: ReviewItem):
        """Notify reviewer of new assignment (mock implementation)"""
        # In production, this would send email/SMS/app notification
        logger.info(f"Notified {reviewer.name} about review item {review_item.id}")
    
    def submit_feedback(self, feedback: ReviewFeedback) -> str:
        """Submit feedback from a reviewer"""
        
        # Validate feedback
        if not self._validate_feedback(feedback):
            raise ValueError("Invalid feedback data")
        
        # Store feedback
        if feedback.review_item_id not in self.completed_reviews:
            self.completed_reviews[feedback.review_item_id] = []
        
        self.completed_reviews[feedback.review_item_id].append(feedback)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO review_feedback (id, review_item_id, reviewer_id, feedback_data) VALUES (?, ?, ?, ?)",
            (feedback.id, feedback.review_item_id, feedback.reviewer_id, json.dumps(asdict(feedback), default=str))
        )
        conn.commit()
        conn.close()
        
        # Update reviewer statistics
        if feedback.reviewer_id in self.reviewers:
            reviewer = self.reviewers[feedback.reviewer_id]
            reviewer.reviews_completed += 1
            reviewer.average_review_time = (
                (reviewer.average_review_time * (reviewer.reviews_completed - 1) + feedback.review_time_minutes) 
                / reviewer.reviews_completed
            )
        
        logger.info(f"Received feedback {feedback.id} from {feedback.reviewer_id}")
        
        # Check if review is complete
        self._check_review_completion(feedback.review_item_id)
        
        return feedback.id
    
    def _validate_feedback(self, feedback: ReviewFeedback) -> bool:
        """Validate feedback data"""
        # Check rating ranges
        ratings = [feedback.accuracy_rating, feedback.naturalness_rating, 
                  feedback.cultural_rating, feedback.overall_rating, feedback.confidence_level]
        
        if not all(1 <= rating <= 5 for rating in ratings):
            return False
        
        # Check required fields
        if not feedback.comments or len(feedback.comments.strip()) < 10:
            return False
        
        return True
    
    def _check_review_completion(self, review_item_id: str):
        """Check if review item has enough feedback to be considered complete"""
        
        if review_item_id not in self.completed_reviews:
            return
        
        feedback_list = self.completed_reviews[review_item_id]
        
        # Need at least 2 reviews for completion
        if len(feedback_list) >= 2:
            # Calculate consensus
            consensus_score = self._calculate_consensus(feedback_list)
            
            if consensus_score > 0.7:  # High consensus
                self._finalize_review(review_item_id, ReviewStatus.COMPLETED)
            elif len(feedback_list) >= 3:  # More reviews needed for low consensus
                if consensus_score > 0.5:
                    self._finalize_review(review_item_id, ReviewStatus.COMPLETED)
                else:
                    self._finalize_review(review_item_id, ReviewStatus.DISPUTED)
    
    def _calculate_consensus(self, feedback_list: List[ReviewFeedback]) -> float:
        """Calculate consensus score among reviewers"""
        if len(feedback_list) < 2:
            return 0.0
        
        # Calculate agreement on overall ratings
        overall_ratings = [f.overall_rating for f in feedback_list]
        rating_variance = sum((r - sum(overall_ratings)/len(overall_ratings))**2 for r in overall_ratings) / len(overall_ratings)
        
        # Calculate agreement on system accuracy
        system_agreements = [f.agrees_with_system for f in feedback_list]
        agreement_ratio = sum(system_agreements) / len(system_agreements)
        
        # Combine metrics
        consensus_score = (1 - rating_variance/4) * 0.6 + abs(agreement_ratio - 0.5) * 2 * 0.4
        
        return max(0, min(1, consensus_score))
    
    def _finalize_review(self, review_item_id: str, status: ReviewStatus):
        """Finalize review with given status"""
        
        # Update database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE review_items SET status = ? WHERE id = ?",
            (status.value, review_item_id)
        )
        conn.commit()
        conn.close()
        
        # Remove from pending
        if review_item_id in self.pending_reviews:
            del self.pending_reviews[review_item_id]
        
        logger.info(f"Finalized review {review_item_id} with status {status.value}")
        
        # Generate improvement recommendations
        if status == ReviewStatus.COMPLETED:
            self._generate_improvement_recommendations(review_item_id)
    
    def _generate_improvement_recommendations(self, review_item_id: str):
        """Generate recommendations based on reviewer feedback"""
        
        if review_item_id not in self.completed_reviews:
            return
        
        feedback_list = self.completed_reviews[review_item_id]
        
        # Analyze common issues
        all_issues = []
        all_corrections = []
        
        for feedback in feedback_list:
            all_issues.extend(feedback.flagged_issues)
            all_corrections.extend(feedback.suggested_corrections)
        
        # Count issue frequency
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Generate recommendations
        recommendations = {
            'review_item_id': review_item_id,
            'common_issues': sorted(issue_counts.items(), key=lambda x: x[1], reverse=True),
            'suggested_improvements': all_corrections,
            'average_ratings': {
                'accuracy': sum(f.accuracy_rating for f in feedback_list) / len(feedback_list),
                'naturalness': sum(f.naturalness_rating for f in feedback_list) / len(feedback_list),
                'cultural': sum(f.cultural_rating for f in feedback_list) / len(feedback_list),
                'overall': sum(f.overall_rating for f in feedback_list) / len(feedback_list)
            },
            'consensus_score': self._calculate_consensus(feedback_list)
        }
        
        logger.info(f"Generated improvement recommendations for {review_item_id}")
        return recommendations
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """Get comprehensive review statistics"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get review counts by status
        cursor.execute("SELECT status, COUNT(*) FROM review_items GROUP BY status")
        status_counts = dict(cursor.fetchall())
        
        # Get average ratings
        cursor.execute("""
            SELECT AVG(json_extract(feedback_data, '$.overall_rating')) as avg_rating,
                   AVG(json_extract(feedback_data, '$.review_time_minutes')) as avg_time
            FROM review_feedback
        """)
        avg_stats = cursor.fetchone()
        
        conn.close()
        
        # Calculate reviewer statistics
        reviewer_stats = {
            'total_reviewers': len(self.reviewers),
            'active_reviewers': len([r for r in self.reviewers.values() if r.reviews_completed > 0]),
            'average_experience': sum(r.years_signing for r in self.reviewers.values()) / len(self.reviewers) if self.reviewers else 0
        }
        
        return {
            'review_counts': status_counts,
            'average_rating': avg_stats[0] if avg_stats[0] else 0,
            'average_review_time': avg_stats[1] if avg_stats[1] else 0,
            'reviewer_statistics': reviewer_stats,
            'total_feedback_items': sum(len(feedback_list) for feedback_list in self.completed_reviews.values())
        }
    
    def get_pending_reviews_for_reviewer(self, reviewer_id: str) -> List[ReviewItem]:
        """Get pending reviews assigned to a specific reviewer"""
        
        # In a full implementation, this would track assignments
        # For now, return high-priority items matching reviewer's specialization
        
        if reviewer_id not in self.reviewers:
            return []
        
        reviewer = self.reviewers[reviewer_id]
        suitable_reviews = []
        
        for review_item in self.pending_reviews.values():
            # Check if suitable for this reviewer
            if self._is_suitable_for_reviewer(review_item, reviewer):
                suitable_reviews.append(review_item)
        
        # Sort by priority and creation time
        suitable_reviews.sort(key=lambda r: (-r.priority, r.created_at))
        
        return suitable_reviews[:5]  # Return top 5
    
    def _is_suitable_for_reviewer(self, review_item: ReviewItem, reviewer: ReviewerProfile) -> bool:
        """Check if review item is suitable for a specific reviewer"""
        
        # Check specialization
        if review_item.type == ReviewType.TRANSLATION_QUALITY:
            return reviewer.translation_experience > 0
        elif review_item.type == ReviewType.CULTURAL_APPROPRIATENESS:
            return reviewer.asl_fluency in ["native", "near_native"]
        elif review_item.type == ReviewType.AVATAR_NATURALNESS:
            return reviewer.years_signing >= 10
        
        return True
    
    def export_review_data(self, output_path: str):
        """Export review data for analysis"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'reviewers': [asdict(reviewer) for reviewer in self.reviewers.values()],
            'completed_reviews': {},
            'statistics': self.get_review_statistics()
        }
        
        # Add completed reviews (anonymized)
        for review_id, feedback_list in self.completed_reviews.items():
            export_data['completed_reviews'][review_id] = [
                {
                    'accuracy_rating': f.accuracy_rating,
                    'naturalness_rating': f.naturalness_rating,
                    'cultural_rating': f.cultural_rating,
                    'overall_rating': f.overall_rating,
                    'confidence_level': f.confidence_level,
                    'agrees_with_system': f.agrees_with_system,
                    'flagged_issues': f.flagged_issues,
                    'review_time_minutes': f.review_time_minutes,
                    'reviewer_experience': self.reviewers[f.reviewer_id].years_signing if f.reviewer_id in self.reviewers else 0
                }
                for f in feedback_list
            ]
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported review data to {output_path}")

# Example usage and test cases
if __name__ == '__main__':
    # Initialize review system
    review_system = DeafSignerReviewSystem()
    
    # Register sample reviewers
    reviewer1 = ReviewerProfile(
        id="reviewer_001",
        name="Sarah Johnson",
        email="sarah@example.com",
        asl_fluency="native",
        years_signing=25,
        deaf_since="birth",
        education_level="masters",
        asl_teaching_experience=10,
        translation_experience=5,
        specialization_domains=["medical", "educational"],
        regional_variants=["ASL-Northeast"],
        reviews_completed=0,
        average_review_time=0.0,
        reliability_score=1.0,
        available_hours={"monday": ["9-17"], "tuesday": ["9-17"]},
        timezone="EST",
        max_reviews_per_day=5
    )
    
    reviewer2 = ReviewerProfile(
        id="reviewer_002",
        name="Michael Chen",
        email="michael@example.com",
        asl_fluency="near_native",
        years_signing=15,
        deaf_since="early_childhood",
        education_level="bachelors",
        asl_teaching_experience=3,
        translation_experience=8,
        specialization_domains=["legal", "business"],
        regional_variants=["ASL-West"],
        reviews_completed=0,
        average_review_time=0.0,
        reliability_score=1.0,
        available_hours={"wednesday": ["10-18"], "thursday": ["10-18"]},
        timezone="PST",
        max_reviews_per_day=3
    )
    
    # Register reviewers
    review_system.register_reviewer(reviewer1)
    review_system.register_reviewer(reviewer2)
    
    # Submit sample content for review
    sample_content = {
        "video_url": "https://example.com/sign_video.mp4",
        "duration": 5.2,
        "context": "Medical consultation"
    }
    
    sample_system_output = {
        "gloss_sequence": ["DOCTOR", "SAY", "YOU", "NEED", "MEDICINE"],
        "translation": "The doctor says you need medicine",
        "confidence": 0.85
    }
    
    review_id = review_system.submit_for_review(
        ReviewType.TRANSLATION_QUALITY,
        sample_content,
        sample_system_output,
        {"domain": "medical", "complexity": "medium"},
        priority=4
    )
    
    # Submit sample feedback
    feedback1 = ReviewFeedback(
        id=str(uuid.uuid4()),
        review_item_id=review_id,
        reviewer_id="reviewer_001",
        reviewer_profile={"name": "Sarah Johnson", "experience": 25},
        accuracy_rating=4,
        naturalness_rating=3,
        cultural_rating=4,
        overall_rating=4,
        comments="Good translation overall, but the sign for MEDICINE could be more natural. The gloss sequence is accurate.",
        suggested_corrections=[
            {"type": "gloss", "original": "MEDICINE", "suggested": "MEDICINE++", "reason": "Emphasis needed"}
        ],
        flagged_issues=["sign_naturalness"],
        confidence_level=4,
        agrees_with_system=True,
        review_time_minutes=12.5,
        submitted_at=datetime.now()
    )
    
    feedback2 = ReviewFeedback(
        id=str(uuid.uuid4()),
        review_item_id=review_id,
        reviewer_id="reviewer_002",
        reviewer_profile={"name": "Michael Chen", "experience": 15},
        accuracy_rating=4,
        naturalness_rating=4,
        cultural_rating=4,
        overall_rating=4,
        comments="Accurate translation. The medical context is handled well.",
        suggested_corrections=[],
        flagged_issues=[],
        confidence_level=4,
        agrees_with_system=True,
        review_time_minutes=8.3,
        submitted_at=datetime.now()
    )
    
    # Submit feedback
    review_system.submit_feedback(feedback1)
    review_system.submit_feedback(feedback2)
    
    # Get statistics
    stats = review_system.get_review_statistics()
    print("Review System Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Export data
    review_system.export_review_data("review_export.json")
    
    print("Human-in-the-loop review system demonstration completed!")
