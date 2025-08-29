import json
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class TermMatch:
    """Terminology match result"""
    original_term: str
    enhanced_term: str
    confidence: float
    domain: str
    definition: Optional[str] = None
    context: Optional[str] = None

class TerminologyEnhancer:
    """Enhances translations with domain-specific terminology"""
    
    def __init__(self):
        self.termbanks: Dict[str, Dict[str, Any]] = {}
        self.general_terms = self._load_general_terms()
        
        logger.info("TerminologyEnhancer initialized")
    
    def _load_general_terms(self) -> Dict[str, str]:
        """Load general sign language terminology"""
        return {
            # Common ASL terms with enhanced descriptions
            "HELLO": "greeting, salutation",
            "GOODBYE": "farewell, departure",
            "THANK-YOU": "expression of gratitude",
            "PLEASE": "polite request marker",
            "SORRY": "apology, regret",
            "YES": "affirmative response",
            "NO": "negative response",
            "GOOD": "positive evaluation",
            "BAD": "negative evaluation",
            "HAPPY": "positive emotional state",
            "SAD": "negative emotional state",
            "LOVE": "strong positive emotion",
            "LIKE": "preference, enjoyment",
            "WANT": "desire, wish",
            "NEED": "necessity, requirement",
            "GO": "movement away",
            "COME": "movement toward",
            "STOP": "cessation of action",
            "START": "beginning of action",
            "FINISH": "completion of action"
        }
    
    def load_termbank(self, domain: str, file_path: str) -> bool:
        """Load terminology database for a specific domain"""
        try:
            # Mock termbank data for development
            mock_termbanks = {
                "medical": {
                    "DOCTOR": {
                        "enhanced": "medical doctor, physician",
                        "definition": "A licensed medical practitioner",
                        "context": "healthcare, medical consultation"
                    },
                    "MEDICINE": {
                        "enhanced": "medication, pharmaceutical treatment",
                        "definition": "Substance used to treat illness",
                        "context": "healthcare, treatment"
                    },
                    "HOSPITAL": {
                        "enhanced": "medical facility, healthcare institution",
                        "definition": "Institution providing medical care",
                        "context": "healthcare, emergency"
                    },
                    "PAIN": {
                        "enhanced": "physical discomfort, medical symptom",
                        "definition": "Unpleasant sensory experience",
                        "context": "healthcare, symptoms"
                    }
                },
                "legal": {
                    "LAWYER": {
                        "enhanced": "attorney, legal counsel",
                        "definition": "Licensed legal practitioner",
                        "context": "legal proceedings, consultation"
                    },
                    "COURT": {
                        "enhanced": "judicial tribunal, legal venue",
                        "definition": "Place where legal proceedings occur",
                        "context": "legal system, justice"
                    },
                    "CONTRACT": {
                        "enhanced": "legal agreement, binding document",
                        "definition": "Legally enforceable agreement",
                        "context": "legal documents, business"
                    }
                },
                "education": {
                    "TEACHER": {
                        "enhanced": "educator, instructor",
                        "definition": "Professional who provides education",
                        "context": "academic, learning environment"
                    },
                    "SCHOOL": {
                        "enhanced": "educational institution, learning facility",
                        "definition": "Institution for teaching students",
                        "context": "education, academic"
                    },
                    "STUDENT": {
                        "enhanced": "learner, pupil",
                        "definition": "Person engaged in learning",
                        "context": "education, academic"
                    },
                    "BOOK": {
                        "enhanced": "textbook, educational material",
                        "definition": "Written or printed work for learning",
                        "context": "education, study materials"
                    }
                }
            }
            
            if domain in mock_termbanks:
                self.termbanks[domain] = mock_termbanks[domain]
                logger.info(f"Loaded {len(self.termbanks[domain])} terms for domain '{domain}'")
                return True
            else:
                logger.warning(f"No mock data available for domain '{domain}'")
                return False
                
        except Exception as e:
            logger.error(f"Error loading termbank for domain '{domain}': {e}")
            return False
    
    async def enhance_translation(self, 
                                text: str,
                                domain: str = "general",
                                context: Dict[str, Any] = None) -> List[TermMatch]:
        """Enhance translation with domain-specific terminology"""
        try:
            matches = []
            
            # Tokenize text
            tokens = self._tokenize_text(text)
            
            # Find terminology matches
            for token in tokens:
                # Check domain-specific terms first
                if domain in self.termbanks:
                    domain_match = self._find_domain_match(token, domain)
                    if domain_match:
                        matches.append(domain_match)
                        continue
                
                # Check general terms
                general_match = self._find_general_match(token)
                if general_match:
                    matches.append(general_match)
            
            # Apply contextual enhancements
            if context:
                matches = self._apply_contextual_enhancements(matches, context)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error enhancing translation: {e}")
            return []
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for terminology matching"""
        try:
            # Simple tokenization - split on whitespace and punctuation
            tokens = re.findall(r'\b\w+\b', text.upper())
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return []
    
    def _find_domain_match(self, token: str, domain: str) -> Optional[TermMatch]:
        """Find match in domain-specific terminology"""
        try:
            if domain not in self.termbanks:
                return None
            
            termbank = self.termbanks[domain]
            
            if token in termbank:
                term_data = termbank[token]
                return TermMatch(
                    original_term=token,
                    enhanced_term=term_data.get("enhanced", token),
                    confidence=0.9,  # High confidence for exact matches
                    domain=domain,
                    definition=term_data.get("definition"),
                    context=term_data.get("context")
                )
            
            # Try fuzzy matching
            fuzzy_match = self._fuzzy_match(token, list(termbank.keys()))
            if fuzzy_match:
                term_data = termbank[fuzzy_match]
                return TermMatch(
                    original_term=token,
                    enhanced_term=term_data.get("enhanced", fuzzy_match),
                    confidence=0.7,  # Lower confidence for fuzzy matches
                    domain=domain,
                    definition=term_data.get("definition"),
                    context=term_data.get("context")
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding domain match: {e}")
            return None
    
    def _find_general_match(self, token: str) -> Optional[TermMatch]:
        """Find match in general terminology"""
        try:
            if token in self.general_terms:
                return TermMatch(
                    original_term=token,
                    enhanced_term=self.general_terms[token],
                    confidence=0.8,
                    domain="general"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding general match: {e}")
            return None
    
    def _fuzzy_match(self, token: str, candidates: List[str], threshold: float = 0.8) -> Optional[str]:
        """Find fuzzy match using simple string similarity"""
        try:
            best_match = None
            best_score = 0.0
            
            for candidate in candidates:
                # Simple Jaccard similarity
                set1 = set(token.lower())
                set2 = set(candidate.lower())
                
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                
                similarity = intersection / union if union > 0 else 0.0
                
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = candidate
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error in fuzzy matching: {e}")
            return None
    
    def _apply_contextual_enhancements(self, 
                                     matches: List[TermMatch],
                                     context: Dict[str, Any]) -> List[TermMatch]:
        """Apply contextual enhancements to matches"""
        try:
            enhanced_matches = []
            
            # Get context information
            topic_tracking = context.get("topic_tracking", {})
            current_topics = topic_tracking.get("current_topics", [])
            
            for match in matches:
                enhanced_match = match
                
                # Boost confidence if term relates to current topics
                if any(topic.lower() in match.enhanced_term.lower() for topic in current_topics):
                    enhanced_match.confidence = min(1.0, match.confidence + 0.1)
                
                # Add contextual information
                if current_topics:
                    enhanced_match.context = f"Related to: {', '.join(current_topics[:3])}"
                
                enhanced_matches.append(enhanced_match)
            
            return enhanced_matches
            
        except Exception as e:
            logger.error(f"Error applying contextual enhancements: {e}")
            return matches
    
    def get_domain_statistics(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a domain's terminology"""
        try:
            if domain not in self.termbanks:
                return {}
            
            termbank = self.termbanks[domain]
            
            return {
                "domain": domain,
                "total_terms": len(termbank),
                "terms_with_definitions": sum(1 for term_data in termbank.values() 
                                            if term_data.get("definition")),
                "terms_with_context": sum(1 for term_data in termbank.values() 
                                        if term_data.get("context")),
                "sample_terms": list(termbank.keys())[:5]
            }
            
        except Exception as e:
            logger.error(f"Error getting domain statistics: {e}")
            return {}
    
    def add_custom_term(self, 
                       domain: str,
                       original: str,
                       enhanced: str,
                       definition: Optional[str] = None,
                       context: Optional[str] = None) -> bool:
        """Add a custom term to a domain"""
        try:
            if domain not in self.termbanks:
                self.termbanks[domain] = {}
            
            self.termbanks[domain][original.upper()] = {
                "enhanced": enhanced,
                "definition": definition,
                "context": context
            }
            
            logger.info(f"Added custom term '{original}' to domain '{domain}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding custom term: {e}")
            return False
    
    def export_termbank(self, domain: str, file_path: str) -> bool:
        """Export termbank to file"""
        try:
            if domain not in self.termbanks:
                logger.error(f"Domain '{domain}' not found")
                return False
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.termbanks[domain], f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported termbank for domain '{domain}' to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting termbank: {e}")
            return False
