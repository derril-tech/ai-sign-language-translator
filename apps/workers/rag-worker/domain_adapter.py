import logging
from typing import Dict, List, Any, Optional
import re

logger = logging.getLogger(__name__)

class DomainAdapter:
    """Adapts translations based on detected domain context"""
    
    def __init__(self):
        self.domain_keywords = {
            "medical": ["doctor", "hospital", "medicine", "pain", "treatment", "health", "patient", "nurse"],
            "legal": ["lawyer", "court", "contract", "law", "judge", "attorney", "legal", "rights"],
            "education": ["teacher", "school", "student", "book", "learn", "study", "class", "homework"],
            "business": ["work", "job", "company", "meeting", "project", "manager", "office", "client"],
            "family": ["mother", "father", "child", "family", "home", "parent", "brother", "sister"]
        }
        
        self.domain_adaptations = {
            "medical": {
                "formality_level": "high",
                "precision_required": True,
                "common_phrases": ["medical history", "symptoms", "diagnosis", "treatment plan"]
            },
            "legal": {
                "formality_level": "very_high",
                "precision_required": True,
                "common_phrases": ["legal rights", "court proceedings", "legal document", "attorney-client"]
            },
            "education": {
                "formality_level": "medium",
                "precision_required": False,
                "common_phrases": ["learning objectives", "academic performance", "educational goals"]
            }
        }
    
    def detect_domain(self, 
                     text: str,
                     context: Dict[str, Any],
                     semantic_roles: List[Dict[str, Any]]) -> Optional[str]:
        """Detect the domain of the conversation"""
        try:
            text_lower = text.lower()
            domain_scores = {}
            
            # Score based on keywords
            for domain, keywords in self.domain_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    domain_scores[domain] = score
            
            # Boost score based on context topics
            topic_tracking = context.get("topic_tracking", {})
            current_topics = topic_tracking.get("current_topics", [])
            
            for topic in current_topics:
                topic_lower = topic.lower()
                for domain, keywords in self.domain_keywords.items():
                    if any(keyword in topic_lower for keyword in keywords):
                        domain_scores[domain] = domain_scores.get(domain, 0) + 2
            
            # Return domain with highest score
            if domain_scores:
                return max(domain_scores, key=domain_scores.get)
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting domain: {e}")
            return None
    
    def apply_adaptations(self, 
                         text: str,
                         domain: str,
                         terminology_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply domain-specific adaptations"""
        try:
            adaptations = []
            
            if domain not in self.domain_adaptations:
                return adaptations
            
            domain_config = self.domain_adaptations[domain]
            
            # Formality adaptation
            formality_adaptation = self._adapt_formality(text, domain_config["formality_level"])
            if formality_adaptation:
                adaptations.append(formality_adaptation)
            
            # Precision adaptation
            if domain_config["precision_required"]:
                precision_adaptation = self._adapt_precision(text, terminology_matches)
                if precision_adaptation:
                    adaptations.append(precision_adaptation)
            
            # Common phrases adaptation
            phrase_adaptation = self._adapt_common_phrases(text, domain_config["common_phrases"])
            if phrase_adaptation:
                adaptations.append(phrase_adaptation)
            
            return adaptations
            
        except Exception as e:
            logger.error(f"Error applying adaptations: {e}")
            return []
    
    def _adapt_formality(self, text: str, formality_level: str) -> Optional[Dict[str, Any]]:
        """Adapt text formality level"""
        try:
            if formality_level == "high" or formality_level == "very_high":
                # Suggest more formal alternatives
                informal_patterns = {
                    r'\bwanna\b': 'want to',
                    r'\bgonna\b': 'going to',
                    r'\bkinda\b': 'kind of',
                    r'\byeah\b': 'yes',
                    r'\bnope\b': 'no'
                }
                
                suggestions = []
                for pattern, replacement in informal_patterns.items():
                    if re.search(pattern, text, re.IGNORECASE):
                        suggestions.append(f"Consider using '{replacement}' instead of informal variant")
                
                if suggestions:
                    return {
                        "type": "formality",
                        "level": formality_level,
                        "suggestions": suggestions
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error adapting formality: {e}")
            return None
    
    def _adapt_precision(self, text: str, terminology_matches: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Adapt for precision requirements"""
        try:
            suggestions = []
            
            # Check for vague terms that should be more specific
            vague_terms = ["thing", "stuff", "something", "kind of", "sort of"]
            
            for term in vague_terms:
                if term in text.lower():
                    suggestions.append(f"Consider being more specific than '{term}'")
            
            # Suggest using enhanced terminology
            if terminology_matches:
                suggestions.append("Use precise terminology where available")
            
            if suggestions:
                return {
                    "type": "precision",
                    "suggestions": suggestions
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error adapting precision: {e}")
            return None
    
    def _adapt_common_phrases(self, text: str, common_phrases: List[str]) -> Optional[Dict[str, Any]]:
        """Suggest common domain phrases"""
        try:
            # Simple phrase suggestion based on context
            suggestions = []
            
            text_lower = text.lower()
            for phrase in common_phrases:
                # If part of the phrase is mentioned, suggest the full phrase
                phrase_words = phrase.lower().split()
                if any(word in text_lower for word in phrase_words):
                    suggestions.append(f"Consider using standard phrase: '{phrase}'")
            
            if suggestions:
                return {
                    "type": "common_phrases",
                    "suggestions": suggestions[:3]  # Limit to 3 suggestions
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error adapting common phrases: {e}")
            return None
