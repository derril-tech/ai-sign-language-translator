import asyncio
import logging
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from shared.nats_client import NATSClient
from terminology_enhancer import TerminologyEnhancer
from vector_store import VectorStore
from domain_adapter import DomainAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Worker", version="1.0.0")

class RAGRequest(BaseModel):
    session_id: str
    timestamp: float
    translated_text: str
    confidence: float
    semantic_roles: List[Dict[str, Any]]
    discourse_markers: List[str]
    context_info: Dict[str, Any]
    domain: Optional[str] = "general"

class RAGResponse(BaseModel):
    session_id: str
    timestamp: float
    enhanced_text: str
    terminology_matches: List[Dict[str, Any]]
    domain_adaptations: List[Dict[str, Any]]
    confidence_boost: float
    processing_time_ms: float

class RAGWorker:
    def __init__(self):
        self.nats_client = NATSClient()
        self.terminology_enhancer = TerminologyEnhancer()
        self.vector_store = VectorStore(embedding_dim=384)
        self.domain_adapter = DomainAdapter()
        
        # Load terminology databases
        self._load_terminology_databases()
        
    def _load_terminology_databases(self):
        """Load terminology databases for different domains"""
        try:
            # Load common sign language terminology
            self.terminology_enhancer.load_termbank("medical", "data/termbanks/medical.json")
            self.terminology_enhancer.load_termbank("legal", "data/termbanks/legal.json")
            self.terminology_enhancer.load_termbank("education", "data/termbanks/education.json")
            logger.info("Terminology databases loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load terminology databases: {e}")
            logger.info("Using mock terminology for development")
        
    async def start(self):
        """Start the RAG worker"""
        await self.nats_client.connect()
        await self.nats_client.subscribe("rag.enrich", self.process_translation, queue="rag-workers")
        logger.info("RAG worker started and listening for translations")
        
    async def process_translation(self, data: Dict[str, Any]):
        """Process translation to enhance with terminology"""
        start_time = time.time()
        
        try:
            request = RAGRequest(**data)
            
            # Detect domain from context
            detected_domain = self.domain_adapter.detect_domain(
                text=request.translated_text,
                context=request.context_info,
                semantic_roles=request.semantic_roles
            )
            
            # Enhance with terminology
            terminology_matches = await self.terminology_enhancer.enhance_translation(
                text=request.translated_text,
                domain=detected_domain or request.domain,
                context=request.context_info
            )
            
            # Apply domain adaptations
            domain_adaptations = self.domain_adapter.apply_adaptations(
                text=request.translated_text,
                domain=detected_domain or request.domain,
                terminology_matches=terminology_matches
            )
            
            # Generate enhanced text
            enhanced_text = self._generate_enhanced_text(
                original_text=request.translated_text,
                terminology_matches=terminology_matches,
                domain_adaptations=domain_adaptations
            )
            
            # Calculate confidence boost
            confidence_boost = self._calculate_confidence_boost(
                terminology_matches, domain_adaptations
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            response = RAGResponse(
                session_id=request.session_id,
                timestamp=request.timestamp,
                enhanced_text=enhanced_text,
                terminology_matches=terminology_matches,
                domain_adaptations=domain_adaptations,
                confidence_boost=confidence_boost,
                processing_time_ms=processing_time
            )
            
            # Publish enhanced translation
            await self.nats_client.publish("translation.final", response.dict())
            logger.debug(f"Enhanced translation for session {request.session_id} in {processing_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error processing RAG enhancement: {e}")
    
    def _generate_enhanced_text(self, 
                              original_text: str,
                              terminology_matches: List[Dict[str, Any]],
                              domain_adaptations: List[Dict[str, Any]]) -> str:
        """Generate enhanced text with terminology and adaptations"""
        try:
            enhanced_text = original_text
            
            # Apply terminology enhancements
            for match in terminology_matches:
                if hasattr(match, 'original_term') and hasattr(match, 'enhanced_term'):
                    original = match.original_term
                    enhanced = match.enhanced_term
                    if original and enhanced and original != enhanced:
                        enhanced_text = enhanced_text.replace(original.lower(), enhanced.lower())
            
            return enhanced_text
            
        except Exception as e:
            logger.error(f"Error generating enhanced text: {e}")
            return original_text
    
    def _calculate_confidence_boost(self, 
                                  terminology_matches: List[Dict[str, Any]],
                                  domain_adaptations: List[Dict[str, Any]]) -> float:
        """Calculate confidence boost from enhancements"""
        try:
            boost = 0.0
            
            # Boost from terminology matches
            high_conf_matches = sum(1 for match in terminology_matches 
                                  if hasattr(match, 'confidence') and match.confidence > 0.8)
            boost += high_conf_matches * 0.05  # 5% boost per high-confidence match
            
            # Boost from domain adaptations
            boost += len(domain_adaptations) * 0.02  # 2% boost per adaptation
            
            return min(boost, 0.2)  # Cap at 20% boost
            
        except Exception as e:
            logger.error(f"Error calculating confidence boost: {e}")
            return 0.0

# Global worker instance
worker = RAGWorker()

@app.on_event("startup")
async def startup_event():
    await worker.start()

@app.on_event("shutdown")
async def shutdown_event():
    await worker.nats_client.disconnect()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "rag-worker"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
