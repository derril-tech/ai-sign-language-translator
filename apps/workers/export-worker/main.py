#!/usr/bin/env python3
"""
Export Worker - Handles transcript generation, vocabulary highlighting, and analytics export
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Import shared utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from nats_client import NATSClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Export Worker", version="1.0.0")

@dataclass
class TranscriptEntry:
    timestamp: float
    speaker: str
    text: str
    confidence: float
    gloss_sequence: List[str]
    type: str  # 'sign', 'speech', 'system'
    vocabulary_highlights: List[Dict[str, Any]]

@dataclass
class SessionAnalytics:
    session_id: str
    start_time: float
    end_time: float
    duration: float
    participant_count: int
    total_translations: int
    average_confidence: float
    vocabulary_usage: Dict[str, int]
    domain_distribution: Dict[str, int]
    quality_metrics: Dict[str, float]
    turn_statistics: Dict[str, Any]

class ExportRequest(BaseModel):
    session_id: str
    export_type: str  # 'transcript', 'analytics', 'vocabulary', 'full'
    format: str  # 'pdf', 'json', 'csv', 'srt'
    options: Dict[str, Any] = {}
    privacy_settings: Dict[str, bool] = {}

class ExportResponse(BaseModel):
    export_id: str
    session_id: str
    status: str
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    created_at: float
    metadata: Dict[str, Any] = {}

class ExportWorker:
    def __init__(self):
        self.nats_client = NATSClient()
        self.export_dir = Path("exports")
        self.export_dir.mkdir(exist_ok=True)
        
        # Export job tracking
        self.active_exports: Dict[str, Dict[str, Any]] = {}
        
    async def start(self):
        """Start the export worker"""
        await self.nats_client.connect()
        await self.nats_client.subscribe("export.request", self.handle_export_request, queue="export-workers")
        logger.info("Export worker started and listening for export requests")
        
    async def handle_export_request(self, data: Dict[str, Any]):
        """Handle incoming export requests"""
        try:
            request = ExportRequest(**data)
            export_id = f"export_{int(time.time())}_{request.session_id}"
            
            logger.info(f"Processing export request {export_id} for session {request.session_id}")
            
            # Track export job
            self.active_exports[export_id] = {
                "status": "processing",
                "started_at": time.time(),
                "request": request
            }
            
            # Process export based on type
            result = await self.process_export(export_id, request)
            
            # Update job status
            self.active_exports[export_id]["status"] = "completed"
            self.active_exports[export_id]["result"] = result
            
            # Publish completion
            await self.nats_client.publish("export.completed", {
                "export_id": export_id,
                "session_id": request.session_id,
                "result": asdict(result)
            })
            
            logger.info(f"Export {export_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing export request: {e}")
            
            if 'export_id' in locals():
                self.active_exports[export_id]["status"] = "failed"
                self.active_exports[export_id]["error"] = str(e)
                
                await self.nats_client.publish("export.failed", {
                    "export_id": export_id,
                    "session_id": data.get("session_id", "unknown"),
                    "error": str(e)
                })
    
    async def process_export(self, export_id: str, request: ExportRequest) -> ExportResponse:
        """Process export request based on type and format"""
        
        # Fetch session data (mock implementation)
        session_data = await self.fetch_session_data(request.session_id)
        
        if request.export_type == "transcript":
            return await self.export_transcript(export_id, session_data, request)
        elif request.export_type == "analytics":
            return await self.export_analytics(export_id, session_data, request)
        elif request.export_type == "vocabulary":
            return await self.export_vocabulary(export_id, session_data, request)
        elif request.export_type == "full":
            return await self.export_full_report(export_id, session_data, request)
        else:
            raise ValueError(f"Unknown export type: {request.export_type}")
    
    async def fetch_session_data(self, session_id: str) -> Dict[str, Any]:
        """Fetch session data from database (mock implementation)"""
        # In production, this would query the database
        return {
            "session_id": session_id,
            "start_time": time.time() - 3600,
            "end_time": time.time(),
            "transcript": [
                {
                    "timestamp": time.time() - 3500,
                    "speaker": "User A",
                    "text": "Hello, how are you today?",
                    "confidence": 0.92,
                    "gloss_sequence": ["HELLO", "HOW", "YOU", "TODAY"],
                    "type": "sign",
                    "vocabulary_highlights": [
                        {"term": "HELLO", "domain": "general", "confidence": 0.95}
                    ]
                },
                {
                    "timestamp": time.time() - 3400,
                    "speaker": "User B",
                    "text": "I am doing well, thank you.",
                    "confidence": 0.88,
                    "gloss_sequence": ["I", "DOING", "WELL", "THANK", "YOU"],
                    "type": "speech",
                    "vocabulary_highlights": [
                        {"term": "THANK", "domain": "general", "confidence": 0.90}
                    ]
                }
            ],
            "analytics": {
                "participant_count": 2,
                "total_translations": 156,
                "average_confidence": 0.87,
                "vocabulary_usage": {"HELLO": 5, "THANK": 8, "YOU": 12},
                "domain_distribution": {"general": 80, "medical": 15, "business": 5},
                "quality_metrics": {
                    "avg_fps": 29.8,
                    "avg_latency": 245,
                    "pose_accuracy": 0.91,
                    "gloss_accuracy": 0.85
                }
            }
        }
    
    async def export_transcript(self, export_id: str, session_data: Dict[str, Any], request: ExportRequest) -> ExportResponse:
        """Export transcript in requested format"""
        
        if request.format == "pdf":
            file_path = await self.generate_transcript_pdf(export_id, session_data, request)
        elif request.format == "srt":
            file_path = await self.generate_transcript_srt(export_id, session_data, request)
        elif request.format == "json":
            file_path = await self.generate_transcript_json(export_id, session_data, request)
        elif request.format == "csv":
            file_path = await self.generate_transcript_csv(export_id, session_data, request)
        else:
            raise ValueError(f"Unsupported transcript format: {request.format}")
        
        file_size = os.path.getsize(file_path)
        
        return ExportResponse(
            export_id=export_id,
            session_id=request.session_id,
            status="completed",
            file_path=str(file_path),
            file_size=file_size,
            created_at=time.time(),
            metadata={
                "export_type": "transcript",
                "format": request.format,
                "entry_count": len(session_data["transcript"])
            }
        )
    
    async def generate_transcript_pdf(self, export_id: str, session_data: Dict[str, Any], request: ExportRequest) -> str:
        """Generate PDF transcript"""
        file_path = self.export_dir / f"{export_id}_transcript.pdf"
        
        doc = SimpleDocTemplate(str(file_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center
        )
        story.append(Paragraph("Sign Language Translation Transcript", title_style))
        
        # Session info
        session_info = [
            ["Session ID:", session_data["session_id"]],
            ["Date:", datetime.fromtimestamp(session_data["start_time"]).strftime("%Y-%m-%d %H:%M:%S")],
            ["Duration:", f"{(session_data['end_time'] - session_data['start_time']) / 60:.1f} minutes"],
            ["Entries:", str(len(session_data["transcript"]))]
        ]
        
        info_table = Table(session_info, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        # Transcript entries
        story.append(Paragraph("Transcript", styles['Heading2']))
        
        for entry in session_data["transcript"]:
            timestamp = datetime.fromtimestamp(entry["timestamp"]).strftime("%H:%M:%S")
            speaker = entry["speaker"] if not request.privacy_settings.get("redact_names", False) else "[SPEAKER]"
            text = self.apply_privacy_redaction(entry["text"], request.privacy_settings)
            confidence = f"{entry['confidence']*100:.0f}%"
            
            # Entry header
            header_text = f"[{timestamp}] {speaker} ({confidence} confidence)"
            story.append(Paragraph(header_text, styles['Normal']))
            
            # Entry text
            story.append(Paragraph(text, styles['BodyText']))
            
            # Gloss sequence if available
            if entry.get("gloss_sequence"):
                gloss_text = " ".join(entry["gloss_sequence"])
                story.append(Paragraph(f"<i>Gloss: {gloss_text}</i>", styles['BodyText']))
            
            # Vocabulary highlights
            if entry.get("vocabulary_highlights"):
                highlights = [f"{h['term']} ({h['domain']})" for h in entry["vocabulary_highlights"]]
                if highlights:
                    story.append(Paragraph(f"<i>Vocabulary: {', '.join(highlights)}</i>", styles['BodyText']))
            
            story.append(Spacer(1, 12))
        
        doc.build(story)
        return str(file_path)
    
    async def generate_transcript_srt(self, export_id: str, session_data: Dict[str, Any], request: ExportRequest) -> str:
        """Generate SRT subtitle file"""
        file_path = self.export_dir / f"{export_id}_transcript.srt"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(session_data["transcript"], 1):
                start_time = entry["timestamp"]
                end_time = start_time + 3  # 3 second duration
                
                start_srt = self.timestamp_to_srt(start_time)
                end_srt = self.timestamp_to_srt(end_time)
                
                speaker = entry["speaker"] if not request.privacy_settings.get("redact_names", False) else "[SPEAKER]"
                text = self.apply_privacy_redaction(entry["text"], request.privacy_settings)
                
                f.write(f"{i}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{speaker}: {text}\n\n")
        
        return str(file_path)
    
    async def generate_transcript_json(self, export_id: str, session_data: Dict[str, Any], request: ExportRequest) -> str:
        """Generate JSON transcript"""
        file_path = self.export_dir / f"{export_id}_transcript.json"
        
        # Apply privacy settings
        transcript = []
        for entry in session_data["transcript"]:
            processed_entry = entry.copy()
            if request.privacy_settings.get("redact_names", False):
                processed_entry["speaker"] = "[SPEAKER]"
            processed_entry["text"] = self.apply_privacy_redaction(entry["text"], request.privacy_settings)
            transcript.append(processed_entry)
        
        export_data = {
            "export_info": {
                "export_id": export_id,
                "generated_at": datetime.now().isoformat(),
                "privacy_settings": request.privacy_settings,
                "format": "json"
            },
            "session": {
                "session_id": session_data["session_id"],
                "start_time": session_data["start_time"],
                "end_time": session_data["end_time"],
                "duration": session_data["end_time"] - session_data["start_time"]
            },
            "transcript": transcript
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    async def generate_transcript_csv(self, export_id: str, session_data: Dict[str, Any], request: ExportRequest) -> str:
        """Generate CSV transcript"""
        file_path = self.export_dir / f"{export_id}_transcript.csv"
        
        # Prepare data
        rows = []
        for entry in session_data["transcript"]:
            speaker = entry["speaker"] if not request.privacy_settings.get("redact_names", False) else "[SPEAKER]"
            text = self.apply_privacy_redaction(entry["text"], request.privacy_settings)
            gloss = " ".join(entry.get("gloss_sequence", []))
            
            rows.append({
                "timestamp": datetime.fromtimestamp(entry["timestamp"]).isoformat(),
                "speaker": speaker,
                "text": text,
                "confidence": entry["confidence"],
                "type": entry["type"],
                "gloss_sequence": gloss
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False, encoding='utf-8')
        
        return str(file_path)
    
    async def export_analytics(self, export_id: str, session_data: Dict[str, Any], request: ExportRequest) -> ExportResponse:
        """Export session analytics"""
        
        analytics = self.calculate_session_analytics(session_data)
        
        if request.format == "pdf":
            file_path = await self.generate_analytics_pdf(export_id, analytics, request)
        elif request.format == "json":
            file_path = await self.generate_analytics_json(export_id, analytics, request)
        else:
            raise ValueError(f"Unsupported analytics format: {request.format}")
        
        file_size = os.path.getsize(file_path)
        
        return ExportResponse(
            export_id=export_id,
            session_id=request.session_id,
            status="completed",
            file_path=str(file_path),
            file_size=file_size,
            created_at=time.time(),
            metadata={
                "export_type": "analytics",
                "format": request.format,
                "metrics_count": len(analytics.quality_metrics)
            }
        )
    
    def calculate_session_analytics(self, session_data: Dict[str, Any]) -> SessionAnalytics:
        """Calculate comprehensive session analytics"""
        
        transcript = session_data["transcript"]
        start_time = session_data["start_time"]
        end_time = session_data["end_time"]
        
        # Basic metrics
        total_translations = len(transcript)
        average_confidence = sum(entry["confidence"] for entry in transcript) / max(total_translations, 1)
        
        # Vocabulary usage
        vocabulary_usage = {}
        for entry in transcript:
            for gloss in entry.get("gloss_sequence", []):
                vocabulary_usage[gloss] = vocabulary_usage.get(gloss, 0) + 1
        
        # Domain distribution
        domain_distribution = {}
        for entry in transcript:
            for highlight in entry.get("vocabulary_highlights", []):
                domain = highlight.get("domain", "unknown")
                domain_distribution[domain] = domain_distribution.get(domain, 0) + 1
        
        # Turn statistics
        speakers = set(entry["speaker"] for entry in transcript)
        turn_stats = {
            "total_speakers": len(speakers),
            "turns_per_speaker": {speaker: sum(1 for entry in transcript if entry["speaker"] == speaker) 
                                for speaker in speakers},
            "avg_turn_length": sum(len(entry["text"].split()) for entry in transcript) / max(total_translations, 1)
        }
        
        return SessionAnalytics(
            session_id=session_data["session_id"],
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            participant_count=len(speakers),
            total_translations=total_translations,
            average_confidence=average_confidence,
            vocabulary_usage=vocabulary_usage,
            domain_distribution=domain_distribution,
            quality_metrics=session_data.get("analytics", {}).get("quality_metrics", {}),
            turn_statistics=turn_stats
        )
    
    async def generate_analytics_pdf(self, export_id: str, analytics: SessionAnalytics, request: ExportRequest) -> str:
        """Generate analytics PDF report"""
        file_path = self.export_dir / f"{export_id}_analytics.pdf"
        
        doc = SimpleDocTemplate(str(file_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph("Session Analytics Report", title_style))
        
        # Session overview
        story.append(Paragraph("Session Overview", styles['Heading2']))
        
        overview_data = [
            ["Session ID:", analytics.session_id],
            ["Duration:", f"{analytics.duration / 60:.1f} minutes"],
            ["Participants:", str(analytics.participant_count)],
            ["Total Translations:", str(analytics.total_translations)],
            ["Average Confidence:", f"{analytics.average_confidence * 100:.1f}%"]
        ]
        
        overview_table = Table(overview_data, colWidths=[2*inch, 4*inch])
        overview_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 20))
        
        # Top vocabulary
        story.append(Paragraph("Top Vocabulary Usage", styles['Heading2']))
        
        top_vocab = sorted(analytics.vocabulary_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        vocab_data = [["Term", "Usage Count"]]
        vocab_data.extend(top_vocab)
        
        vocab_table = Table(vocab_data, colWidths=[3*inch, 2*inch])
        vocab_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(vocab_table)
        story.append(Spacer(1, 20))
        
        # Domain distribution
        story.append(Paragraph("Domain Distribution", styles['Heading2']))
        
        domain_data = [["Domain", "Usage Count", "Percentage"]]
        total_domain_usage = sum(analytics.domain_distribution.values())
        
        for domain, count in analytics.domain_distribution.items():
            percentage = (count / max(total_domain_usage, 1)) * 100
            domain_data.append([domain.title(), str(count), f"{percentage:.1f}%"])
        
        domain_table = Table(domain_data, colWidths=[2*inch, 2*inch, 2*inch])
        domain_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(domain_table)
        
        doc.build(story)
        return str(file_path)
    
    async def generate_analytics_json(self, export_id: str, analytics: SessionAnalytics, request: ExportRequest) -> str:
        """Generate analytics JSON report"""
        file_path = self.export_dir / f"{export_id}_analytics.json"
        
        export_data = {
            "export_info": {
                "export_id": export_id,
                "generated_at": datetime.now().isoformat(),
                "format": "json"
            },
            "analytics": asdict(analytics)
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    async def export_vocabulary(self, export_id: str, session_data: Dict[str, Any], request: ExportRequest) -> ExportResponse:
        """Export vocabulary highlights and usage"""
        
        vocabulary_data = self.extract_vocabulary_data(session_data)
        
        if request.format == "json":
            file_path = await self.generate_vocabulary_json(export_id, vocabulary_data, request)
        elif request.format == "csv":
            file_path = await self.generate_vocabulary_csv(export_id, vocabulary_data, request)
        else:
            raise ValueError(f"Unsupported vocabulary format: {request.format}")
        
        file_size = os.path.getsize(file_path)
        
        return ExportResponse(
            export_id=export_id,
            session_id=request.session_id,
            status="completed",
            file_path=str(file_path),
            file_size=file_size,
            created_at=time.time(),
            metadata={
                "export_type": "vocabulary",
                "format": request.format,
                "unique_terms": len(vocabulary_data)
            }
        )
    
    def extract_vocabulary_data(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract vocabulary usage data from session"""
        
        vocabulary_map = {}
        
        for entry in session_data["transcript"]:
            for highlight in entry.get("vocabulary_highlights", []):
                term = highlight["term"]
                if term not in vocabulary_map:
                    vocabulary_map[term] = {
                        "term": term,
                        "domain": highlight.get("domain", "unknown"),
                        "usage_count": 0,
                        "first_used": entry["timestamp"],
                        "last_used": entry["timestamp"],
                        "confidence_scores": [],
                        "contexts": []
                    }
                
                vocab_entry = vocabulary_map[term]
                vocab_entry["usage_count"] += 1
                vocab_entry["last_used"] = entry["timestamp"]
                vocab_entry["confidence_scores"].append(highlight.get("confidence", 0.8))
                vocab_entry["contexts"].append({
                    "timestamp": entry["timestamp"],
                    "speaker": entry["speaker"],
                    "context": entry["text"][:100] + "..." if len(entry["text"]) > 100 else entry["text"]
                })
        
        # Calculate average confidence for each term
        for term_data in vocabulary_map.values():
            if term_data["confidence_scores"]:
                term_data["average_confidence"] = sum(term_data["confidence_scores"]) / len(term_data["confidence_scores"])
            else:
                term_data["average_confidence"] = 0.0
            
            # Keep only recent contexts (last 5)
            term_data["contexts"] = term_data["contexts"][-5:]
            del term_data["confidence_scores"]  # Remove raw scores
        
        return list(vocabulary_map.values())
    
    async def generate_vocabulary_json(self, export_id: str, vocabulary_data: List[Dict[str, Any]], request: ExportRequest) -> str:
        """Generate vocabulary JSON export"""
        file_path = self.export_dir / f"{export_id}_vocabulary.json"
        
        export_data = {
            "export_info": {
                "export_id": export_id,
                "generated_at": datetime.now().isoformat(),
                "format": "json"
            },
            "vocabulary": vocabulary_data
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    async def generate_vocabulary_csv(self, export_id: str, vocabulary_data: List[Dict[str, Any]], request: ExportRequest) -> str:
        """Generate vocabulary CSV export"""
        file_path = self.export_dir / f"{export_id}_vocabulary.csv"
        
        # Flatten data for CSV
        rows = []
        for vocab in vocabulary_data:
            rows.append({
                "term": vocab["term"],
                "domain": vocab["domain"],
                "usage_count": vocab["usage_count"],
                "average_confidence": vocab["average_confidence"],
                "first_used": datetime.fromtimestamp(vocab["first_used"]).isoformat(),
                "last_used": datetime.fromtimestamp(vocab["last_used"]).isoformat(),
                "recent_contexts": "; ".join([ctx["context"] for ctx in vocab["contexts"][-3:]])
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False, encoding='utf-8')
        
        return str(file_path)
    
    async def export_full_report(self, export_id: str, session_data: Dict[str, Any], request: ExportRequest) -> ExportResponse:
        """Export comprehensive report with all data"""
        
        # Generate all components
        transcript_response = await self.export_transcript(f"{export_id}_transcript", session_data, 
                                                         ExportRequest(session_id=request.session_id, 
                                                                     export_type="transcript", 
                                                                     format="json",
                                                                     privacy_settings=request.privacy_settings))
        
        analytics_response = await self.export_analytics(f"{export_id}_analytics", session_data,
                                                        ExportRequest(session_id=request.session_id,
                                                                    export_type="analytics",
                                                                    format="json"))
        
        vocabulary_response = await self.export_vocabulary(f"{export_id}_vocabulary", session_data,
                                                         ExportRequest(session_id=request.session_id,
                                                                     export_type="vocabulary",
                                                                     format="json"))
        
        # Combine into full report
        file_path = self.export_dir / f"{export_id}_full_report.json"
        
        full_report = {
            "export_info": {
                "export_id": export_id,
                "generated_at": datetime.now().isoformat(),
                "format": "json",
                "type": "full_report"
            },
            "session_id": request.session_id,
            "components": {
                "transcript": json.loads(open(transcript_response.file_path).read()),
                "analytics": json.loads(open(analytics_response.file_path).read()),
                "vocabulary": json.loads(open(vocabulary_response.file_path).read())
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        
        file_size = os.path.getsize(file_path)
        
        return ExportResponse(
            export_id=export_id,
            session_id=request.session_id,
            status="completed",
            file_path=str(file_path),
            file_size=file_size,
            created_at=time.time(),
            metadata={
                "export_type": "full_report",
                "format": "json",
                "components": ["transcript", "analytics", "vocabulary"]
            }
        )
    
    def apply_privacy_redaction(self, text: str, privacy_settings: Dict[str, bool]) -> str:
        """Apply privacy redaction to text"""
        if not privacy_settings:
            return text
        
        redacted_text = text
        
        if privacy_settings.get("redact_names", False):
            # Simple name redaction (in production, use NLP)
            import re
            redacted_text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME REDACTED]', redacted_text)
        
        if privacy_settings.get("redact_phone", False):
            import re
            redacted_text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE REDACTED]', redacted_text)
        
        if privacy_settings.get("redact_email", False):
            import re
            redacted_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REDACTED]', redacted_text)
        
        return redacted_text
    
    def timestamp_to_srt(self, timestamp: float) -> str:
        """Convert timestamp to SRT format"""
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%H:%M:%S,000")

# Global worker instance
worker = ExportWorker()

@app.on_event("startup")
async def startup_event():
    await worker.start()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "export-worker"}

@app.get("/exports/{export_id}")
async def get_export_status(export_id: str):
    if export_id not in worker.active_exports:
        raise HTTPException(status_code=404, detail="Export not found")
    
    return worker.active_exports[export_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
