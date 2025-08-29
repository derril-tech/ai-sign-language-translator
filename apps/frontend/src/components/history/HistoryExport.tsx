'use client';

import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Download, 
  FileText, 
  Film, 
  Database,
  Eye,
  EyeOff,
  Calendar,
  Clock,
  User,
  Shield,
  Filter,
  Search
} from 'lucide-react';
import jsPDF from 'jspdf';

interface SessionRecord {
  id: string;
  start_time: string;
  end_time: string;
  duration: number;
  mode: 'sign-to-text' | 'text-to-sign' | 'conversation';
  participant_count: number;
  total_translations: number;
  average_confidence: number;
  transcript: TranscriptEntry[];
  metadata: {
    device_info: string;
    app_version: string;
    quality_metrics: any;
  };
}

interface TranscriptEntry {
  timestamp: number;
  speaker: string;
  text: string;
  confidence: number;
  type: 'sign' | 'speech' | 'system';
  gloss_sequence?: string[];
}

interface HistoryExportProps {
  className?: string;
}

export function HistoryExport({ className = '' }: HistoryExportProps) {
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [selectedSessions, setSelectedSessions] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [dateRange, setDateRange] = useState({ start: '', end: '' });
  const [privacyMode, setPrivacyMode] = useState(true);
  const [exportFormat, setExportFormat] = useState<'pdf' | 'srt' | 'json'>('pdf');
  const [isExporting, setIsExporting] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Mock session data for demonstration
  React.useEffect(() => {
    const mockSessions: SessionRecord[] = [
      {
        id: 'session_1',
        start_time: '2024-01-15T10:30:00Z',
        end_time: '2024-01-15T11:15:00Z',
        duration: 2700000, // 45 minutes
        mode: 'conversation',
        participant_count: 2,
        total_translations: 156,
        average_confidence: 0.87,
        transcript: [
          {
            timestamp: 0,
            speaker: 'User A',
            text: 'Hello, how are you today?',
            confidence: 0.92,
            type: 'sign',
            gloss_sequence: ['HELLO', 'HOW', 'YOU', 'TODAY']
          },
          {
            timestamp: 3000,
            speaker: 'User B',
            text: 'I am doing well, thank you.',
            confidence: 0.88,
            type: 'speech'
          }
        ],
        metadata: {
          device_info: 'Chrome 120.0.0 on Windows 11',
          app_version: '1.0.0',
          quality_metrics: { avg_fps: 29.8, avg_latency: 245 }
        }
      },
      {
        id: 'session_2',
        start_time: '2024-01-14T14:20:00Z',
        end_time: '2024-01-14T14:35:00Z',
        duration: 900000, // 15 minutes
        mode: 'sign-to-text',
        participant_count: 1,
        total_translations: 42,
        average_confidence: 0.79,
        transcript: [
          {
            timestamp: 0,
            speaker: 'User',
            text: 'I need help with my appointment.',
            confidence: 0.84,
            type: 'sign',
            gloss_sequence: ['I', 'NEED', 'HELP', 'MY', 'APPOINTMENT']
          }
        ],
        metadata: {
          device_info: 'Safari 17.2 on macOS 14',
          app_version: '1.0.0',
          quality_metrics: { avg_fps: 30.0, avg_latency: 198 }
        }
      }
    ];
    
    setSessions(mockSessions);
  }, []);

  // Filter sessions based on search and date range
  const filteredSessions = sessions.filter(session => {
    const matchesSearch = searchQuery === '' || 
      session.transcript.some(entry => 
        entry.text.toLowerCase().includes(searchQuery.toLowerCase())
      );
    
    const sessionDate = new Date(session.start_time).toISOString().split('T')[0];
    const matchesDateRange = 
      (!dateRange.start || sessionDate >= dateRange.start) &&
      (!dateRange.end || sessionDate <= dateRange.end);
    
    return matchesSearch && matchesDateRange;
  });

  // Toggle session selection
  const toggleSessionSelection = useCallback((sessionId: string) => {
    setSelectedSessions(prev => {
      const newSet = new Set(prev);
      if (newSet.has(sessionId)) {
        newSet.delete(sessionId);
      } else {
        newSet.add(sessionId);
      }
      return newSet;
    });
  }, []);

  // Select all filtered sessions
  const selectAllSessions = useCallback(() => {
    setSelectedSessions(new Set(filteredSessions.map(s => s.id)));
  }, [filteredSessions]);

  // Clear selection
  const clearSelection = useCallback(() => {
    setSelectedSessions(new Set());
  }, []);

  // Redact personal information
  const redactText = useCallback((text: string): string => {
    if (!privacyMode) return text;
    
    // Simple redaction patterns
    return text
      .replace(/\b[A-Z][a-z]+ [A-Z][a-z]+\b/g, '[NAME REDACTED]') // Names
      .replace(/\b\d{3}-\d{3}-\d{4}\b/g, '[PHONE REDACTED]') // Phone numbers
      .replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, '[EMAIL REDACTED]') // Emails
      .replace(/\b\d{1,5}\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b/gi, '[ADDRESS REDACTED]'); // Addresses
  }, [privacyMode]);

  // Export to PDF
  const exportToPDF = useCallback(async (sessions: SessionRecord[]) => {
    const pdf = new jsPDF();
    let yPosition = 20;
    
    // Title
    pdf.setFontSize(20);
    pdf.text('Sign Language Translation History', 20, yPosition);
    yPosition += 20;
    
    // Export info
    pdf.setFontSize(10);
    pdf.text(`Generated: ${new Date().toLocaleString()}`, 20, yPosition);
    pdf.text(`Privacy Mode: ${privacyMode ? 'Enabled' : 'Disabled'}`, 120, yPosition);
    yPosition += 15;
    
    sessions.forEach((session, sessionIndex) => {
      // Check if we need a new page
      if (yPosition > 250) {
        pdf.addPage();
        yPosition = 20;
      }
      
      // Session header
      pdf.setFontSize(14);
      pdf.text(`Session ${sessionIndex + 1}`, 20, yPosition);
      yPosition += 10;
      
      pdf.setFontSize(10);
      pdf.text(`Date: ${new Date(session.start_time).toLocaleString()}`, 20, yPosition);
      pdf.text(`Duration: ${Math.round(session.duration / 60000)} minutes`, 120, yPosition);
      yPosition += 8;
      
      pdf.text(`Mode: ${session.mode}`, 20, yPosition);
      pdf.text(`Confidence: ${Math.round(session.average_confidence * 100)}%`, 120, yPosition);
      yPosition += 15;
      
      // Transcript
      pdf.setFontSize(12);
      pdf.text('Transcript:', 20, yPosition);
      yPosition += 10;
      
      session.transcript.forEach(entry => {
        if (yPosition > 270) {
          pdf.addPage();
          yPosition = 20;
        }
        
        pdf.setFontSize(9);
        const timestamp = new Date(entry.timestamp).toLocaleTimeString();
        const speaker = privacyMode ? '[SPEAKER]' : entry.speaker;
        const text = redactText(entry.text);
        
        pdf.text(`[${timestamp}] ${speaker}: ${text}`, 25, yPosition);
        yPosition += 6;
      });
      
      yPosition += 10;
    });
    
    // Save PDF
    const filename = `sign_language_history_${new Date().toISOString().split('T')[0]}.pdf`;
    pdf.save(filename);
  }, [privacyMode, redactText]);

  // Export to SRT (subtitle format)
  const exportToSRT = useCallback((sessions: SessionRecord[]) => {
    let srtContent = '';
    let subtitleIndex = 1;
    
    sessions.forEach(session => {
      session.transcript.forEach(entry => {
        const startTime = new Date(entry.timestamp);
        const endTime = new Date(entry.timestamp + 3000); // 3 second duration
        
        const formatTime = (date: Date) => {
          const hours = date.getUTCHours().toString().padStart(2, '0');
          const minutes = date.getUTCMinutes().toString().padStart(2, '0');
          const seconds = date.getUTCSeconds().toString().padStart(2, '0');
          const milliseconds = date.getUTCMilliseconds().toString().padStart(3, '0');
          return `${hours}:${minutes}:${seconds},${milliseconds}`;
        };
        
        const speaker = privacyMode ? '[SPEAKER]' : entry.speaker;
        const text = redactText(entry.text);
        
        srtContent += `${subtitleIndex}\n`;
        srtContent += `${formatTime(startTime)} --> ${formatTime(endTime)}\n`;
        srtContent += `${speaker}: ${text}\n\n`;
        
        subtitleIndex++;
      });
    });
    
    const blob = new Blob([srtContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sign_language_subtitles_${new Date().toISOString().split('T')[0]}.srt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [privacyMode, redactText]);

  // Export to JSON
  const exportToJSON = useCallback((sessions: SessionRecord[]) => {
    const exportData = {
      export_info: {
        generated_at: new Date().toISOString(),
        privacy_mode: privacyMode,
        app_version: '1.0.0',
        session_count: sessions.length
      },
      sessions: sessions.map(session => ({
        ...session,
        transcript: session.transcript.map(entry => ({
          ...entry,
          speaker: privacyMode ? '[SPEAKER]' : entry.speaker,
          text: redactText(entry.text)
        }))
      }))
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sign_language_data_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [privacyMode, redactText]);

  // Handle export
  const handleExport = useCallback(async () => {
    const sessionsToExport = sessions.filter(s => selectedSessions.has(s.id));
    
    if (sessionsToExport.length === 0) {
      alert('Please select at least one session to export.');
      return;
    }
    
    setIsExporting(true);
    
    try {
      switch (exportFormat) {
        case 'pdf':
          await exportToPDF(sessionsToExport);
          break;
        case 'srt':
          exportToSRT(sessionsToExport);
          break;
        case 'json':
          exportToJSON(sessionsToExport);
          break;
      }
    } catch (error) {
      console.error('Export error:', error);
      alert('Error during export. Please try again.');
    } finally {
      setIsExporting(false);
    }
  }, [selectedSessions, sessions, exportFormat, exportToPDF, exportToSRT, exportToJSON]);

  // Format duration
  const formatDuration = (ms: number): string => {
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <div className={`bg-black/20 backdrop-blur-md rounded-lg border border-white/10 ${className}`}>
      {/* Header */}
      <div className="p-6 border-b border-white/10">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <Database className="w-6 h-6 text-green-400" />
            <h2 className="text-xl font-bold text-white">Session History & Export</h2>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setPrivacyMode(!privacyMode)}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
                privacyMode 
                  ? 'bg-green-500/20 text-green-400' 
                  : 'bg-red-500/20 text-red-400'
              }`}
            >
              {privacyMode ? <Shield className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              <span className="text-sm">
                {privacyMode ? 'Privacy On' : 'Privacy Off'}
              </span>
            </button>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search transcripts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-black/20 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
            />
          </div>
          
          <div className="flex space-x-2">
            <input
              type="date"
              value={dateRange.start}
              onChange={(e) => setDateRange(prev => ({ ...prev, start: e.target.value }))}
              className="flex-1 px-3 py-2 bg-black/20 border border-white/10 rounded-lg text-white focus:outline-none focus:border-blue-500"
            />
            <input
              type="date"
              value={dateRange.end}
              onChange={(e) => setDateRange(prev => ({ ...prev, end: e.target.value }))}
              className="flex-1 px-3 py-2 bg-black/20 border border-white/10 rounded-lg text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          
          <div className="flex space-x-2">
            <button
              onClick={selectAllSessions}
              className="px-3 py-2 text-sm text-blue-400 hover:text-white transition-colors"
            >
              Select All
            </button>
            <button
              onClick={clearSelection}
              className="px-3 py-2 text-sm text-gray-400 hover:text-white transition-colors"
            >
              Clear
            </button>
          </div>
        </div>
      </div>

      {/* Sessions List */}
      <div className="p-6">
        <div className="space-y-4 max-h-96 overflow-y-auto">
          <AnimatePresence>
            {filteredSessions.map(session => (
              <motion.div
                key={session.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className={`border rounded-lg p-4 cursor-pointer transition-all ${
                  selectedSessions.has(session.id)
                    ? 'border-blue-500 bg-blue-500/10'
                    : 'border-white/10 bg-black/20 hover:bg-white/5'
                }`}
                onClick={() => toggleSessionSelection(session.id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-4 mb-2">
                      <h3 className="text-white font-medium">
                        Session {new Date(session.start_time).toLocaleDateString()}
                      </h3>
                      <span className="px-2 py-1 text-xs bg-gray-500/20 text-gray-300 rounded">
                        {session.mode}
                      </span>
                      <span className="text-sm text-gray-400">
                        {formatDuration(session.duration)}
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-400 mb-3">
                      <div className="flex items-center space-x-1">
                        <User className="w-4 h-4" />
                        <span>{session.participant_count} participant(s)</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <FileText className="w-4 h-4" />
                        <span>{session.total_translations} translations</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Clock className="w-4 h-4" />
                        <span>{new Date(session.start_time).toLocaleTimeString()}</span>
                      </div>
                      <div>
                        <span>Confidence: {Math.round(session.average_confidence * 100)}%</span>
                      </div>
                    </div>
                    
                    {/* Preview of transcript */}
                    <div className="text-sm text-gray-300">
                      <strong>Preview:</strong>{' '}
                      {session.transcript.slice(0, 2).map((entry, index) => (
                        <span key={index}>
                          {redactText(entry.text)}
                          {index < 1 && session.transcript.length > 1 ? ' ... ' : ''}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  <div className="ml-4">
                    <input
                      type="checkbox"
                      checked={selectedSessions.has(session.id)}
                      onChange={() => toggleSessionSelection(session.id)}
                      className="w-4 h-4 text-blue-500 rounded focus:ring-blue-500"
                    />
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {filteredSessions.length === 0 && (
            <div className="text-center py-8">
              <Calendar className="w-12 h-12 text-gray-500 mx-auto mb-3" />
              <p className="text-gray-400">
                No sessions found matching your criteria.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Export Controls */}
      {selectedSessions.size > 0 && (
        <div className="p-6 border-t border-white/10">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <span className="text-white">
                {selectedSessions.size} session(s) selected
              </span>
              
              <select
                value={exportFormat}
                onChange={(e) => setExportFormat(e.target.value as any)}
                className="px-3 py-2 bg-black/20 border border-white/10 rounded text-white focus:outline-none focus:border-blue-500"
              >
                <option value="pdf">PDF Report</option>
                <option value="srt">SRT Subtitles</option>
                <option value="json">JSON Data</option>
              </select>
            </div>
            
            <button
              onClick={handleExport}
              disabled={isExporting}
              className="flex items-center space-x-2 px-4 py-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-500 text-white rounded-lg transition-colors"
            >
              <Download className="w-4 h-4" />
              <span>{isExporting ? 'Exporting...' : 'Export'}</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
