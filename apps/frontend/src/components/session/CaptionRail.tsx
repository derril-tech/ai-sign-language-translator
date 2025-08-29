'use client';

import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Volume2, VolumeX, Copy, Download } from 'lucide-react';

interface CaptionRailProps {
  currentText: string;
  partialText: string;
  confidence: number;
  isLive: boolean;
  className?: string;
}

interface CaptionSegment {
  id: string;
  text: string;
  timestamp: number;
  confidence: number;
  isPartial: boolean;
}

export function CaptionRail({
  currentText,
  partialText,
  confidence,
  isLive,
  className = ''
}: CaptionRailProps) {
  const [segments, setSegments] = useState<CaptionSegment[]>([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Add new segments when text changes
  useEffect(() => {
    if (currentText && currentText.trim()) {
      const newSegment: CaptionSegment = {
        id: `segment_${Date.now()}`,
        text: currentText,
        timestamp: Date.now(),
        confidence,
        isPartial: false
      };

      setSegments(prev => {
        // Remove any existing partial segments and add the new committed segment
        const withoutPartials = prev.filter(s => !s.isPartial);
        return [...withoutPartials, newSegment].slice(-20); // Keep last 20 segments
      });
    }
  }, [currentText, confidence]);

  // Handle partial text
  useEffect(() => {
    if (partialText && partialText.trim()) {
      const partialSegment: CaptionSegment = {
        id: 'partial_current',
        text: partialText,
        timestamp: Date.now(),
        confidence,
        isPartial: true
      };

      setSegments(prev => {
        // Remove existing partial and add new one
        const withoutPartials = prev.filter(s => !s.isPartial);
        return [...withoutPartials, partialSegment];
      });
    } else {
      // Remove partial segments when no partial text
      setSegments(prev => prev.filter(s => !s.isPartial));
    }
  }, [partialText, confidence]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [segments, autoScroll]);

  // Handle manual scroll
  const handleScroll = () => {
    if (scrollContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
      const isAtBottom = scrollTop + clientHeight >= scrollHeight - 10;
      setAutoScroll(isAtBottom);
    }
  };

  // Get confidence color
  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.8) return 'text-green-400';
    if (conf >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  // Get confidence background
  const getConfidenceBackground = (conf: number) => {
    if (conf >= 0.8) return 'bg-green-500/10 border-green-500/20';
    if (conf >= 0.6) return 'bg-yellow-500/10 border-yellow-500/20';
    return 'bg-red-500/10 border-red-500/20';
  };

  // Copy all text
  const copyAllText = () => {
    const allText = segments
      .filter(s => !s.isPartial)
      .map(s => s.text)
      .join(' ');
    navigator.clipboard.writeText(allText);
  };

  // Export transcript
  const exportTranscript = () => {
    const transcript = segments
      .filter(s => !s.isPartial)
      .map(s => ({
        timestamp: new Date(s.timestamp).toISOString(),
        text: s.text,
        confidence: s.confidence
      }));

    const blob = new Blob([JSON.stringify(transcript, null, 2)], {
      type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcript_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className={`bg-black/20 backdrop-blur-md rounded-lg border border-white/10 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/10">
        <div className="flex items-center space-x-3">
          <h3 className="text-white font-medium">Live Captions</h3>
          {isLive && (
            <motion.div
              animate={{ opacity: [1, 0.5, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="flex items-center space-x-1"
            >
              <div className="w-2 h-2 bg-red-500 rounded-full"></div>
              <span className="text-red-400 text-sm">LIVE</span>
            </motion.div>
          )}
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={copyAllText}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title="Copy transcript"
          >
            <Copy className="w-4 h-4" />
          </button>
          <button
            onClick={exportTranscript}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title="Export transcript"
          >
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Caption Content */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="h-32 overflow-y-auto p-4 space-y-2"
      >
        <AnimatePresence>
          {segments.map((segment) => (
            <motion.div
              key={segment.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={`p-3 rounded-lg border ${
                segment.isPartial 
                  ? 'bg-blue-500/10 border-blue-500/20 border-dashed' 
                  : getConfidenceBackground(segment.confidence)
              }`}
            >
              <div className="flex items-start justify-between">
                <p className={`text-sm ${
                  segment.isPartial ? 'text-blue-300 italic' : 'text-white'
                }`}>
                  {segment.text}
                  {segment.isPartial && (
                    <motion.span
                      animate={{ opacity: [1, 0] }}
                      transition={{ duration: 1, repeat: Infinity }}
                      className="ml-1"
                    >
                      |
                    </motion.span>
                  )}
                </p>
                
                <div className="flex items-center space-x-2 ml-3">
                  {!segment.isPartial && (
                    <span className={`text-xs ${getConfidenceColor(segment.confidence)}`}>
                      {Math.round(segment.confidence * 100)}%
                    </span>
                  )}
                  <span className="text-xs text-gray-500">
                    {new Date(segment.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Empty State */}
        {segments.length === 0 && (
          <div className="text-center py-8">
            <Volume2 className="w-8 h-8 text-gray-500 mx-auto mb-2" />
            <p className="text-gray-400 text-sm">
              {isLive ? 'Listening for sign language...' : 'Start session to see captions'}
            </p>
          </div>
        )}
      </div>

      {/* Auto-scroll indicator */}
      {!autoScroll && (
        <motion.button
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          onClick={() => {
            setAutoScroll(true);
            if (scrollContainerRef.current) {
              scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
            }
          }}
          className="absolute bottom-2 right-2 bg-blue-500 hover:bg-blue-600 text-white text-xs px-2 py-1 rounded-full transition-colors"
        >
          ↓ New captions
        </motion.button>
      )}
    </div>
  );
}
