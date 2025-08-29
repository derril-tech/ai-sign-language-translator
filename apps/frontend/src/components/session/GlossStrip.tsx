'use client';

import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Eye, EyeOff } from 'lucide-react';

interface GlossToken {
  id: string;
  token: string;
  confidence: number;
  timestamp: number;
  isActive: boolean;
}

interface GlossStripProps {
  glossSequence: string[];
  isActive: boolean;
  className?: string;
}

export function GlossStrip({
  glossSequence,
  isActive,
  className = ''
}: GlossStripProps) {
  const [glossTokens, setGlossTokens] = useState<GlossToken[]>([]);
  const [isVisible, setIsVisible] = useState(true);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Convert gloss sequence to tokens
  useEffect(() => {
    if (glossSequence && glossSequence.length > 0) {
      const newTokens = glossSequence.map((token, index) => ({
        id: `gloss_${Date.now()}_${index}`,
        token: token.toUpperCase(),
        confidence: 0.8 + Math.random() * 0.2, // Mock confidence
        timestamp: Date.now() + index * 100,
        isActive: index === glossSequence.length - 1
      }));

      setGlossTokens(prev => {
        // Merge with existing tokens, keeping last 15
        const combined = [...prev, ...newTokens];
        return combined.slice(-15);
      });
    }
  }, [glossSequence]);

  // Auto-scroll to show latest tokens
  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollLeft = scrollContainerRef.current.scrollWidth;
    }
  }, [glossTokens]);

  // Mark tokens as inactive after a delay
  useEffect(() => {
    const timer = setTimeout(() => {
      setGlossTokens(prev => 
        prev.map(token => ({ ...token, isActive: false }))
      );
    }, 1000);

    return () => clearTimeout(timer);
  }, [glossTokens]);

  // Get confidence color
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'border-green-400 text-green-300';
    if (confidence >= 0.6) return 'border-yellow-400 text-yellow-300';
    return 'border-red-400 text-red-300';
  };

  // Get confidence background
  const getConfidenceBackground = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-500/10';
    if (confidence >= 0.6) return 'bg-yellow-500/10';
    return 'bg-red-500/10';
  };

  if (!isVisible) {
    return (
      <div className={`bg-black/20 backdrop-blur-md rounded-lg border border-white/10 p-4 ${className}`}>
        <div className="flex items-center justify-between">
          <h3 className="text-white font-medium">Gloss Sequence</h3>
          <button
            onClick={() => setIsVisible(true)}
            className="p-1 text-gray-400 hover:text-white transition-colors"
            title="Show gloss sequence"
          >
            <Eye className="w-4 h-4" />
          </button>
        </div>
        <p className="text-gray-400 text-sm mt-2">Gloss sequence hidden</p>
      </div>
    );
  }

  return (
    <div className={`bg-black/20 backdrop-blur-md rounded-lg border border-white/10 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/10">
        <div className="flex items-center space-x-3">
          <h3 className="text-white font-medium">Gloss Sequence</h3>
          {isActive && (
            <motion.div
              animate={{ opacity: [1, 0.5, 1] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="text-xs text-blue-400 bg-blue-500/10 px-2 py-1 rounded"
            >
              DETECTING
            </motion.div>
          )}
        </div>

        <button
          onClick={() => setIsVisible(false)}
          className="p-1 text-gray-400 hover:text-white transition-colors"
          title="Hide gloss sequence"
        >
          <EyeOff className="w-4 h-4" />
        </button>
      </div>

      {/* Gloss Tokens */}
      <div className="p-4">
        <div
          ref={scrollContainerRef}
          className="flex space-x-2 overflow-x-auto scrollbar-hide pb-2"
          style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
        >
          <AnimatePresence>
            {glossTokens.map((token) => (
              <motion.div
                key={token.id}
                initial={{ opacity: 0, scale: 0.8, y: 20 }}
                animate={{ 
                  opacity: 1, 
                  scale: token.isActive ? 1.1 : 1, 
                  y: 0 
                }}
                exit={{ opacity: 0, scale: 0.8, y: -20 }}
                transition={{ 
                  type: "spring", 
                  stiffness: 300, 
                  damping: 25 
                }}
                className={`
                  relative flex-shrink-0 px-3 py-2 rounded-lg border-2 transition-all duration-300
                  ${getConfidenceColor(token.confidence)}
                  ${getConfidenceBackground(token.confidence)}
                  ${token.isActive ? 'shadow-lg shadow-blue-500/20' : ''}
                `}
              >
                {/* Token Text */}
                <span className="text-sm font-mono font-medium whitespace-nowrap">
                  {token.token}
                </span>

                {/* Confidence Badge */}
                <div className="absolute -top-1 -right-1 bg-black/50 text-white text-xs px-1 rounded">
                  {Math.round(token.confidence * 100)}
                </div>

                {/* Active Indicator */}
                {token.isActive && (
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 0.6, repeat: Infinity }}
                    className="absolute -top-1 -left-1 w-2 h-2 bg-blue-400 rounded-full"
                  />
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Loading Indicator */}
          {isActive && glossTokens.length === 0 && (
            <motion.div
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="flex-shrink-0 px-4 py-2 bg-gray-500/10 border-2 border-dashed border-gray-500/30 rounded-lg"
            >
              <span className="text-sm text-gray-400">Analyzing...</span>
            </motion.div>
          )}
        </div>

        {/* Empty State */}
        {!isActive && glossTokens.length === 0 && (
          <div className="text-center py-4">
            <div className="text-gray-500 text-sm">
              Start signing to see gloss tokens appear here
            </div>
          </div>
        )}

        {/* Info Text */}
        <div className="mt-3 text-xs text-gray-400">
          <p>
            Gloss tokens represent the linguistic units of sign language. 
            Colors indicate confidence: 
            <span className="text-green-300 ml-1">High</span>,
            <span className="text-yellow-300 ml-1">Medium</span>,
            <span className="text-red-300 ml-1">Low</span>
          </p>
        </div>
      </div>
    </div>
  );
}
