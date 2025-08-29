'use client';

import { useState, useRef, useCallback, useEffect } from 'react';

interface TranslationResult {
  finalText: string;
  partialText: string;
  glossSequence: string[];
  confidence: number;
  timestamp: number;
  processingTime: number;
}

interface UseTranslationPipelineReturn {
  translationResult: TranslationResult | null;
  isProcessing: boolean;
  confidence: number;
  processFrame: (frameData: string) => Promise<void>;
  resetPipeline: () => void;
}

export function useTranslationPipeline(sessionId: string): UseTranslationPipelineReturn {
  const [translationResult, setTranslationResult] = useState<TranslationResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [confidence, setConfidence] = useState(0);
  
  const processingRef = useRef(false);
  const frameQueueRef = useRef<string[]>([]);
  const lastProcessTimeRef = useRef(0);

  // Mock API endpoints (in production, these would be actual backend endpoints)
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001/api/v1';

  // Process a single frame through the translation pipeline
  const processFrame = useCallback(async (frameData: string) => {
    if (!frameData || processingRef.current) {
      return;
    }

    // Throttle processing to avoid overwhelming the backend
    const now = Date.now();
    if (now - lastProcessTimeRef.current < 100) { // Max 10 FPS processing
      return;
    }
    lastProcessTimeRef.current = now;

    processingRef.current = true;
    setIsProcessing(true);

    try {
      // Step 1: Send frame to pose worker
      const poseResponse = await fetch(`${API_BASE}/pose/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          frame_data: frameData,
          timestamp: now / 1000,
          frame_id: `frame_${now}`
        })
      });

      if (!poseResponse.ok) {
        throw new Error('Pose processing failed');
      }

      // For now, simulate the full pipeline with mock data
      // In production, this would be handled by the NATS message bus
      await simulateTranslationPipeline(sessionId, frameData, now);

    } catch (error) {
      console.error('Translation pipeline error:', error);
    } finally {
      processingRef.current = false;
      setIsProcessing(false);
    }
  }, [sessionId, API_BASE]);

  // Simulate the full translation pipeline (for development)
  const simulateTranslationPipeline = async (
    sessionId: string, 
    frameData: string, 
    timestamp: number
  ) => {
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 200 + Math.random() * 300));

    // Mock translation results
    const mockGlossSequences = [
      ['HELLO', 'HOW', 'YOU'],
      ['THANK', 'YOU', 'VERY', 'MUCH'],
      ['I', 'UNDERSTAND', 'YOU'],
      ['PLEASE', 'REPEAT', 'THAT'],
      ['GOOD', 'MORNING', 'NICE', 'MEET', 'YOU'],
      ['I', 'NEED', 'HELP', 'WITH', 'THIS'],
      ['CAN', 'YOU', 'EXPLAIN', 'MORE'],
      ['YES', 'I', 'AGREE', 'WITH', 'YOU']
    ];

    const mockTexts = [
      'Hello, how are you?',
      'Thank you very much.',
      'I understand you.',
      'Please repeat that.',
      'Good morning, nice to meet you.',
      'I need help with this.',
      'Can you explain more?',
      'Yes, I agree with you.'
    ];

    const randomIndex = Math.floor(Math.random() * mockTexts.length);
    const mockConfidence = 0.6 + Math.random() * 0.3; // 60-90% confidence

    // Simulate partial text first
    const fullText = mockTexts[randomIndex];
    const partialLength = Math.floor(fullText.length * 0.7);
    const partialText = fullText.substring(0, partialLength) + '...';

    // Update with partial result
    setTranslationResult({
      finalText: '',
      partialText: partialText,
      glossSequence: mockGlossSequences[randomIndex].slice(0, -1),
      confidence: mockConfidence * 0.8,
      timestamp: timestamp,
      processingTime: 200 + Math.random() * 100
    });
    setConfidence(mockConfidence * 0.8);

    // Simulate additional processing time for final result
    await new Promise(resolve => setTimeout(resolve, 300 + Math.random() * 200));

    // Update with final result
    setTranslationResult({
      finalText: fullText,
      partialText: '',
      glossSequence: mockGlossSequences[randomIndex],
      confidence: mockConfidence,
      timestamp: timestamp,
      processingTime: 500 + Math.random() * 200
    });
    setConfidence(mockConfidence);
  };

  // Reset the pipeline state
  const resetPipeline = useCallback(() => {
    setTranslationResult(null);
    setIsProcessing(false);
    setConfidence(0);
    processingRef.current = false;
    frameQueueRef.current = [];
    lastProcessTimeRef.current = 0;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      processingRef.current = false;
    };
  }, []);

  return {
    translationResult,
    isProcessing,
    confidence,
    processFrame,
    resetPipeline
  };
}
