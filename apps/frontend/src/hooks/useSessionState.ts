'use client';

import { useState, useCallback, useRef, useEffect } from 'react';

interface SessionState {
  sessionId: string;
  isActive: boolean;
  startTime: number | null;
  endTime: number | null;
  mode: 'sign-to-text' | 'text-to-sign' | 'conversation';
  totalFramesProcessed: number;
  totalTranslations: number;
  averageConfidence: number;
  duration: number;
}

interface UseSessionStateReturn {
  sessionState: SessionState;
  startSession: (sessionId: string) => Promise<void>;
  endSession: () => Promise<void>;
  updateSession: (updates: Partial<SessionState>) => void;
  getSessionStats: () => SessionStats;
}

interface SessionStats {
  duration: string;
  framesPerSecond: number;
  translationsPerMinute: number;
  averageConfidence: number;
  totalTranslations: number;
}

export function useSessionState(): UseSessionStateReturn {
  const [sessionState, setSessionState] = useState<SessionState>({
    sessionId: '',
    isActive: false,
    startTime: null,
    endTime: null,
    mode: 'sign-to-text',
    totalFramesProcessed: 0,
    totalTranslations: 0,
    averageConfidence: 0,
    duration: 0
  });

  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const confidenceHistoryRef = useRef<number[]>([]);

  // Start a new session
  const startSession = useCallback(async (sessionId: string) => {
    try {
      const startTime = Date.now();
      
      // Initialize session state
      setSessionState(prev => ({
        ...prev,
        sessionId,
        isActive: true,
        startTime,
        endTime: null,
        totalFramesProcessed: 0,
        totalTranslations: 0,
        averageConfidence: 0,
        duration: 0
      }));

      // Reset confidence history
      confidenceHistoryRef.current = [];

      // Start duration tracking
      intervalRef.current = setInterval(() => {
        setSessionState(prev => ({
          ...prev,
          duration: prev.startTime ? Date.now() - prev.startTime : 0
        }));
      }, 1000);

      // Call backend to initialize session
      const response = await fetch('/api/sessions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sessionId,
          startTime,
          mode: sessionState.mode
        })
      });

      if (!response.ok) {
        throw new Error('Failed to start session');
      }

      console.log(`Session ${sessionId} started successfully`);
      
    } catch (error) {
      console.error('Error starting session:', error);
      // Reset state on error
      setSessionState(prev => ({
        ...prev,
        isActive: false,
        startTime: null
      }));
    }
  }, [sessionState.mode]);

  // End the current session
  const endSession = useCallback(async () => {
    try {
      const endTime = Date.now();
      
      // Stop duration tracking
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }

      // Update session state
      setSessionState(prev => ({
        ...prev,
        isActive: false,
        endTime,
        duration: prev.startTime ? endTime - prev.startTime : 0
      }));

      // Call backend to end session
      if (sessionState.sessionId) {
        const response = await fetch(`/api/sessions/${sessionState.sessionId}`, {
          method: 'PATCH',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            endTime,
            totalFramesProcessed: sessionState.totalFramesProcessed,
            totalTranslations: sessionState.totalTranslations,
            averageConfidence: sessionState.averageConfidence
          })
        });

        if (!response.ok) {
          console.warn('Failed to update session on backend');
        }
      }

      console.log(`Session ${sessionState.sessionId} ended successfully`);
      
    } catch (error) {
      console.error('Error ending session:', error);
    }
  }, [sessionState]);

  // Update session state
  const updateSession = useCallback((updates: Partial<SessionState>) => {
    setSessionState(prev => {
      const updated = { ...prev, ...updates };
      
      // Update confidence history and calculate average
      if (updates.averageConfidence !== undefined) {
        confidenceHistoryRef.current.push(updates.averageConfidence);
        
        // Keep only recent confidence values (last 100)
        if (confidenceHistoryRef.current.length > 100) {
          confidenceHistoryRef.current = confidenceHistoryRef.current.slice(-100);
        }
        
        // Calculate running average
        const sum = confidenceHistoryRef.current.reduce((a, b) => a + b, 0);
        updated.averageConfidence = sum / confidenceHistoryRef.current.length;
      }
      
      return updated;
    });
  }, []);

  // Get session statistics
  const getSessionStats = useCallback((): SessionStats => {
    const duration = sessionState.duration;
    const durationSeconds = duration / 1000;
    const durationMinutes = durationSeconds / 60;

    // Format duration
    const formatDuration = (ms: number): string => {
      const seconds = Math.floor(ms / 1000);
      const minutes = Math.floor(seconds / 60);
      const hours = Math.floor(minutes / 60);
      
      if (hours > 0) {
        return `${hours}:${(minutes % 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
      } else {
        return `${minutes}:${(seconds % 60).toString().padStart(2, '0')}`;
      }
    };

    return {
      duration: formatDuration(duration),
      framesPerSecond: durationSeconds > 0 ? sessionState.totalFramesProcessed / durationSeconds : 0,
      translationsPerMinute: durationMinutes > 0 ? sessionState.totalTranslations / durationMinutes : 0,
      averageConfidence: sessionState.averageConfidence,
      totalTranslations: sessionState.totalTranslations
    };
  }, [sessionState]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Auto-save session state to localStorage
  useEffect(() => {
    if (sessionState.sessionId && sessionState.isActive) {
      localStorage.setItem('currentSession', JSON.stringify({
        sessionId: sessionState.sessionId,
        startTime: sessionState.startTime,
        mode: sessionState.mode
      }));
    } else {
      localStorage.removeItem('currentSession');
    }
  }, [sessionState.sessionId, sessionState.isActive, sessionState.startTime, sessionState.mode]);

  // Restore session on page reload
  useEffect(() => {
    const savedSession = localStorage.getItem('currentSession');
    if (savedSession) {
      try {
        const parsed = JSON.parse(savedSession);
        const timeSinceStart = Date.now() - parsed.startTime;
        
        // Only restore if less than 1 hour old
        if (timeSinceStart < 3600000) {
          setSessionState(prev => ({
            ...prev,
            sessionId: parsed.sessionId,
            startTime: parsed.startTime,
            mode: parsed.mode,
            duration: timeSinceStart,
            isActive: false // Don't auto-resume active state
          }));
        } else {
          localStorage.removeItem('currentSession');
        }
      } catch (error) {
        console.error('Error restoring session:', error);
        localStorage.removeItem('currentSession');
      }
    }
  }, []);

  return {
    sessionState,
    startSession,
    endSession,
    updateSession,
    getSessionStats
  };
}
