'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CameraStage } from '@/components/session/CameraStage';
import { CaptionRail } from '@/components/session/CaptionRail';
import { GlossStrip } from '@/components/session/GlossStrip';
import { TTSDock } from '@/components/session/TTSDock';
import { AvatarPanel } from '@/components/session/AvatarPanel';
import { ConfidenceBar } from '@/components/session/ConfidenceBar';
import { SessionControls } from '@/components/session/SessionControls';
import { useWebRTC } from '@/hooks/useWebRTC';
import { useTranslationPipeline } from '@/hooks/useTranslationPipeline';
import { useSessionState } from '@/hooks/useSessionState';

interface SessionData {
  sessionId: string;
  isActive: boolean;
  mode: 'sign-to-text' | 'text-to-sign' | 'conversation';
  confidence: number;
  currentText: string;
  partialText: string;
  glossSequence: string[];
  audioEnabled: boolean;
  videoEnabled: boolean;
}

export default function SessionPage() {
  const [sessionData, setSessionData] = useState<SessionData>({
    sessionId: '',
    isActive: false,
    mode: 'sign-to-text',
    confidence: 0,
    currentText: '',
    partialText: '',
    glossSequence: [],
    audioEnabled: true,
    videoEnabled: true,
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Custom hooks for WebRTC and translation pipeline
  const { 
    stream, 
    isConnected, 
    startCamera, 
    stopCamera, 
    captureFrame 
  } = useWebRTC();

  const {
    processFrame,
    translationResult,
    isProcessing,
    confidence
  } = useTranslationPipeline(sessionData.sessionId);

  const {
    sessionState,
    updateSession,
    startSession,
    endSession
  } = useSessionState();

  // Initialize session
  useEffect(() => {
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionData(prev => ({ ...prev, sessionId: newSessionId }));
  }, []);

  // Handle session start/stop
  const handleSessionToggle = async () => {
    if (sessionData.isActive) {
      await endSession();
      await stopCamera();
      setSessionData(prev => ({ ...prev, isActive: false }));
    } else {
      await startCamera();
      await startSession(sessionData.sessionId);
      setSessionData(prev => ({ ...prev, isActive: true }));
    }
  };

  // Handle mode change
  const handleModeChange = (mode: SessionData['mode']) => {
    setSessionData(prev => ({ ...prev, mode }));
  };

  // Process video frames
  useEffect(() => {
    if (!sessionData.isActive || !stream || !videoRef.current) return;

    const processVideoFrame = async () => {
      if (videoRef.current && canvasRef.current) {
        const frame = captureFrame(videoRef.current, canvasRef.current);
        if (frame) {
          await processFrame(frame);
        }
      }
    };

    const interval = setInterval(processVideoFrame, 33); // ~30 FPS
    return () => clearInterval(interval);
  }, [sessionData.isActive, stream, captureFrame, processFrame]);

  // Update session data from translation results
  useEffect(() => {
    if (translationResult) {
      setSessionData(prev => ({
        ...prev,
        currentText: translationResult.finalText || prev.currentText,
        partialText: translationResult.partialText || '',
        glossSequence: translationResult.glossSequence || [],
        confidence: confidence
      }));
    }
  }, [translationResult, confidence]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="bg-black/20 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <h1 className="text-xl font-bold text-white">
                AI Sign Language Translator
              </h1>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  sessionData.isActive ? 'bg-green-400' : 'bg-red-400'
                }`} />
                <span className="text-sm text-gray-300">
                  {sessionData.isActive ? 'Live' : 'Offline'}
                </span>
              </div>
            </div>
            
            <SessionControls
              isActive={sessionData.isActive}
              mode={sessionData.mode}
              onToggleSession={handleSessionToggle}
              onModeChange={handleModeChange}
              audioEnabled={sessionData.audioEnabled}
              videoEnabled={sessionData.videoEnabled}
              onAudioToggle={(enabled) => 
                setSessionData(prev => ({ ...prev, audioEnabled: enabled }))
              }
              onVideoToggle={(enabled) => 
                setSessionData(prev => ({ ...prev, videoEnabled: enabled }))
              }
            />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-8rem)]">
          
          {/* Left Column - Camera and Controls */}
          <div className="lg:col-span-2 space-y-4">
            
            {/* Camera Stage */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="relative"
            >
              <CameraStage
                ref={videoRef}
                canvasRef={canvasRef}
                stream={stream}
                isActive={sessionData.isActive}
                showPoseOverlay={true}
                onFrameCapture={captureFrame}
              />
              
              {/* Confidence Overlay */}
              <div className="absolute top-4 right-4">
                <ConfidenceBar 
                  confidence={sessionData.confidence}
                  isProcessing={isProcessing}
                />
              </div>
            </motion.div>

            {/* Caption Rail */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <CaptionRail
                currentText={sessionData.currentText}
                partialText={sessionData.partialText}
                confidence={sessionData.confidence}
                isLive={sessionData.isActive}
              />
            </motion.div>

            {/* Gloss Strip */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <GlossStrip
                glossSequence={sessionData.glossSequence}
                isActive={sessionData.isActive}
              />
            </motion.div>
          </div>

          {/* Right Column - Avatar and TTS */}
          <div className="space-y-4">
            
            {/* Avatar Panel */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="h-2/3"
            >
              <AvatarPanel
                mode={sessionData.mode}
                text={sessionData.currentText}
                isActive={sessionData.isActive}
                className="h-full"
              />
            </motion.div>

            {/* TTS Dock */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="h-1/3"
            >
              <TTSDock
                text={sessionData.currentText}
                isEnabled={sessionData.audioEnabled}
                onSpeak={(text) => console.log('Speaking:', text)}
                onVoiceChange={(voiceId) => console.log('Voice changed:', voiceId)}
              />
            </motion.div>
          </div>
        </div>
      </main>

      {/* Hidden canvas for frame processing */}
      <canvas
        ref={canvasRef}
        className="hidden"
        width={640}
        height={480}
      />

      {/* Loading overlay */}
      <AnimatePresence>
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
          >
            <div className="bg-white/10 backdrop-blur-md rounded-lg p-6 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-4"></div>
              <p className="text-white">Processing translation...</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
