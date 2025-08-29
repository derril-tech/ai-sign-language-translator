'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Mic, 
  MicOff, 
  Hand, 
  RotateCcw, 
  MessageCircle, 
  Volume2,
  Pause,
  Play,
  SkipForward,
  AlertCircle,
  Clock,
  Users
} from 'lucide-react';

interface ConversationControlsProps {
  isActive: boolean;
  currentSpeaker: 'user' | 'partner' | null;
  turnDuration: number;
  onTurnRequest: () => void;
  onPushToTalk: (active: boolean) => void;
  onClarifyRequest: (text: string) => void;
  onRepeatRequest: () => void;
  onPauseConversation: () => void;
  className?: string;
}

interface TurnState {
  speaker: 'user' | 'partner' | null;
  startTime: number;
  duration: number;
  isTransitioning: boolean;
}

export function ConversationControls({
  isActive,
  currentSpeaker,
  turnDuration,
  onTurnRequest,
  onPushToTalk,
  onClarifyRequest,
  onRepeatRequest,
  onPauseConversation,
  className = ''
}: ConversationControlsProps) {
  const [isPushToTalkActive, setIsPushToTalkActive] = useState(false);
  const [showClarifyDialog, setShowClarifyDialog] = useState(false);
  const [clarifyText, setClarifyText] = useState('');
  const [turnQueue, setTurnQueue] = useState<string[]>([]);
  const [conversationPaused, setConversationPaused] = useState(false);
  
  const pushToTalkTimeoutRef = useRef<NodeJS.Timeout>();
  const turnTimerRef = useRef<NodeJS.Timeout>();

  // Handle push-to-talk activation
  const handlePushToTalkStart = useCallback(() => {
    setIsPushToTalkActive(true);
    onPushToTalk(true);
    
    // Auto-release after 30 seconds
    pushToTalkTimeoutRef.current = setTimeout(() => {
      handlePushToTalkEnd();
    }, 30000);
  }, [onPushToTalk]);

  // Handle push-to-talk release
  const handlePushToTalkEnd = useCallback(() => {
    setIsPushToTalkActive(false);
    onPushToTalk(false);
    
    if (pushToTalkTimeoutRef.current) {
      clearTimeout(pushToTalkTimeoutRef.current);
    }
  }, [onPushToTalk]);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isActive) return;

      switch (event.code) {
        case 'Space':
          if (!isPushToTalkActive) {
            event.preventDefault();
            handlePushToTalkStart();
          }
          break;
        case 'KeyT':
          if (event.ctrlKey) {
            event.preventDefault();
            onTurnRequest();
          }
          break;
        case 'KeyR':
          if (event.ctrlKey) {
            event.preventDefault();
            onRepeatRequest();
          }
          break;
        case 'KeyC':
          if (event.ctrlKey) {
            event.preventDefault();
            setShowClarifyDialog(true);
          }
          break;
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.code === 'Space' && isPushToTalkActive) {
        event.preventDefault();
        handlePushToTalkEnd();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('keyup', handleKeyUp);
    };
  }, [isActive, isPushToTalkActive, handlePushToTalkStart, handlePushToTalkEnd, onTurnRequest, onRepeatRequest]);

  // Handle clarify request
  const handleClarifySubmit = useCallback(() => {
    if (clarifyText.trim()) {
      onClarifyRequest(clarifyText.trim());
      setClarifyText('');
      setShowClarifyDialog(false);
    }
  }, [clarifyText, onClarifyRequest]);

  // Format turn duration
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Get speaker indicator color
  const getSpeakerColor = (speaker: string | null) => {
    switch (speaker) {
      case 'user': return 'text-blue-400 bg-blue-500/20';
      case 'partner': return 'text-green-400 bg-green-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  return (
    <div className={`bg-black/20 backdrop-blur-md rounded-lg border border-white/10 p-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <MessageCircle className="w-5 h-5 text-purple-400" />
          <h3 className="text-white font-medium">Conversation Controls</h3>
          {isActive && (
            <motion.div
              animate={{ opacity: [1, 0.5, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="text-xs text-green-400 bg-green-500/10 px-2 py-1 rounded"
            >
              ACTIVE
            </motion.div>
          )}
        </div>

        <button
          onClick={() => {
            setConversationPaused(!conversationPaused);
            onPauseConversation();
          }}
          className={`p-2 rounded transition-colors ${
            conversationPaused 
              ? 'text-green-400 hover:text-green-300' 
              : 'text-yellow-400 hover:text-yellow-300'
          }`}
          title={conversationPaused ? 'Resume conversation' : 'Pause conversation'}
        >
          {conversationPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
        </button>
      </div>

      {/* Current Speaker & Turn Timer */}
      <div className="mb-4 p-3 bg-black/20 rounded-lg">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <Users className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-300">Current Speaker:</span>
            <span className={`text-sm px-2 py-1 rounded ${getSpeakerColor(currentSpeaker)}`}>
              {currentSpeaker ? (currentSpeaker === 'user' ? 'You' : 'Partner') : 'None'}
            </span>
          </div>
          
          {currentSpeaker && (
            <div className="flex items-center space-x-2 text-sm text-gray-400">
              <Clock className="w-4 h-4" />
              <span>{formatDuration(turnDuration)}</span>
            </div>
          )}
        </div>

        {/* Turn Progress Bar */}
        {currentSpeaker && (
          <div className="w-full bg-gray-700 rounded-full h-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min((turnDuration / 120) * 100, 100)}%` }}
              className={`h-2 rounded-full ${
                turnDuration > 90 ? 'bg-red-500' : turnDuration > 60 ? 'bg-yellow-500' : 'bg-green-500'
              }`}
            />
          </div>
        )}
      </div>

      {/* Main Controls */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        
        {/* Push-to-Talk */}
        <motion.button
          onMouseDown={handlePushToTalkStart}
          onMouseUp={handlePushToTalkEnd}
          onMouseLeave={handlePushToTalkEnd}
          whilePressed={{ scale: 0.95 }}
          className={`
            relative p-4 rounded-lg border-2 transition-all duration-200 flex flex-col items-center space-y-2
            ${isPushToTalkActive
              ? 'border-red-500 bg-red-500/20 text-red-400'
              : 'border-blue-500/30 bg-blue-500/10 text-blue-400 hover:bg-blue-500/20'
            }
          `}
          disabled={!isActive || conversationPaused}
        >
          {isPushToTalkActive ? (
            <MicOff className="w-6 h-6" />
          ) : (
            <Mic className="w-6 h-6" />
          )}
          <span className="text-xs font-medium">
            {isPushToTalkActive ? 'Release to Stop' : 'Push to Talk'}
          </span>
          <span className="text-xs text-gray-400">Space</span>
          
          {isPushToTalkActive && (
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
              className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"
            />
          )}
        </motion.button>

        {/* Request Turn */}
        <button
          onClick={onTurnRequest}
          className="p-4 rounded-lg border-2 border-green-500/30 bg-green-500/10 text-green-400 hover:bg-green-500/20 transition-all duration-200 flex flex-col items-center space-y-2"
          disabled={!isActive || conversationPaused || currentSpeaker === 'user'}
        >
          <Hand className="w-6 h-6" />
          <span className="text-xs font-medium">Request Turn</span>
          <span className="text-xs text-gray-400">Ctrl+T</span>
        </button>

        {/* Clarify */}
        <button
          onClick={() => setShowClarifyDialog(true)}
          className="p-4 rounded-lg border-2 border-yellow-500/30 bg-yellow-500/10 text-yellow-400 hover:bg-yellow-500/20 transition-all duration-200 flex flex-col items-center space-y-2"
          disabled={!isActive || conversationPaused}
        >
          <AlertCircle className="w-6 h-6" />
          <span className="text-xs font-medium">Ask to Clarify</span>
          <span className="text-xs text-gray-400">Ctrl+C</span>
        </button>

        {/* Repeat */}
        <button
          onClick={onRepeatRequest}
          className="p-4 rounded-lg border-2 border-purple-500/30 bg-purple-500/10 text-purple-400 hover:bg-purple-500/20 transition-all duration-200 flex flex-col items-center space-y-2"
          disabled={!isActive || conversationPaused}
        >
          <RotateCcw className="w-6 h-6" />
          <span className="text-xs font-medium">Repeat Last</span>
          <span className="text-xs text-gray-400">Ctrl+R</span>
        </button>
      </div>

      {/* Turn Queue */}
      {turnQueue.length > 0 && (
        <div className="mb-4 p-3 bg-orange-500/10 border border-orange-500/20 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <SkipForward className="w-4 h-4 text-orange-400" />
            <span className="text-sm text-orange-400 font-medium">Turn Queue</span>
          </div>
          <div className="space-y-1">
            {turnQueue.map((speaker, index) => (
              <div key={index} className="text-xs text-gray-300">
                {index + 1}. {speaker}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Status Messages */}
      <div className="space-y-2">
        {conversationPaused && (
          <div className="p-2 bg-yellow-500/10 border border-yellow-500/20 rounded text-yellow-400 text-sm">
            Conversation paused. Click resume to continue.
          </div>
        )}
        
        {!isActive && (
          <div className="p-2 bg-gray-500/10 border border-gray-500/20 rounded text-gray-400 text-sm">
            Start a session to enable conversation controls.
          </div>
        )}

        {currentSpeaker === 'partner' && (
          <div className="p-2 bg-blue-500/10 border border-blue-500/20 rounded text-blue-400 text-sm">
            Partner is speaking. Use "Request Turn" when ready to respond.
          </div>
        )}
      </div>

      {/* Keyboard Shortcuts Help */}
      <details className="mt-4">
        <summary className="text-sm text-gray-400 cursor-pointer hover:text-white">
          Keyboard Shortcuts
        </summary>
        <div className="mt-2 space-y-1 text-xs text-gray-500">
          <div><kbd className="bg-gray-700 px-1 rounded">Space</kbd> - Push to talk</div>
          <div><kbd className="bg-gray-700 px-1 rounded">Ctrl+T</kbd> - Request turn</div>
          <div><kbd className="bg-gray-700 px-1 rounded">Ctrl+R</kbd> - Repeat last</div>
          <div><kbd className="bg-gray-700 px-1 rounded">Ctrl+C</kbd> - Ask to clarify</div>
        </div>
      </details>

      {/* Clarify Dialog */}
      <AnimatePresence>
        {showClarifyDialog && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="bg-gray-800 rounded-lg border border-white/10 p-6 w-full max-w-md mx-4"
            >
              <h3 className="text-lg font-medium text-white mb-4">Ask for Clarification</h3>
              
              <textarea
                value={clarifyText}
                onChange={(e) => setClarifyText(e.target.value)}
                placeholder="What would you like clarified? (e.g., 'Could you repeat the part about...')"
                className="w-full h-24 px-3 py-2 bg-black/20 border border-white/10 rounded text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 resize-none"
                autoFocus
              />
              
              <div className="flex items-center justify-end space-x-3 mt-4">
                <button
                  onClick={() => {
                    setShowClarifyDialog(false);
                    setClarifyText('');
                  }}
                  className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
                >
                  Cancel
                </button>
                
                <button
                  onClick={handleClarifySubmit}
                  disabled={!clarifyText.trim()}
                  className="px-4 py-2 bg-yellow-500 hover:bg-yellow-600 disabled:bg-gray-500 text-white rounded transition-colors"
                >
                  Send Request
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
