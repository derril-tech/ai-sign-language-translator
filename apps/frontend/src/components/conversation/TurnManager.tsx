'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Users, 
  Clock, 
  ArrowRight, 
  Pause, 
  Play,
  AlertTriangle,
  CheckCircle,
  XCircle
} from 'lucide-react';

interface Participant {
  id: string;
  name: string;
  role: 'user' | 'interpreter' | 'client';
  isActive: boolean;
  totalSpeakTime: number;
  turnCount: number;
  lastActivity: number;
}

interface Turn {
  id: string;
  participantId: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  wordCount: number;
  confidence: number;
  interrupted: boolean;
}

interface TurnManagerProps {
  participants: Participant[];
  currentTurn: Turn | null;
  turnHistory: Turn[];
  maxTurnDuration: number;
  onTurnChange: (participantId: string) => void;
  onTurnEnd: () => void;
  onTurnRequest: (participantId: string) => void;
  className?: string;
}

export function TurnManager({
  participants,
  currentTurn,
  turnHistory,
  maxTurnDuration = 120, // 2 minutes default
  onTurnChange,
  onTurnEnd,
  onTurnRequest,
  className = ''
}: TurnManagerProps) {
  const [turnQueue, setTurnQueue] = useState<string[]>([]);
  const [autoTurnEnabled, setAutoTurnEnabled] = useState(true);
  const [turnWarningShown, setTurnWarningShown] = useState(false);
  const [conversationStats, setConversationStats] = useState({
    totalDuration: 0,
    totalTurns: 0,
    averageTurnLength: 0,
    participationBalance: 0
  });

  const turnTimerRef = useRef<NodeJS.Timeout>();
  const warningTimerRef = useRef<NodeJS.Timeout>();

  // Calculate current turn duration
  const currentTurnDuration = currentTurn 
    ? Math.floor((Date.now() - currentTurn.startTime) / 1000)
    : 0;

  // Update conversation statistics
  useEffect(() => {
    const totalDuration = turnHistory.reduce((sum, turn) => sum + (turn.duration || 0), 0);
    const totalTurns = turnHistory.length;
    const averageTurnLength = totalTurns > 0 ? totalDuration / totalTurns : 0;
    
    // Calculate participation balance (0 = perfectly balanced, 1 = completely unbalanced)
    const participantTimes = participants.map(p => p.totalSpeakTime);
    const maxTime = Math.max(...participantTimes);
    const minTime = Math.min(...participantTimes);
    const participationBalance = maxTime > 0 ? (maxTime - minTime) / maxTime : 0;

    setConversationStats({
      totalDuration,
      totalTurns,
      averageTurnLength,
      participationBalance
    });
  }, [turnHistory, participants]);

  // Handle turn timeout
  useEffect(() => {
    if (currentTurn && autoTurnEnabled) {
      // Set warning timer at 80% of max duration
      const warningTime = maxTurnDuration * 0.8 * 1000;
      warningTimerRef.current = setTimeout(() => {
        setTurnWarningShown(true);
      }, warningTime);

      // Set auto-end timer at max duration
      turnTimerRef.current = setTimeout(() => {
        handleAutoTurnEnd();
      }, maxTurnDuration * 1000);

      return () => {
        if (turnTimerRef.current) clearTimeout(turnTimerRef.current);
        if (warningTimerRef.current) clearTimeout(warningTimerRef.current);
        setTurnWarningShown(false);
      };
    }
  }, [currentTurn, maxTurnDuration, autoTurnEnabled]);

  // Auto-end turn when time limit reached
  const handleAutoTurnEnd = useCallback(() => {
    if (currentTurn) {
      onTurnEnd();
      
      // Start next turn if someone is in queue
      if (turnQueue.length > 0) {
        const nextParticipant = turnQueue[0];
        setTurnQueue(prev => prev.slice(1));
        setTimeout(() => onTurnChange(nextParticipant), 500);
      }
    }
  }, [currentTurn, turnQueue, onTurnEnd, onTurnChange]);

  // Add participant to turn queue
  const addToQueue = useCallback((participantId: string) => {
    setTurnQueue(prev => {
      if (!prev.includes(participantId) && participantId !== currentTurn?.participantId) {
        return [...prev, participantId];
      }
      return prev;
    });
    onTurnRequest(participantId);
  }, [currentTurn, onTurnRequest]);

  // Remove participant from queue
  const removeFromQueue = useCallback((participantId: string) => {
    setTurnQueue(prev => prev.filter(id => id !== participantId));
  }, []);

  // Manually end current turn
  const handleManualTurnEnd = useCallback(() => {
    onTurnEnd();
    setTurnWarningShown(false);
  }, [onTurnEnd]);

  // Start next turn from queue
  const startNextTurn = useCallback(() => {
    if (turnQueue.length > 0) {
      const nextParticipant = turnQueue[0];
      setTurnQueue(prev => prev.slice(1));
      onTurnChange(nextParticipant);
    }
  }, [turnQueue, onTurnChange]);

  // Get participant by ID
  const getParticipant = useCallback((id: string) => {
    return participants.find(p => p.id === id);
  }, [participants]);

  // Format duration
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Get turn progress percentage
  const getTurnProgress = (): number => {
    return Math.min((currentTurnDuration / maxTurnDuration) * 100, 100);
  };

  // Get progress color based on time remaining
  const getProgressColor = (): string => {
    const progress = getTurnProgress();
    if (progress >= 90) return 'bg-red-500';
    if (progress >= 80) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  return (
    <div className={`bg-black/20 backdrop-blur-md rounded-lg border border-white/10 p-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Users className="w-5 h-5 text-blue-400" />
          <h3 className="text-white font-medium">Turn Management</h3>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setAutoTurnEnabled(!autoTurnEnabled)}
            className={`text-xs px-2 py-1 rounded transition-colors ${
              autoTurnEnabled 
                ? 'bg-green-500/20 text-green-400' 
                : 'bg-gray-500/20 text-gray-400'
            }`}
          >
            Auto-turn: {autoTurnEnabled ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>

      {/* Current Turn Display */}
      {currentTurn && (
        <div className="mb-4 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse" />
              <span className="text-white font-medium">
                {getParticipant(currentTurn.participantId)?.name || 'Unknown'} is speaking
              </span>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1 text-sm text-gray-300">
                <Clock className="w-4 h-4" />
                <span>{formatDuration(currentTurnDuration)}</span>
                <span className="text-gray-500">/ {formatDuration(maxTurnDuration)}</span>
              </div>
              
              <button
                onClick={handleManualTurnEnd}
                className="p-1 text-red-400 hover:text-red-300 transition-colors"
                title="End turn"
              >
                <XCircle className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Turn Progress Bar */}
          <div className="relative w-full bg-gray-700 rounded-full h-3 mb-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${getTurnProgress()}%` }}
              className={`h-3 rounded-full ${getProgressColor()} transition-colors`}
            />
            
            {/* Warning indicator */}
            {turnWarningShown && (
              <motion.div
                animate={{ opacity: [1, 0.5, 1] }}
                transition={{ duration: 1, repeat: Infinity }}
                className="absolute right-2 top-1/2 transform -translate-y-1/2"
              >
                <AlertTriangle className="w-4 h-4 text-yellow-400" />
              </motion.div>
            )}
          </div>

          {/* Turn Stats */}
          <div className="grid grid-cols-3 gap-4 text-xs text-gray-400">
            <div>
              <span className="block text-white">{currentTurn.wordCount || 0}</span>
              <span>Words</span>
            </div>
            <div>
              <span className="block text-white">{Math.round(currentTurn.confidence * 100)}%</span>
              <span>Confidence</span>
            </div>
            <div>
              <span className="block text-white">
                {Math.round(((currentTurn.wordCount || 0) / Math.max(currentTurnDuration / 60, 1)))}
              </span>
              <span>WPM</span>
            </div>
          </div>
        </div>
      )}

      {/* Turn Queue */}
      {turnQueue.length > 0 && (
        <div className="mb-4 p-3 bg-orange-500/10 border border-orange-500/20 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-orange-400 font-medium">Turn Queue</span>
            {!currentTurn && (
              <button
                onClick={startNextTurn}
                className="text-xs text-orange-400 hover:text-orange-300 transition-colors"
              >
                Start Next
              </button>
            )}
          </div>
          
          <div className="space-y-2">
            {turnQueue.map((participantId, index) => {
              const participant = getParticipant(participantId);
              return (
                <div key={participantId} className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-400">{index + 1}.</span>
                    <span className="text-sm text-white">{participant?.name || 'Unknown'}</span>
                  </div>
                  
                  <button
                    onClick={() => removeFromQueue(participantId)}
                    className="text-xs text-red-400 hover:text-red-300 transition-colors"
                  >
                    Remove
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Participants */}
      <div className="mb-4">
        <h4 className="text-sm text-gray-300 font-medium mb-2">Participants</h4>
        <div className="space-y-2">
          {participants.map(participant => (
            <div
              key={participant.id}
              className={`p-3 rounded-lg border transition-all ${
                currentTurn?.participantId === participant.id
                  ? 'border-blue-500 bg-blue-500/10'
                  : 'border-white/10 bg-black/20'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`w-2 h-2 rounded-full ${
                    participant.isActive ? 'bg-green-500' : 'bg-gray-500'
                  }`} />
                  <span className="text-white font-medium">{participant.name}</span>
                  <span className="text-xs text-gray-400 capitalize">{participant.role}</span>
                </div>
                
                <div className="flex items-center space-x-3">
                  <div className="text-xs text-gray-400 text-right">
                    <div>{formatDuration(Math.floor(participant.totalSpeakTime / 1000))}</div>
                    <div>{participant.turnCount} turns</div>
                  </div>
                  
                  {participant.id !== currentTurn?.participantId && (
                    <button
                      onClick={() => addToQueue(participant.id)}
                      disabled={turnQueue.includes(participant.id)}
                      className={`p-1 rounded transition-colors ${
                        turnQueue.includes(participant.id)
                          ? 'text-gray-500 cursor-not-allowed'
                          : 'text-blue-400 hover:text-blue-300'
                      }`}
                      title="Request turn"
                    >
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Conversation Statistics */}
      <div className="p-3 bg-gray-500/10 rounded-lg">
        <h4 className="text-sm text-gray-300 font-medium mb-2">Session Stats</h4>
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div>
            <span className="block text-white">{formatDuration(Math.floor(conversationStats.totalDuration / 1000))}</span>
            <span className="text-gray-400">Total Duration</span>
          </div>
          <div>
            <span className="block text-white">{conversationStats.totalTurns}</span>
            <span className="text-gray-400">Total Turns</span>
          </div>
          <div>
            <span className="block text-white">{formatDuration(Math.floor(conversationStats.averageTurnLength / 1000))}</span>
            <span className="text-gray-400">Avg Turn Length</span>
          </div>
          <div>
            <span className={`block ${
              conversationStats.participationBalance < 0.3 ? 'text-green-400' :
              conversationStats.participationBalance < 0.6 ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {Math.round((1 - conversationStats.participationBalance) * 100)}%
            </span>
            <span className="text-gray-400">Balance</span>
          </div>
        </div>
      </div>

      {/* Warning Messages */}
      <AnimatePresence>
        {turnWarningShown && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mt-3 p-2 bg-yellow-500/10 border border-yellow-500/20 rounded text-yellow-400 text-sm flex items-center space-x-2"
          >
            <AlertTriangle className="w-4 h-4" />
            <span>Turn time limit approaching. Consider wrapping up.</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
