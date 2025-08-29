'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, 
  Pause, 
  Volume2, 
  VolumeX, 
  Settings, 
  RotateCcw,
  Download,
  Mic
} from 'lucide-react';

interface Voice {
  id: string;
  name: string;
  language: string;
  gender: 'male' | 'female' | 'neutral';
}

interface TTSDockProps {
  text: string;
  isEnabled: boolean;
  onSpeak: (text: string) => void;
  onVoiceChange: (voiceId: string) => void;
  className?: string;
}

export function TTSDock({
  text,
  isEnabled,
  onSpeak,
  onVoiceChange,
  className = ''
}: TTSDockProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(0.8);
  const [speed, setSpeed] = useState(1.0);
  const [selectedVoice, setSelectedVoice] = useState('default');
  const [showSettings, setShowSettings] = useState(false);
  const [audioQueue, setAudioQueue] = useState<string[]>([]);
  
  const audioRef = useRef<HTMLAudioElement>(null);

  // Available voices
  const voices: Voice[] = [
    { id: 'default', name: 'Default Voice', language: 'en-US', gender: 'neutral' },
    { id: 'female1', name: 'Sarah', language: 'en-US', gender: 'female' },
    { id: 'male1', name: 'David', language: 'en-US', gender: 'male' },
    { id: 'female2', name: 'Emma', language: 'en-US', gender: 'female' },
    { id: 'male2', name: 'James', language: 'en-US', gender: 'male' },
  ];

  // Handle text-to-speech
  const handleSpeak = async () => {
    if (!text || !isEnabled) return;

    setIsPlaying(true);
    
    try {
      // Call the TTS service
      onSpeak(text);
      
      // Simulate audio playback
      setTimeout(() => {
        setIsPlaying(false);
      }, text.length * 50); // Rough estimate based on text length
      
    } catch (error) {
      console.error('TTS Error:', error);
      setIsPlaying(false);
    }
  };

  // Handle stop
  const handleStop = () => {
    setIsPlaying(false);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };

  // Handle voice change
  const handleVoiceChange = (voiceId: string) => {
    setSelectedVoice(voiceId);
    onVoiceChange(voiceId);
  };

  // Auto-speak new text (optional)
  const [autoSpeak, setAutoSpeak] = useState(false);
  useEffect(() => {
    if (autoSpeak && text && isEnabled && !isPlaying) {
      const timer = setTimeout(() => {
        handleSpeak();
      }, 500); // Small delay to avoid rapid firing
      
      return () => clearTimeout(timer);
    }
  }, [text, autoSpeak, isEnabled, isPlaying]);

  // Download audio
  const handleDownloadAudio = () => {
    // In a real implementation, this would download the generated audio
    console.log('Downloading audio for:', text);
  };

  return (
    <div className={`bg-black/20 backdrop-blur-md rounded-lg border border-white/10 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/10">
        <div className="flex items-center space-x-3">
          <Volume2 className="w-5 h-5 text-blue-400" />
          <h3 className="text-white font-medium">Text-to-Speech</h3>
          {isPlaying && (
            <motion.div
              animate={{ opacity: [1, 0.5, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
              className="text-xs text-green-400 bg-green-500/10 px-2 py-1 rounded"
            >
              SPEAKING
            </motion.div>
          )}
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setAutoSpeak(!autoSpeak)}
            className={`p-2 rounded transition-colors ${
              autoSpeak 
                ? 'bg-blue-500/20 text-blue-400' 
                : 'text-gray-400 hover:text-white'
            }`}
            title="Auto-speak new text"
          >
            <Mic className="w-4 h-4" />
          </button>
          
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title="TTS Settings"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Controls */}
      <div className="p-4">
        {/* Text Display */}
        <div className="bg-black/20 rounded-lg p-3 mb-4 min-h-[60px] max-h-[120px] overflow-y-auto">
          {text ? (
            <p className="text-white text-sm leading-relaxed">{text}</p>
          ) : (
            <p className="text-gray-400 text-sm italic">
              No text to speak. Start signing to generate captions.
            </p>
          )}
        </div>

        {/* Playback Controls */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            {/* Play/Pause Button */}
            <button
              onClick={isPlaying ? handleStop : handleSpeak}
              disabled={!text || !isEnabled}
              className={`
                p-3 rounded-full transition-all duration-200
                ${!text || !isEnabled
                  ? 'bg-gray-500/20 text-gray-500 cursor-not-allowed'
                  : isPlaying
                    ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                    : 'bg-blue-500/20 text-blue-400 hover:bg-blue-500/30'
                }
              `}
            >
              {isPlaying ? (
                <Pause className="w-5 h-5" />
              ) : (
                <Play className="w-5 h-5" />
              )}
            </button>

            {/* Repeat Button */}
            <button
              onClick={handleSpeak}
              disabled={!text || !isEnabled || isPlaying}
              className="p-2 text-gray-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              title="Repeat"
            >
              <RotateCcw className="w-4 h-4" />
            </button>

            {/* Download Button */}
            <button
              onClick={handleDownloadAudio}
              disabled={!text}
              className="p-2 text-gray-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              title="Download audio"
            >
              <Download className="w-4 h-4" />
            </button>
          </div>

          {/* Volume Control */}
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsMuted(!isMuted)}
              className="p-2 text-gray-400 hover:text-white transition-colors"
            >
              {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </button>
            
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={isMuted ? 0 : volume}
              onChange={(e) => setVolume(parseFloat(e.target.value))}
              className="w-20 accent-blue-500"
            />
            
            <span className="text-xs text-gray-400 w-8">
              {Math.round((isMuted ? 0 : volume) * 100)}
            </span>
          </div>
        </div>

        {/* Voice Selection */}
        <div className="mb-4">
          <label className="block text-sm text-gray-300 mb-2">Voice</label>
          <select
            value={selectedVoice}
            onChange={(e) => handleVoiceChange(e.target.value)}
            className="w-full bg-black/20 border border-white/10 rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
          >
            {voices.map((voice) => (
              <option key={voice.id} value={voice.id} className="bg-gray-800">
                {voice.name} ({voice.gender}, {voice.language})
              </option>
            ))}
          </select>
        </div>

        {/* Settings Panel */}
        <AnimatePresence>
          {showSettings && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="border-t border-white/10 pt-4"
            >
              {/* Speed Control */}
              <div className="mb-4">
                <label className="block text-sm text-gray-300 mb-2">
                  Speed: {speed.toFixed(1)}x
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  value={speed}
                  onChange={(e) => setSpeed(parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              {/* Auto-speak Toggle */}
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-300">Auto-speak new text</span>
                <button
                  onClick={() => setAutoSpeak(!autoSpeak)}
                  className={`
                    relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                    ${autoSpeak ? 'bg-blue-600' : 'bg-gray-600'}
                  `}
                >
                  <span
                    className={`
                      inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                      ${autoSpeak ? 'translate-x-6' : 'translate-x-1'}
                    `}
                  />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Hidden Audio Element */}
      <audio
        ref={audioRef}
        onEnded={() => setIsPlaying(false)}
        onError={() => setIsPlaying(false)}
        volume={isMuted ? 0 : volume}
      />
    </div>
  );
}
