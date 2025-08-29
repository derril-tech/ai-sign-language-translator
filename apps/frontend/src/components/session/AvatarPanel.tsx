'use client';

import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { User, Settings, Play, Pause } from 'lucide-react';
import { AvatarRenderer } from '@/lib/avatar/AvatarRenderer';

interface AvatarPanelProps {
  mode: 'sign-to-text' | 'text-to-sign' | 'conversation';
  text: string;
  isActive: boolean;
  className?: string;
}

export function AvatarPanel({
  mode,
  text,
  isActive,
  className = ''
}: AvatarPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const avatarRendererRef = useRef<AvatarRenderer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [avatarLoaded, setAvatarLoaded] = useState(false);

  // Initialize avatar renderer
  useEffect(() => {
    if (canvasRef.current && !avatarRendererRef.current) {
      try {
        avatarRendererRef.current = new AvatarRenderer(canvasRef.current);
        avatarRendererRef.current.initialize();
        setAvatarLoaded(true);
      } catch (error) {
        console.error('Failed to initialize avatar renderer:', error);
      }
    }

    return () => {
      if (avatarRendererRef.current) {
        avatarRendererRef.current.dispose();
        avatarRendererRef.current = null;
      }
    };
  }, []);

  // Handle text changes for text-to-sign mode
  useEffect(() => {
    if (avatarRendererRef.current && text && mode === 'text-to-sign' && isActive) {
      avatarRendererRef.current.animateText(text);
      setIsPlaying(true);
    }
  }, [text, mode, isActive]);

  // Handle avatar animation playback
  const handlePlayPause = () => {
    if (avatarRendererRef.current) {
      if (isPlaying) {
        avatarRendererRef.current.stop();
        setIsPlaying(false);
      } else {
        if (text) {
          avatarRendererRef.current.animateText(text);
          setIsPlaying(true);
        } else {
          avatarRendererRef.current.play();
          setIsPlaying(true);
        }
      }
    }
  };

  return (
    <div className={`bg-black/20 backdrop-blur-md rounded-lg border border-white/10 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/10">
        <div className="flex items-center space-x-3">
          <User className="w-5 h-5 text-purple-400" />
          <h3 className="text-white font-medium">Sign Avatar</h3>
          {mode === 'text-to-sign' && (
            <span className="text-xs text-purple-400 bg-purple-500/10 px-2 py-1 rounded">
              Text → Sign
            </span>
          )}
        </div>

        <div className="flex items-center space-x-2">
          {text && (
            <button
              onClick={handlePlayPause}
              className="p-2 text-gray-400 hover:text-white transition-colors"
              title={isPlaying ? 'Pause animation' : 'Play animation'}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </button>
          )}
          
          <button className="p-2 text-gray-400 hover:text-white transition-colors">
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Avatar Canvas */}
      <div className="relative p-4">
        <canvas
          ref={canvasRef}
          width={300}
          height={400}
          className="w-full h-full bg-gradient-to-b from-gray-800 to-gray-900 rounded-lg"
        />

        {/* Loading State */}
        {!avatarLoaded && (
          <div className="absolute inset-4 bg-gray-800 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-400 mx-auto mb-2"></div>
              <p className="text-gray-400 text-sm">Loading avatar...</p>
            </div>
          </div>
        )}

        {/* No Text State */}
        {!text && avatarLoaded && (
          <div className="absolute inset-4 flex items-center justify-center">
            <div className="text-center text-gray-400">
              <User className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p className="text-sm">
                {mode === 'text-to-sign' 
                  ? 'Enter text to see sign animation'
                  : 'Avatar ready for conversation'
                }
              </p>
            </div>
          </div>
        )}

        {/* Animation Status */}
        {text && isActive && (
          <motion.div
            animate={{ opacity: [1, 0.5, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="absolute top-6 right-6 bg-purple-500/20 backdrop-blur-sm rounded px-2 py-1"
          >
            <span className="text-purple-400 text-xs">Signing</span>
          </motion.div>
        )}
      </div>

      {/* Current Text Display */}
      {text && (
        <div className="p-4 border-t border-white/10">
          <div className="bg-black/20 rounded p-3">
            <p className="text-white text-sm leading-relaxed">{text}</p>
          </div>
        </div>
      )}

      {/* Avatar Info */}
      <div className="px-4 pb-4">
        <div className="text-xs text-gray-400 space-y-1">
          <div className="flex justify-between">
            <span>Avatar Model:</span>
            <span>Default ASL Signer</span>
          </div>
          <div className="flex justify-between">
            <span>Sign Language:</span>
            <span>American Sign Language</span>
          </div>
          <div className="flex justify-between">
            <span>Quality:</span>
            <span className="text-green-400">High</span>
          </div>
        </div>
      </div>
    </div>
  );
}
