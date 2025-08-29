'use client';

import { motion } from 'framer-motion';
import { 
  Play, 
  Pause, 
  Camera, 
  CameraOff, 
  Mic, 
  MicOff,
  Settings,
  ArrowLeftRight,
  MessageSquare
} from 'lucide-react';

interface SessionControlsProps {
  isActive: boolean;
  mode: 'sign-to-text' | 'text-to-sign' | 'conversation';
  onToggleSession: () => void;
  onModeChange: (mode: 'sign-to-text' | 'text-to-sign' | 'conversation') => void;
  audioEnabled: boolean;
  videoEnabled: boolean;
  onAudioToggle: (enabled: boolean) => void;
  onVideoToggle: (enabled: boolean) => void;
  className?: string;
}

export function SessionControls({
  isActive,
  mode,
  onToggleSession,
  onModeChange,
  audioEnabled,
  videoEnabled,
  onAudioToggle,
  onVideoToggle,
  className = ''
}: SessionControlsProps) {
  
  const modes = [
    {
      id: 'sign-to-text' as const,
      label: 'Sign → Text',
      icon: ArrowLeftRight,
      description: 'Translate sign language to text'
    },
    {
      id: 'text-to-sign' as const,
      label: 'Text → Sign',
      icon: ArrowLeftRight,
      description: 'Convert text to sign language avatar'
    },
    {
      id: 'conversation' as const,
      label: 'Conversation',
      icon: MessageSquare,
      description: 'Bidirectional conversation mode'
    }
  ];

  return (
    <div className={`flex items-center space-x-4 ${className}`}>
      
      {/* Mode Selector */}
      <div className="flex items-center space-x-1 bg-black/20 rounded-lg p-1">
        {modes.map((modeOption) => {
          const Icon = modeOption.icon;
          return (
            <button
              key={modeOption.id}
              onClick={() => onModeChange(modeOption.id)}
              className={`
                relative px-3 py-2 rounded-md text-sm font-medium transition-all duration-200
                ${mode === modeOption.id
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'text-gray-300 hover:text-white hover:bg-white/10'
                }
              `}
              title={modeOption.description}
            >
              <div className="flex items-center space-x-2">
                <Icon className="w-4 h-4" />
                <span className="hidden sm:inline">{modeOption.label}</span>
              </div>
              
              {mode === modeOption.id && (
                <motion.div
                  layoutId="activeMode"
                  className="absolute inset-0 bg-blue-500 rounded-md -z-10"
                  initial={false}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
            </button>
          );
        })}
      </div>

      {/* Media Controls */}
      <div className="flex items-center space-x-2">
        {/* Video Toggle */}
        <button
          onClick={() => onVideoToggle(!videoEnabled)}
          className={`
            p-2 rounded-lg transition-all duration-200
            ${videoEnabled
              ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
              : 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
            }
          `}
          title={videoEnabled ? 'Turn off camera' : 'Turn on camera'}
        >
          {videoEnabled ? (
            <Camera className="w-4 h-4" />
          ) : (
            <CameraOff className="w-4 h-4" />
          )}
        </button>

        {/* Audio Toggle */}
        <button
          onClick={() => onAudioToggle(!audioEnabled)}
          className={`
            p-2 rounded-lg transition-all duration-200
            ${audioEnabled
              ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
              : 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
            }
          `}
          title={audioEnabled ? 'Mute audio' : 'Unmute audio'}
        >
          {audioEnabled ? (
            <Mic className="w-4 h-4" />
          ) : (
            <MicOff className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* Main Session Toggle */}
      <motion.button
        onClick={onToggleSession}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        className={`
          flex items-center space-x-2 px-6 py-2 rounded-lg font-medium transition-all duration-200
          ${isActive
            ? 'bg-red-500 hover:bg-red-600 text-white shadow-lg'
            : 'bg-blue-500 hover:bg-blue-600 text-white shadow-lg'
          }
        `}
      >
        {isActive ? (
          <>
            <Pause className="w-4 h-4" />
            <span>Stop Session</span>
          </>
        ) : (
          <>
            <Play className="w-4 h-4" />
            <span>Start Session</span>
          </>
        )}
      </motion.button>

      {/* Settings */}
      <button
        className="p-2 text-gray-400 hover:text-white transition-colors"
        title="Session settings"
      >
        <Settings className="w-4 h-4" />
      </button>
    </div>
  );
}
