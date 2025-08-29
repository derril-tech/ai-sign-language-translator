'use client';

import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface ConfidenceBarProps {
  confidence: number;
  isProcessing: boolean;
  trend?: 'up' | 'down' | 'stable';
  className?: string;
}

export function ConfidenceBar({
  confidence,
  isProcessing,
  trend = 'stable',
  className = ''
}: ConfidenceBarProps) {
  // Get confidence level and color
  const getConfidenceLevel = (conf: number) => {
    if (conf >= 0.8) return { level: 'High', color: 'green' };
    if (conf >= 0.6) return { level: 'Good', color: 'yellow' };
    if (conf >= 0.4) return { level: 'Fair', color: 'orange' };
    return { level: 'Low', color: 'red' };
  };

  const { level, color } = getConfidenceLevel(confidence);

  // Get trend icon
  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-3 h-3 text-green-400" />;
      case 'down':
        return <TrendingDown className="w-3 h-3 text-red-400" />;
      default:
        return <Minus className="w-3 h-3 text-gray-400" />;
    }
  };

  // Get color classes
  const getColorClasses = (colorName: string) => {
    const colors = {
      green: {
        bg: 'bg-green-500',
        text: 'text-green-400',
        border: 'border-green-500/30'
      },
      yellow: {
        bg: 'bg-yellow-500',
        text: 'text-yellow-400',
        border: 'border-yellow-500/30'
      },
      orange: {
        bg: 'bg-orange-500',
        text: 'text-orange-400',
        border: 'border-orange-500/30'
      },
      red: {
        bg: 'bg-red-500',
        text: 'text-red-400',
        border: 'border-red-500/30'
      }
    };
    return colors[colorName as keyof typeof colors] || colors.red;
  };

  const colorClasses = getColorClasses(color);

  return (
    <div className={`bg-black/40 backdrop-blur-md rounded-lg border border-white/10 p-3 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-gray-300 font-medium">Confidence</span>
        {getTrendIcon()}
      </div>

      {/* Confidence Bar */}
      <div className="relative">
        {/* Background Bar */}
        <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
          {/* Progress Bar */}
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${confidence * 100}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className={`h-full ${colorClasses.bg} relative`}
          >
            {/* Animated Shine Effect */}
            {isProcessing && (
              <motion.div
                animate={{ x: [-20, 100] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent w-5"
              />
            )}
          </motion.div>
        </div>

        {/* Confidence Markers */}
        <div className="absolute inset-0 flex justify-between items-center px-1">
          {[0.2, 0.4, 0.6, 0.8].map((marker) => (
            <div
              key={marker}
              className="w-px h-3 bg-gray-600"
              style={{ left: `${marker * 100}%` }}
            />
          ))}
        </div>
      </div>

      {/* Confidence Details */}
      <div className="flex items-center justify-between mt-2">
        <div className="flex items-center space-x-2">
          <span className={`text-xs font-medium ${colorClasses.text}`}>
            {level}
          </span>
          <span className="text-xs text-gray-400">
            {Math.round(confidence * 100)}%
          </span>
        </div>

        {/* Processing Indicator */}
        {isProcessing && (
          <motion.div
            animate={{ opacity: [1, 0.5, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
            className="flex items-center space-x-1"
          >
            <div className="w-1 h-1 bg-blue-400 rounded-full" />
            <div className="w-1 h-1 bg-blue-400 rounded-full" />
            <div className="w-1 h-1 bg-blue-400 rounded-full" />
          </motion.div>
        )}
      </div>

      {/* Confidence Breakdown (Optional) */}
      <div className="mt-2 space-y-1">
        <div className="flex justify-between text-xs">
          <span className="text-gray-400">Pose</span>
          <span className="text-gray-300">85%</span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-gray-400">Gloss</span>
          <span className="text-gray-300">78%</span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-gray-400">Semantic</span>
          <span className="text-gray-300">82%</span>
        </div>
      </div>
    </div>
  );
}
