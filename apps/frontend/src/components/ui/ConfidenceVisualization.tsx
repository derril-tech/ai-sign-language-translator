'use client';

import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, TrendingDown, Activity, AlertTriangle } from 'lucide-react';

interface ConfidenceData {
  timestamp: number;
  overall: number;
  pose: number;
  gloss: number;
  semantic: number;
  rag: number;
}

interface ConfidenceVisualizationProps {
  confidenceHistory: ConfidenceData[];
  currentConfidence: number;
  isLive: boolean;
  showDetails?: boolean;
  className?: string;
}

export function ConfidenceVisualization({
  confidenceHistory,
  currentConfidence,
  isLive,
  showDetails = true,
  className = ''
}: ConfidenceVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [trend, setTrend] = useState<'up' | 'down' | 'stable'>('stable');
  const [alertLevel, setAlertLevel] = useState<'none' | 'low' | 'medium' | 'high'>('none');

  // Calculate trend from recent data
  useEffect(() => {
    if (confidenceHistory.length >= 5) {
      const recent = confidenceHistory.slice(-5);
      const first = recent[0].overall;
      const last = recent[recent.length - 1].overall;
      const diff = last - first;
      
      if (diff > 0.05) setTrend('up');
      else if (diff < -0.05) setTrend('down');
      else setTrend('stable');
    }
  }, [confidenceHistory]);

  // Calculate alert level
  useEffect(() => {
    if (currentConfidence < 0.3) setAlertLevel('high');
    else if (currentConfidence < 0.5) setAlertLevel('medium');
    else if (currentConfidence < 0.7) setAlertLevel('low');
    else setAlertLevel('none');
  }, [currentConfidence]);

  // Draw confidence graph
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || confidenceHistory.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = canvas;
    const padding = 20;
    const graphWidth = width - padding * 2;
    const graphHeight = height - padding * 2;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw background grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    
    // Horizontal grid lines
    for (let i = 0; i <= 4; i++) {
      const y = padding + (graphHeight / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Vertical grid lines
    const timeSpan = Math.max(1, confidenceHistory.length - 1);
    for (let i = 0; i <= 4; i++) {
      const x = padding + (graphWidth / 4) * i;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
    }

    // Draw confidence lines
    const drawLine = (data: number[], color: string, lineWidth: number = 2) => {
      if (data.length < 2) return;

      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();

      data.forEach((value, index) => {
        const x = padding + (graphWidth / Math.max(1, data.length - 1)) * index;
        const y = padding + graphHeight * (1 - value);
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();
    };

    // Extract data series
    const overallData = confidenceHistory.map(d => d.overall);
    const poseData = confidenceHistory.map(d => d.pose);
    const glossData = confidenceHistory.map(d => d.gloss);
    const semanticData = confidenceHistory.map(d => d.semantic);

    // Draw component lines
    if (showDetails) {
      drawLine(poseData, '#10b981', 1); // Green for pose
      drawLine(glossData, '#f59e0b', 1); // Amber for gloss
      drawLine(semanticData, '#8b5cf6', 1); // Purple for semantic
    }

    // Draw overall confidence line (thicker)
    drawLine(overallData, '#3b82f6', 3); // Blue for overall

    // Draw current confidence indicator
    if (overallData.length > 0) {
      const lastX = padding + graphWidth;
      const lastY = padding + graphHeight * (1 - currentConfidence);
      
      ctx.fillStyle = '#3b82f6';
      ctx.beginPath();
      ctx.arc(lastX, lastY, 4, 0, 2 * Math.PI);
      ctx.fill();
    }

  }, [confidenceHistory, currentConfidence, showDetails]);

  // Get confidence color
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    if (confidence >= 0.4) return 'text-orange-400';
    return 'text-red-400';
  };

  // Get trend icon
  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-green-400" />;
      case 'down':
        return <TrendingDown className="w-4 h-4 text-red-400" />;
      default:
        return <Activity className="w-4 h-4 text-gray-400" />;
    }
  };

  // Get alert icon
  const getAlertIcon = () => {
    if (alertLevel === 'none') return null;
    
    const colors = {
      low: 'text-yellow-400',
      medium: 'text-orange-400',
      high: 'text-red-400'
    };

    return (
      <motion.div
        animate={{ opacity: [1, 0.5, 1] }}
        transition={{ duration: 1, repeat: Infinity }}
        className={`${colors[alertLevel]}`}
      >
        <AlertTriangle className="w-4 h-4" />
      </motion.div>
    );
  };

  return (
    <div className={`bg-black/20 backdrop-blur-md rounded-lg border border-white/10 p-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <h3 className="text-white font-medium">Confidence Monitor</h3>
          {isLive && (
            <motion.div
              animate={{ opacity: [1, 0.5, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="text-xs text-blue-400 bg-blue-500/10 px-2 py-1 rounded"
            >
              LIVE
            </motion.div>
          )}
        </div>

        <div className="flex items-center space-x-2">
          {getAlertIcon()}
          {getTrendIcon()}
        </div>
      </div>

      {/* Current Confidence Display */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-300">Overall Confidence</span>
          <span className={`text-lg font-bold ${getConfidenceColor(currentConfidence)}`}>
            {Math.round(currentConfidence * 100)}%
          </span>
        </div>

        {/* Confidence Bar */}
        <div className="relative h-3 bg-gray-700 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${currentConfidence * 100}%` }}
            transition={{ duration: 0.5 }}
            className={`h-full rounded-full ${
              currentConfidence >= 0.8 ? 'bg-green-500' :
              currentConfidence >= 0.6 ? 'bg-yellow-500' :
              currentConfidence >= 0.4 ? 'bg-orange-500' : 'bg-red-500'
            }`}
          />
          
          {/* Threshold markers */}
          <div className="absolute inset-0 flex justify-between items-center px-1">
            {[0.2, 0.4, 0.6, 0.8].map((threshold) => (
              <div
                key={threshold}
                className="w-px h-4 bg-gray-600"
                style={{ left: `${threshold * 100}%` }}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Component Breakdown */}
      {showDetails && confidenceHistory.length > 0 && (
        <div className="mb-4 space-y-2">
          <h4 className="text-sm text-gray-300 font-medium">Component Breakdown</h4>
          
          {[
            { name: 'Pose Detection', value: confidenceHistory[confidenceHistory.length - 1]?.pose || 0, color: 'bg-green-500' },
            { name: 'Gloss Recognition', value: confidenceHistory[confidenceHistory.length - 1]?.gloss || 0, color: 'bg-yellow-500' },
            { name: 'Semantic Translation', value: confidenceHistory[confidenceHistory.length - 1]?.semantic || 0, color: 'bg-purple-500' },
            { name: 'RAG Enhancement', value: confidenceHistory[confidenceHistory.length - 1]?.rag || 0, color: 'bg-blue-500' }
          ].map((component) => (
            <div key={component.name} className="flex items-center justify-between text-xs">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${component.color}`} />
                <span className="text-gray-400">{component.name}</span>
              </div>
              <span className="text-gray-300">{Math.round(component.value * 100)}%</span>
            </div>
          ))}
        </div>
      )}

      {/* Confidence Graph */}
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={300}
          height={120}
          className="w-full h-30 bg-gray-900/50 rounded"
        />
        
        {/* Graph Labels */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1 left-2 text-xs text-gray-500">100%</div>
          <div className="absolute top-1/2 left-2 text-xs text-gray-500">50%</div>
          <div className="absolute bottom-1 left-2 text-xs text-gray-500">0%</div>
          <div className="absolute bottom-1 right-2 text-xs text-gray-500">Now</div>
        </div>
      </div>

      {/* Legend */}
      {showDetails && (
        <div className="mt-3 flex flex-wrap gap-3 text-xs">
          <div className="flex items-center space-x-1">
            <div className="w-2 h-0.5 bg-blue-500" />
            <span className="text-gray-400">Overall</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-0.5 bg-green-500" />
            <span className="text-gray-400">Pose</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-0.5 bg-yellow-500" />
            <span className="text-gray-400">Gloss</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-0.5 bg-purple-500" />
            <span className="text-gray-400">Semantic</span>
          </div>
        </div>
      )}

      {/* Status Messages */}
      <AnimatePresence>
        {alertLevel !== 'none' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className={`mt-3 p-2 rounded text-xs ${
              alertLevel === 'high' ? 'bg-red-500/10 text-red-400' :
              alertLevel === 'medium' ? 'bg-orange-500/10 text-orange-400' :
              'bg-yellow-500/10 text-yellow-400'
            }`}
          >
            {alertLevel === 'high' && 'Low confidence detected. Check camera positioning and lighting.'}
            {alertLevel === 'medium' && 'Moderate confidence. Consider adjusting signing speed or clarity.'}
            {alertLevel === 'low' && 'Confidence could be improved. Ensure clear hand visibility.'}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
