'use client';

import { forwardRef, useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Camera, CameraOff, Settings } from 'lucide-react';

interface CameraStageProps {
  canvasRef: React.RefObject<HTMLCanvasElement>;
  stream: MediaStream | null;
  isActive: boolean;
  showPoseOverlay?: boolean;
  onFrameCapture?: (video: HTMLVideoElement, canvas: HTMLCanvasElement) => string | null;
  className?: string;
}

export const CameraStage = forwardRef<HTMLVideoElement, CameraStageProps>(({
  canvasRef,
  stream,
  isActive,
  showPoseOverlay = true,
  onFrameCapture,
  className = ''
}, ref) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();

  // Set up video stream
  useEffect(() => {
    if (ref && 'current' in ref && ref.current && stream) {
      ref.current.srcObject = stream;
      ref.current.play().catch(console.error);
    }
  }, [stream, ref]);

  // Draw pose overlay
  const drawPoseOverlay = () => {
    if (!overlayCanvasRef.current || !showPoseOverlay) return;

    const canvas = overlayCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Mock pose landmarks for demonstration
    if (isActive) {
      ctx.strokeStyle = '#00ff88';
      ctx.lineWidth = 2;
      ctx.fillStyle = '#00ff88';

      // Draw mock skeleton
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;

      // Body keypoints (simplified)
      const keypoints = [
        { x: centerX, y: centerY - 50 }, // Head
        { x: centerX, y: centerY }, // Torso
        { x: centerX - 30, y: centerY - 20 }, // Left shoulder
        { x: centerX + 30, y: centerY - 20 }, // Right shoulder
        { x: centerX - 50, y: centerY + 20 }, // Left elbow
        { x: centerX + 50, y: centerY + 20 }, // Right elbow
        { x: centerX - 70, y: centerY + 60 }, // Left hand
        { x: centerX + 70, y: centerY + 60 }, // Right hand
      ];

      // Draw keypoints
      keypoints.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 4, 0, 2 * Math.PI);
        ctx.fill();
      });

      // Draw connections
      const connections = [
        [0, 1], // Head to torso
        [1, 2], // Torso to left shoulder
        [1, 3], // Torso to right shoulder
        [2, 4], // Left shoulder to elbow
        [3, 5], // Right shoulder to elbow
        [4, 6], // Left elbow to hand
        [5, 7], // Right elbow to hand
      ];

      ctx.beginPath();
      connections.forEach(([start, end]) => {
        ctx.moveTo(keypoints[start].x, keypoints[start].y);
        ctx.lineTo(keypoints[end].x, keypoints[end].y);
      });
      ctx.stroke();

      // Draw hand regions
      ctx.strokeStyle = '#ff6b6b';
      ctx.lineWidth = 3;
      
      // Left hand box
      ctx.strokeRect(centerX - 85, centerY + 45, 30, 30);
      
      // Right hand box
      ctx.strokeRect(centerX + 55, centerY + 45, 30, 30);
    }

    animationFrameRef.current = requestAnimationFrame(drawPoseOverlay);
  };

  // Start pose overlay animation
  useEffect(() => {
    if (isActive && showPoseOverlay) {
      drawPoseOverlay();
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isActive, showPoseOverlay]);

  return (
    <div className={`relative bg-black rounded-lg overflow-hidden ${className}`}>
      {/* Video Element */}
      <video
        ref={ref}
        className="w-full h-full object-cover"
        autoPlay
        playsInline
        muted
        style={{ transform: 'scaleX(-1)' }} // Mirror for natural feel
      />

      {/* Pose Overlay Canvas */}
      {showPoseOverlay && (
        <canvas
          ref={overlayCanvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          width={640}
          height={480}
          style={{ transform: 'scaleX(-1)' }}
        />
      )}

      {/* Status Indicators */}
      <div className="absolute top-4 left-4 flex items-center space-x-2">
        {isActive ? (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="flex items-center space-x-2 bg-green-500/20 backdrop-blur-sm rounded-full px-3 py-1"
          >
            <Camera className="w-4 h-4 text-green-400" />
            <span className="text-green-400 text-sm font-medium">Live</span>
          </motion.div>
        ) : (
          <div className="flex items-center space-x-2 bg-red-500/20 backdrop-blur-sm rounded-full px-3 py-1">
            <CameraOff className="w-4 h-4 text-red-400" />
            <span className="text-red-400 text-sm font-medium">Offline</span>
          </div>
        )}
      </div>

      {/* Settings Button */}
      <button className="absolute top-4 right-4 p-2 bg-black/20 backdrop-blur-sm rounded-full hover:bg-black/40 transition-colors">
        <Settings className="w-4 h-4 text-white" />
      </button>

      {/* Loading State */}
      {isLoading && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-2"></div>
            <p className="text-white text-sm">Starting camera...</p>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center">
          <div className="text-center text-white p-4">
            <CameraOff className="w-12 h-12 mx-auto mb-2 text-red-400" />
            <p className="text-red-400 font-medium">Camera Error</p>
            <p className="text-sm text-gray-300 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* No Stream State */}
      {!stream && !isLoading && !error && (
        <div className="absolute inset-0 bg-gradient-to-br from-gray-800 to-gray-900 flex items-center justify-center">
          <div className="text-center text-white p-8">
            <Camera className="w-16 h-16 mx-auto mb-4 text-gray-400" />
            <h3 className="text-lg font-medium mb-2">Camera Ready</h3>
            <p className="text-gray-400">Click start to begin translation</p>
          </div>
        </div>
      )}

      {/* Frame Rate Indicator */}
      {isActive && (
        <div className="absolute bottom-4 left-4 bg-black/20 backdrop-blur-sm rounded px-2 py-1">
          <span className="text-white text-xs">30 FPS</span>
        </div>
      )}

      {/* Resolution Indicator */}
      {isActive && (
        <div className="absolute bottom-4 right-4 bg-black/20 backdrop-blur-sm rounded px-2 py-1">
          <span className="text-white text-xs">640x480</span>
        </div>
      )}
    </div>
  );
});
