'use client';

import { useState, useRef, useCallback, useEffect } from 'react';

interface UseWebRTCReturn {
  stream: MediaStream | null;
  isConnected: boolean;
  error: string | null;
  startCamera: () => Promise<void>;
  stopCamera: () => Promise<void>;
  captureFrame: (video: HTMLVideoElement, canvas: HTMLCanvasElement) => string | null;
  switchCamera: () => Promise<void>;
}

export function useWebRTC(): UseWebRTCReturn {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentDeviceId, setCurrentDeviceId] = useState<string>('');
  
  const streamRef = useRef<MediaStream | null>(null);

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      setError(null);
      
      const constraints: MediaStreamConstraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 },
          facingMode: 'user', // Front-facing camera
          deviceId: currentDeviceId || undefined
        },
        audio: false // We'll handle audio separately for TTS/ASR
      };

      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      
      streamRef.current = mediaStream;
      setStream(mediaStream);
      setIsConnected(true);
      
      // Get the actual device ID being used
      const videoTrack = mediaStream.getVideoTracks()[0];
      if (videoTrack) {
        const settings = videoTrack.getSettings();
        if (settings.deviceId) {
          setCurrentDeviceId(settings.deviceId);
        }
      }
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to access camera';
      setError(errorMessage);
      setIsConnected(false);
      console.error('Camera access error:', err);
    }
  }, [currentDeviceId]);

  // Stop camera
  const stopCamera = useCallback(async () => {
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          track.stop();
        });
        streamRef.current = null;
        setStream(null);
        setIsConnected(false);
      }
    } catch (err) {
      console.error('Error stopping camera:', err);
    }
  }, []);

  // Capture frame from video element
  const captureFrame = useCallback((
    video: HTMLVideoElement, 
    canvas: HTMLCanvasElement
  ): string | null => {
    try {
      if (!video || !canvas || video.videoWidth === 0 || video.videoHeight === 0) {
        return null;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) return null;

      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw video frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert to base64 data URL
      return canvas.toDataURL('image/jpeg', 0.8);
      
    } catch (err) {
      console.error('Error capturing frame:', err);
      return null;
    }
  }, []);

  // Switch between available cameras
  const switchCamera = useCallback(async () => {
    try {
      // Get available video devices
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      
      if (videoDevices.length > 1) {
        // Find next device
        const currentIndex = videoDevices.findIndex(device => device.deviceId === currentDeviceId);
        const nextIndex = (currentIndex + 1) % videoDevices.length;
        const nextDevice = videoDevices[nextIndex];
        
        // Stop current stream
        await stopCamera();
        
        // Start with new device
        setCurrentDeviceId(nextDevice.deviceId);
        
        // The useEffect will trigger startCamera with the new device
      }
    } catch (err) {
      console.error('Error switching camera:', err);
      setError('Failed to switch camera');
    }
  }, [currentDeviceId, stopCamera]);

  // Auto-restart camera when device changes
  useEffect(() => {
    if (currentDeviceId && isConnected) {
      startCamera();
    }
  }, [currentDeviceId]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Handle device changes
  useEffect(() => {
    const handleDeviceChange = () => {
      // Optionally restart camera if device list changes
      console.log('Media devices changed');
    };

    navigator.mediaDevices?.addEventListener('devicechange', handleDeviceChange);
    
    return () => {
      navigator.mediaDevices?.removeEventListener('devicechange', handleDeviceChange);
    };
  }, []);

  return {
    stream,
    isConnected,
    error,
    startCamera,
    stopCamera,
    captureFrame,
    switchCamera
  };
}
