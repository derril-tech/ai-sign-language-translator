import numpy as np
import base64
import io
import wave
from typing import Optional, Tuple
import logging
from scipy import signal
import struct

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio processing utilities for TTS output"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
    def process_audio(self, 
                     audio_data: np.ndarray,
                     enhance_quality: bool = True,
                     normalize: bool = True,
                     remove_noise: bool = True) -> np.ndarray:
        """Process audio with various enhancements"""
        try:
            processed_audio = audio_data.copy()
            
            if remove_noise:
                processed_audio = self._remove_noise(processed_audio)
            
            if enhance_quality:
                processed_audio = self._enhance_quality(processed_audio)
            
            if normalize:
                processed_audio = self._normalize_audio(processed_audio)
            
            return processed_audio
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return audio_data
    
    def _remove_noise(self, audio: np.ndarray) -> np.ndarray:
        """Remove noise from audio"""
        try:
            # Simple high-pass filter to remove low-frequency noise
            nyquist = self.sample_rate / 2
            low_cutoff = 80  # Hz
            
            if low_cutoff < nyquist:
                sos = signal.butter(4, low_cutoff / nyquist, btype='high', output='sos')
                filtered_audio = signal.sosfilt(sos, audio)
                return filtered_audio
            
            return audio
            
        except Exception as e:
            logger.error(f"Error removing noise: {e}")
            return audio
    
    def _enhance_quality(self, audio: np.ndarray) -> np.ndarray:
        """Enhance audio quality"""
        try:
            # Apply gentle compression
            compressed_audio = self._compress_audio(audio, threshold=0.7, ratio=3.0)
            
            # Apply subtle EQ boost to speech frequencies
            enhanced_audio = self._apply_speech_eq(compressed_audio)
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Error enhancing quality: {e}")
            return audio
    
    def _compress_audio(self, audio: np.ndarray, threshold: float = 0.7, ratio: float = 3.0) -> np.ndarray:
        """Apply dynamic range compression"""
        try:
            # Simple compression algorithm
            compressed = audio.copy()
            
            # Find samples above threshold
            above_threshold = np.abs(compressed) > threshold
            
            # Apply compression to samples above threshold
            compressed[above_threshold] = np.sign(compressed[above_threshold]) * (
                threshold + (np.abs(compressed[above_threshold]) - threshold) / ratio
            )
            
            return compressed
            
        except Exception as e:
            logger.error(f"Error in compression: {e}")
            return audio
    
    def _apply_speech_eq(self, audio: np.ndarray) -> np.ndarray:
        """Apply EQ optimized for speech"""
        try:
            # Boost speech frequencies (1-4 kHz)
            nyquist = self.sample_rate / 2
            
            # Design bandpass filter for speech enhancement
            low_freq = 1000 / nyquist
            high_freq = 4000 / nyquist
            
            if high_freq < 1.0:
                sos = signal.butter(2, [low_freq, high_freq], btype='band', output='sos')
                speech_band = signal.sosfilt(sos, audio)
                
                # Mix with original (subtle boost)
                enhanced = audio + 0.2 * speech_band
                return enhanced
            
            return audio
            
        except Exception as e:
            logger.error(f"Error applying speech EQ: {e}")
            return audio
    
    def _normalize_audio(self, audio: np.ndarray, target_level: float = 0.8) -> np.ndarray:
        """Normalize audio to target level"""
        try:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                normalized = audio * (target_level / max_val)
                return normalized
            return audio
            
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return audio
    
    def encode_audio_base64(self, audio_data: np.ndarray, format: str = "wav") -> str:
        """Encode audio data to base64 string"""
        try:
            # Convert to 16-bit PCM
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Create WAV file in memory
            buffer = io.BytesIO()
            
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Encode to base64
            buffer.seek(0)
            audio_bytes = buffer.read()
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            
            return f"data:audio/wav;base64,{base64_audio}"
            
        except Exception as e:
            logger.error(f"Error encoding audio to base64: {e}")
            return ""
    
    def decode_audio_base64(self, base64_data: str) -> Optional[np.ndarray]:
        """Decode base64 audio data"""
        try:
            # Remove data URL prefix if present
            if base64_data.startswith('data:audio'):
                base64_data = base64_data.split(',')[1]
            
            # Decode base64
            audio_bytes = base64.b64decode(base64_data)
            
            # Read WAV data
            buffer = io.BytesIO(audio_bytes)
            
            with wave.open(buffer, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
                
                # Convert to float32
                audio_float = audio_data.astype(np.float32) / 32767.0
                
                return audio_float
            
        except Exception as e:
            logger.error(f"Error decoding base64 audio: {e}")
            return None
    
    def resample_audio(self, audio: np.ndarray, 
                      original_rate: int, 
                      target_rate: int) -> np.ndarray:
        """Resample audio to different sample rate"""
        try:
            if original_rate == target_rate:
                return audio
            
            # Calculate resampling ratio
            ratio = target_rate / original_rate
            
            # Use scipy's resample function
            num_samples = int(len(audio) * ratio)
            resampled = signal.resample(audio, num_samples)
            
            return resampled.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return audio
    
    def apply_fade(self, audio: np.ndarray, 
                  fade_in_ms: float = 10, 
                  fade_out_ms: float = 10) -> np.ndarray:
        """Apply fade in/out to audio"""
        try:
            fade_in_samples = int(fade_in_ms * self.sample_rate / 1000)
            fade_out_samples = int(fade_out_ms * self.sample_rate / 1000)
            
            audio_faded = audio.copy()
            
            # Apply fade in
            if fade_in_samples > 0 and fade_in_samples < len(audio):
                fade_in = np.linspace(0, 1, fade_in_samples)
                audio_faded[:fade_in_samples] *= fade_in
            
            # Apply fade out
            if fade_out_samples > 0 and fade_out_samples < len(audio):
                fade_out = np.linspace(1, 0, fade_out_samples)
                audio_faded[-fade_out_samples:] *= fade_out
            
            return audio_faded
            
        except Exception as e:
            logger.error(f"Error applying fade: {e}")
            return audio
    
    def calculate_audio_metrics(self, audio: np.ndarray) -> dict:
        """Calculate audio quality metrics"""
        try:
            metrics = {
                'duration_ms': len(audio) / self.sample_rate * 1000,
                'peak_amplitude': float(np.max(np.abs(audio))),
                'rms_level': float(np.sqrt(np.mean(audio ** 2))),
                'dynamic_range': float(np.max(audio) - np.min(audio)),
                'zero_crossings': int(np.sum(np.diff(np.signbit(audio)))),
                'sample_rate': self.sample_rate
            }
            
            # Calculate SNR estimate (very rough)
            if metrics['rms_level'] > 0:
                noise_floor = np.percentile(np.abs(audio), 10)  # Estimate noise as 10th percentile
                snr_estimate = 20 * np.log10(metrics['rms_level'] / (noise_floor + 1e-8))
                metrics['snr_estimate_db'] = float(snr_estimate)
            else:
                metrics['snr_estimate_db'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating audio metrics: {e}")
            return {}
