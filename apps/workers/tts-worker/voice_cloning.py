import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VoiceProfile:
    """Voice profile for cloning"""
    voice_id: str
    embedding: torch.Tensor
    characteristics: Dict[str, Any]
    quality_score: float

class VoiceEncoder(nn.Module):
    """Neural network for voice encoding"""
    
    def __init__(self, input_dim: int = 80, embedding_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class VoiceCloner:
    """Voice cloning system for personalized TTS"""
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.voice_encoder = VoiceEncoder(embedding_dim=embedding_dim)
        self.voice_profiles: Dict[str, VoiceProfile] = {}
        
        # Load default voices
        self._initialize_default_voices()
        
        logger.info("VoiceCloner initialized")
    
    def _initialize_default_voices(self):
        """Initialize default voice profiles"""
        try:
            # Create default voice embeddings (random for development)
            default_voices = {
                "default": {"pitch": 200, "timbre": "neutral", "accent": "general"},
                "female1": {"pitch": 250, "timbre": "warm", "accent": "general"},
                "male1": {"pitch": 150, "timbre": "deep", "accent": "general"}
            }
            
            for voice_id, characteristics in default_voices.items():
                # Generate random embedding for development
                embedding = torch.randn(self.embedding_dim)
                
                profile = VoiceProfile(
                    voice_id=voice_id,
                    embedding=embedding,
                    characteristics=characteristics,
                    quality_score=0.8
                )
                
                self.voice_profiles[voice_id] = profile
            
            logger.info(f"Initialized {len(default_voices)} default voices")
            
        except Exception as e:
            logger.error(f"Error initializing default voices: {e}")
    
    def clone_voice(self, 
                   voice_id: str,
                   reference_audio: np.ndarray,
                   sample_rate: int = 22050) -> bool:
        """Clone a voice from reference audio"""
        try:
            # Extract features from reference audio
            features = self._extract_voice_features(reference_audio, sample_rate)
            
            if features is None:
                logger.error("Failed to extract voice features")
                return False
            
            # Generate voice embedding
            with torch.no_grad():
                embedding = self.voice_encoder(features)
            
            # Analyze voice characteristics
            characteristics = self._analyze_voice_characteristics(reference_audio, sample_rate)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(reference_audio, characteristics)
            
            # Create voice profile
            profile = VoiceProfile(
                voice_id=voice_id,
                embedding=embedding,
                characteristics=characteristics,
                quality_score=quality_score
            )
            
            self.voice_profiles[voice_id] = profile
            
            logger.info(f"Successfully cloned voice '{voice_id}' with quality score {quality_score:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error cloning voice: {e}")
            return False
    
    def get_voice_embedding(self, voice_id: str) -> Optional[torch.Tensor]:
        """Get voice embedding for synthesis"""
        try:
            if voice_id in self.voice_profiles:
                return self.voice_profiles[voice_id].embedding
            else:
                logger.warning(f"Voice '{voice_id}' not found, using default")
                return self.voice_profiles.get("default", {}).embedding
                
        except Exception as e:
            logger.error(f"Error getting voice embedding: {e}")
            return None
    
    def get_voice_characteristics(self, voice_id: str) -> Dict[str, Any]:
        """Get voice characteristics"""
        try:
            if voice_id in self.voice_profiles:
                return self.voice_profiles[voice_id].characteristics
            else:
                return self.voice_profiles.get("default", VoiceProfile(
                    voice_id="default",
                    embedding=torch.zeros(self.embedding_dim),
                    characteristics={},
                    quality_score=0.0
                )).characteristics
                
        except Exception as e:
            logger.error(f"Error getting voice characteristics: {e}")
            return {}
    
    def list_voices(self) -> List[Dict[str, Any]]:
        """List all available voices"""
        try:
            voices = []
            for voice_id, profile in self.voice_profiles.items():
                voices.append({
                    "id": voice_id,
                    "characteristics": profile.characteristics,
                    "quality_score": profile.quality_score
                })
            return voices
            
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            return []
    
    def _extract_voice_features(self, 
                              audio: np.ndarray, 
                              sample_rate: int) -> Optional[torch.Tensor]:
        """Extract voice features from audio"""
        try:
            # Simple feature extraction (in production, use mel-spectrograms, MFCCs, etc.)
            
            # Calculate basic spectral features
            fft = np.fft.fft(audio)
            magnitude_spectrum = np.abs(fft[:len(fft)//2])
            
            # Downsample to fixed size
            target_size = 80
            if len(magnitude_spectrum) > target_size:
                # Downsample
                indices = np.linspace(0, len(magnitude_spectrum)-1, target_size, dtype=int)
                features = magnitude_spectrum[indices]
            else:
                # Pad with zeros
                features = np.pad(magnitude_spectrum, (0, target_size - len(magnitude_spectrum)))
            
            # Normalize
            if np.max(features) > 0:
                features = features / np.max(features)
            
            return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error extracting voice features: {e}")
            return None
    
    def _analyze_voice_characteristics(self, 
                                     audio: np.ndarray, 
                                     sample_rate: int) -> Dict[str, Any]:
        """Analyze voice characteristics"""
        try:
            characteristics = {}
            
            # Estimate fundamental frequency (pitch)
            pitch = self._estimate_pitch(audio, sample_rate)
            characteristics["pitch"] = pitch
            
            # Analyze spectral characteristics
            spectral_features = self._analyze_spectral_features(audio, sample_rate)
            characteristics.update(spectral_features)
            
            # Estimate speaking rate
            speaking_rate = self._estimate_speaking_rate(audio, sample_rate)
            characteristics["speaking_rate"] = speaking_rate
            
            # Classify timbre (simplified)
            timbre = self._classify_timbre(audio, sample_rate)
            characteristics["timbre"] = timbre
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error analyzing voice characteristics: {e}")
            return {}
    
    def _estimate_pitch(self, audio: np.ndarray, sample_rate: int) -> float:
        """Estimate fundamental frequency"""
        try:
            # Simple autocorrelation-based pitch estimation
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peak in expected pitch range (80-400 Hz)
            min_period = int(sample_rate / 400)  # 400 Hz
            max_period = int(sample_rate / 80)   # 80 Hz
            
            if max_period < len(autocorr):
                peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
                pitch = sample_rate / peak_idx
                return float(pitch)
            
            return 200.0  # Default pitch
            
        except Exception as e:
            logger.error(f"Error estimating pitch: {e}")
            return 200.0
    
    def _analyze_spectral_features(self, 
                                 audio: np.ndarray, 
                                 sample_rate: int) -> Dict[str, Any]:
        """Analyze spectral characteristics"""
        try:
            # Calculate power spectrum
            fft = np.fft.fft(audio)
            power_spectrum = np.abs(fft) ** 2
            freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
            
            # Only use positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_power = power_spectrum[:len(power_spectrum)//2]
            
            # Calculate spectral centroid
            if np.sum(positive_power) > 0:
                spectral_centroid = np.sum(positive_freqs * positive_power) / np.sum(positive_power)
            else:
                spectral_centroid = 1000.0
            
            # Calculate spectral rolloff (frequency below which 85% of energy is contained)
            cumulative_power = np.cumsum(positive_power)
            total_power = cumulative_power[-1]
            rolloff_threshold = 0.85 * total_power
            
            rolloff_idx = np.where(cumulative_power >= rolloff_threshold)[0]
            spectral_rolloff = positive_freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 4000.0
            
            return {
                "spectral_centroid": float(spectral_centroid),
                "spectral_rolloff": float(spectral_rolloff)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing spectral features: {e}")
            return {"spectral_centroid": 1000.0, "spectral_rolloff": 4000.0}
    
    def _estimate_speaking_rate(self, audio: np.ndarray, sample_rate: int) -> float:
        """Estimate speaking rate (words per minute)"""
        try:
            # Simple energy-based segmentation to estimate syllables
            # Smooth the audio
            window_size = int(0.02 * sample_rate)  # 20ms window
            smoothed = np.convolve(np.abs(audio), np.ones(window_size)/window_size, mode='same')
            
            # Find peaks (potential syllables)
            threshold = 0.1 * np.max(smoothed)
            peaks = []
            
            for i in range(1, len(smoothed) - 1):
                if (smoothed[i] > smoothed[i-1] and 
                    smoothed[i] > smoothed[i+1] and 
                    smoothed[i] > threshold):
                    peaks.append(i)
            
            # Estimate speaking rate
            duration_seconds = len(audio) / sample_rate
            syllables_per_second = len(peaks) / duration_seconds if duration_seconds > 0 else 0
            
            # Rough conversion to words per minute (assuming ~1.5 syllables per word)
            words_per_minute = syllables_per_second * 60 / 1.5
            
            return float(words_per_minute)
            
        except Exception as e:
            logger.error(f"Error estimating speaking rate: {e}")
            return 150.0  # Default speaking rate
    
    def _classify_timbre(self, audio: np.ndarray, sample_rate: int) -> str:
        """Classify voice timbre"""
        try:
            # Simple timbre classification based on spectral features
            spectral_features = self._analyze_spectral_features(audio, sample_rate)
            
            centroid = spectral_features.get("spectral_centroid", 1000)
            rolloff = spectral_features.get("spectral_rolloff", 4000)
            
            # Simple classification rules
            if centroid > 1500 and rolloff > 5000:
                return "bright"
            elif centroid < 800 and rolloff < 3000:
                return "deep"
            elif 1000 <= centroid <= 1300:
                return "warm"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error classifying timbre: {e}")
            return "neutral"
    
    def _calculate_quality_score(self, 
                               audio: np.ndarray, 
                               characteristics: Dict[str, Any]) -> float:
        """Calculate voice quality score"""
        try:
            score = 1.0
            
            # Penalize for very short audio
            duration = len(audio) / 22050  # Assume 22050 Hz
            if duration < 1.0:
                score *= 0.5
            elif duration < 3.0:
                score *= 0.8
            
            # Check signal-to-noise ratio (rough estimate)
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 0.01:  # Very quiet
                score *= 0.3
            elif rms < 0.05:
                score *= 0.7
            
            # Check for clipping
            clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
            if clipping_ratio > 0.01:  # More than 1% clipped
                score *= 0.5
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5
