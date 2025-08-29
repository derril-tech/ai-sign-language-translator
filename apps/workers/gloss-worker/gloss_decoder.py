import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Gloss:
    """Individual gloss token with confidence"""
    token: str
    confidence: float
    start_time: float
    end_time: float
    features: Optional[torch.Tensor] = None

@dataclass
class GlossSequence:
    """Sequence of gloss tokens"""
    glosses: List[Gloss]
    confidence: float
    sequence_length: int
    
    def to_dict(self) -> List[Dict]:
        """Convert to dictionary format"""
        return [
            {
                "token": gloss.token,
                "confidence": gloss.confidence,
                "start_time": gloss.start_time,
                "end_time": gloss.end_time
            }
            for gloss in self.glosses
        ]

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class GlossTransformer(nn.Module):
    """Transformer model for gloss sequence decoding"""
    
    def __init__(self, 
                 input_dim: int = 300,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 vocab_size: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection for CTC
        self.output_projection = nn.Linear(hidden_dim, vocab_size + 1)  # +1 for blank token
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Output projection
        logits = self.output_projection(x)
        
        return F.log_softmax(logits, dim=-1)

class CTCDecoder:
    """CTC decoder for gloss sequences"""
    
    def __init__(self, vocab: Dict[int, str], blank_token: int = 0):
        self.vocab = vocab
        self.blank_token = blank_token
        
    def decode_greedy(self, log_probs: torch.Tensor) -> Tuple[List[int], List[float]]:
        """Greedy CTC decoding"""
        # Get most likely tokens
        tokens = torch.argmax(log_probs, dim=-1)
        confidences = torch.max(torch.exp(log_probs), dim=-1)[0]
        
        # Remove blanks and consecutive duplicates
        decoded_tokens = []
        decoded_confidences = []
        prev_token = None
        
        for i, (token, conf) in enumerate(zip(tokens[0], confidences[0])):
            token_id = token.item()
            conf_val = conf.item()
            
            if token_id != self.blank_token and token_id != prev_token:
                decoded_tokens.append(token_id)
                decoded_confidences.append(conf_val)
            
            prev_token = token_id
        
        return decoded_tokens, decoded_confidences
    
    def decode_beam_search(self, log_probs: torch.Tensor, beam_size: int = 5) -> List[Tuple[List[int], float]]:
        """Beam search CTC decoding"""
        # Simplified beam search implementation
        seq_len, vocab_size = log_probs.shape[1], log_probs.shape[2]
        
        # Initialize beam
        beam = [([self.blank_token], 0.0)]
        
        for t in range(seq_len):
            candidates = []
            
            for sequence, score in beam:
                for token_id in range(vocab_size):
                    new_score = score + log_probs[0, t, token_id].item()
                    
                    if token_id == self.blank_token:
                        # Blank token - no change to sequence
                        candidates.append((sequence, new_score))
                    elif len(sequence) == 0 or token_id != sequence[-1]:
                        # New token
                        candidates.append((sequence + [token_id], new_score))
                    else:
                        # Repeated token
                        candidates.append((sequence, new_score))
            
            # Keep top beam_size candidates
            beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        
        return beam

class GlossDecoder:
    """Main gloss decoder class"""
    
    def __init__(self, 
                 vocab_size: int = 5000,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 dropout: float = 0.1):
        
        # Load vocabulary (placeholder)
        self.vocab = self._load_vocabulary(vocab_size)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Initialize model
        self.model = GlossTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # CTC decoder
        self.ctc_decoder = CTCDecoder(self.vocab)
        
        # Set to evaluation mode
        self.model.eval()
        
        logger.info(f"GlossDecoder initialized with vocab_size={vocab_size}")
    
    def _load_vocabulary(self, vocab_size: int) -> Dict[int, str]:
        """Load gloss vocabulary"""
        # Placeholder vocabulary - in production, load from file
        vocab = {0: "<BLANK>"}
        
        # Common ASL glosses
        common_glosses = [
            "HELLO", "GOODBYE", "THANK-YOU", "PLEASE", "SORRY", "YES", "NO",
            "GOOD", "BAD", "HAPPY", "SAD", "LOVE", "LIKE", "WANT", "NEED",
            "GO", "COME", "STOP", "START", "FINISH", "WORK", "HOME", "SCHOOL",
            "EAT", "DRINK", "SLEEP", "WAKE-UP", "SIT", "STAND", "WALK", "RUN",
            "RED", "BLUE", "GREEN", "YELLOW", "BLACK", "WHITE", "BIG", "SMALL",
            "HOT", "COLD", "NEW", "OLD", "FAST", "SLOW", "EASY", "HARD",
            "I", "YOU", "HE", "SHE", "WE", "THEY", "MY", "YOUR", "HIS", "HER"
        ]
        
        for i, gloss in enumerate(common_glosses[:vocab_size-1], 1):
            vocab[i] = gloss
        
        # Fill remaining with placeholder glosses
        for i in range(len(common_glosses) + 1, vocab_size):
            vocab[i] = f"GLOSS_{i}"
        
        return vocab
    
    async def decode_sequence(self, features: torch.Tensor) -> Optional[GlossSequence]:
        """Decode gloss sequence from pose features"""
        try:
            with torch.no_grad():
                # Forward pass through model
                log_probs = self.model(features)
                
                # CTC decoding
                tokens, confidences = self.ctc_decoder.decode_greedy(log_probs)
                
                if not tokens:
                    return None
                
                # Convert to gloss sequence
                glosses = []
                sequence_length = len(tokens)
                
                for i, (token_id, confidence) in enumerate(zip(tokens, confidences)):
                    if token_id in self.vocab:
                        gloss = Gloss(
                            token=self.vocab[token_id],
                            confidence=confidence,
                            start_time=i * 0.033,  # Assuming 30fps
                            end_time=(i + 1) * 0.033
                        )
                        glosses.append(gloss)
                
                if glosses:
                    overall_confidence = np.mean([g.confidence for g in glosses])
                    return GlossSequence(
                        glosses=glosses,
                        confidence=overall_confidence,
                        sequence_length=sequence_length
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Error in gloss decoding: {e}")
            return None
    
    def load_state_dict(self, state_dict: Dict):
        """Load model weights"""
        self.model.load_state_dict(state_dict)
        logger.info("Loaded gloss decoder weights")
    
    def save_state_dict(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved gloss decoder weights to {path}")
