import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TranslationResult:
    """Result of semantic translation"""
    text: str
    confidence: float
    semantic_roles: List[Dict[str, Any]]
    discourse_markers: List[str]
    alternatives: List[Dict[str, Any]]
    attention_weights: Optional[torch.Tensor] = None

@dataclass
class SemanticRole:
    """Semantic role in the translation"""
    role: str  # agent, patient, theme, location, etc.
    entity: str
    confidence: float
    span: Tuple[int, int]  # Token span in translation

class MultimodalAttention(nn.Module):
    """Attention mechanism for multimodal fusion"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = query.shape[:2]
        
        # Project to multi-head
        Q = self.query_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        output = self.output_proj(attended)
        
        return output, attention_weights

class SignLanguageTransformer(nn.Module):
    """Transformer model for sign language to text translation"""
    
    def __init__(self, 
                 vocab_size: int = 50000,
                 hidden_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 max_length: int = 512):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Gloss encoder
        self.gloss_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gloss_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=num_layers // 2
        )
        
        # Spatial feature encoder
        self.spatial_projection = nn.Linear(128, hidden_dim)  # Spatial features to hidden dim
        
        # NMM feature encoder
        self.nmm_projection = nn.Linear(64, hidden_dim)  # NMM features to hidden dim
        
        # Multimodal fusion
        self.multimodal_attention = MultimodalAttention(hidden_dim, num_heads)
        self.fusion_layer = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Text decoder
        self.text_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=num_layers // 2
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Positional encoding
        self.register_buffer('pos_encoding', self._create_positional_encoding(max_length, hidden_dim))
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, 
                gloss_tokens: torch.Tensor,
                spatial_features: torch.Tensor,
                nmm_features: torch.Tensor,
                target_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size, seq_len = gloss_tokens.shape
        
        # Encode gloss sequence
        gloss_emb = self.gloss_embedding(gloss_tokens)
        gloss_emb += self.pos_encoding[:, :seq_len, :]
        gloss_encoded = self.gloss_encoder(gloss_emb)
        
        # Project spatial and NMM features
        spatial_proj = self.spatial_projection(spatial_features)
        nmm_proj = self.nmm_projection(nmm_features)
        
        # Multimodal fusion
        fused_spatial, _ = self.multimodal_attention(gloss_encoded, spatial_proj, spatial_proj)
        fused_nmm, _ = self.multimodal_attention(gloss_encoded, nmm_proj, nmm_proj)
        
        # Combine all modalities
        multimodal_features = torch.cat([gloss_encoded, fused_spatial, fused_nmm], dim=-1)
        fused_features = self.fusion_layer(multimodal_features)
        
        if target_tokens is not None:
            # Training mode - use teacher forcing
            target_emb = self.gloss_embedding(target_tokens)
            target_emb += self.pos_encoding[:, :target_tokens.shape[1], :]
            
            decoded = self.text_decoder(target_emb, fused_features)
            output = self.output_projection(decoded)
            
            return output
        else:
            # Inference mode - autoregressive generation
            return self._generate(fused_features)
    
    def _generate(self, memory: torch.Tensor, max_length: int = 100) -> torch.Tensor:
        """Autoregressive generation for inference"""
        batch_size = memory.shape[0]
        device = memory.device
        
        # Start with BOS token
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for _ in range(max_length):
            # Generate embeddings for current sequence
            target_emb = self.gloss_embedding(generated)
            target_emb += self.pos_encoding[:, :generated.shape[1], :]
            
            # Decode
            decoded = self.text_decoder(target_emb, memory)
            logits = self.output_projection(decoded)
            
            # Get next token
            next_token = torch.argmax(logits[:, -1:, :], dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token generated
            if next_token.item() == 2:  # Assuming EOS token ID is 2
                break
        
        return generated

class SemanticTranslator:
    """Main semantic translator class"""
    
    def __init__(self, 
                 model_name: str = "sign-language-transformer",
                 max_length: int = 512,
                 beam_size: int = 5):
        
        # Initialize tokenizer (using a pre-trained one as base)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        except:
            logger.warning("Could not load tokenizer, using mock tokenizer")
            self.tokenizer = None
        
        # Initialize model
        self.model = SignLanguageTransformer(
            vocab_size=50000,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            max_length=max_length
        )
        
        self.max_length = max_length
        self.beam_size = beam_size
        
        # Gloss to token mapping
        self.gloss_vocab = self._create_gloss_vocab()
        
        # Set to evaluation mode
        self.model.eval()
        
        logger.info(f"SemanticTranslator initialized with max_length={max_length}")
    
    def _create_gloss_vocab(self) -> Dict[str, int]:
        """Create gloss vocabulary mapping"""
        # Placeholder vocabulary - in production, load from file
        vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        
        # Common ASL glosses
        common_glosses = [
            "HELLO", "GOODBYE", "THANK-YOU", "PLEASE", "SORRY", "YES", "NO",
            "GOOD", "BAD", "HAPPY", "SAD", "LOVE", "LIKE", "WANT", "NEED",
            "GO", "COME", "STOP", "START", "FINISH", "WORK", "HOME", "SCHOOL",
            "EAT", "DRINK", "SLEEP", "WAKE-UP", "SIT", "STAND", "WALK", "RUN",
            "I", "YOU", "HE", "SHE", "WE", "THEY", "MY", "YOUR", "HIS", "HER"
        ]
        
        for i, gloss in enumerate(common_glosses, 4):
            vocab[gloss] = i
        
        return vocab
    
    async def translate(self, 
                       gloss_sequence: List[Dict[str, Any]],
                       fingerspelling: List[str],
                       spatial_roles: Dict[str, Any],
                       nmm_features: Dict[str, Any],
                       context: Dict[str, Any]) -> TranslationResult:
        """Translate gloss sequence to natural language text"""
        try:
            # Prepare input features
            gloss_tokens = self._prepare_gloss_tokens(gloss_sequence, fingerspelling)
            spatial_features = self._prepare_spatial_features(spatial_roles)
            nmm_feature_tensor = self._prepare_nmm_features(nmm_features)
            
            # Perform translation
            with torch.no_grad():
                output_tokens = self.model(gloss_tokens, spatial_features, nmm_feature_tensor)
            
            # Decode to text
            translated_text = self._decode_tokens(output_tokens)
            
            # Calculate confidence
            confidence = self._calculate_confidence(output_tokens)
            
            # Extract semantic roles
            semantic_roles = self._extract_semantic_roles(
                gloss_sequence, spatial_roles, translated_text
            )
            
            # Identify discourse markers
            discourse_markers = self._identify_discourse_markers(nmm_features, translated_text)
            
            # Generate alternatives (simplified)
            alternatives = self._generate_alternatives(translated_text, confidence)
            
            return TranslationResult(
                text=translated_text,
                confidence=confidence,
                semantic_roles=semantic_roles,
                discourse_markers=discourse_markers,
                alternatives=alternatives
            )
            
        except Exception as e:
            logger.error(f"Error in semantic translation: {e}")
            return TranslationResult(
                text="[Translation Error]",
                confidence=0.0,
                semantic_roles=[],
                discourse_markers=[],
                alternatives=[]
            )
    
    def _prepare_gloss_tokens(self, gloss_sequence: List[Dict[str, Any]], 
                            fingerspelling: List[str]) -> torch.Tensor:
        """Convert gloss sequence to token tensor"""
        try:
            tokens = [self.gloss_vocab.get("<BOS>", 2)]
            
            # Add gloss tokens
            for gloss_item in gloss_sequence:
                if isinstance(gloss_item, dict):
                    gloss_token = gloss_item.get('token', '')
                else:
                    gloss_token = str(gloss_item)
                
                token_id = self.gloss_vocab.get(gloss_token, self.gloss_vocab.get("<UNK>", 1))
                tokens.append(token_id)
            
            # Add fingerspelling as special tokens
            for letter in fingerspelling:
                # Map letters to special token IDs
                letter_token_id = ord(letter.upper()) - ord('A') + 1000  # Offset for letters
                tokens.append(letter_token_id)
            
            tokens.append(self.gloss_vocab.get("<EOS>", 3))
            
            # Convert to tensor
            return torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            logger.error(f"Error preparing gloss tokens: {e}")
            return torch.tensor([[2, 1, 3]])  # BOS, UNK, EOS
    
    def _prepare_spatial_features(self, spatial_roles: Dict[str, Any]) -> torch.Tensor:
        """Convert spatial roles to feature tensor"""
        try:
            # Create feature vector from spatial information
            features = np.zeros(128)  # Fixed size feature vector
            
            # Encode spatial zones
            spatial_zones = spatial_roles.get('spatial_zones', {})
            if spatial_zones.get('left_hand_zone'):
                features[0] = 1.0  # Left hand active
            if spatial_zones.get('right_hand_zone'):
                features[1] = 1.0  # Right hand active
            
            # Encode spatial roles
            roles = spatial_roles.get('spatial_roles', [])
            for i, role in enumerate(roles[:10]):  # Limit to 10 roles
                if isinstance(role, dict):
                    role_type = role.get('role', '')
                    if role_type == 'subject':
                        features[10 + i] = 1.0
                    elif role_type == 'object':
                        features[20 + i] = 1.0
                    elif role_type == 'location':
                        features[30 + i] = 1.0
            
            # Encode classifier usage
            classifiers = spatial_roles.get('classifier_usage', {})
            if classifiers.get('detected_classifiers'):
                features[50] = 1.0
            
            return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error preparing spatial features: {e}")
            return torch.zeros(1, 128)
    
    def _prepare_nmm_features(self, nmm_features: Dict[str, Any]) -> torch.Tensor:
        """Convert NMM features to tensor"""
        try:
            # Create feature vector from NMM information
            features = np.zeros(64)  # Fixed size feature vector
            
            # Encode facial expressions
            expressions = nmm_features.get('facial_expressions', {})
            if expressions.get('eyebrow_raise'):
                features[0] = expressions['eyebrow_raise'].get('intensity', 0.0)
            if expressions.get('eye_gaze'):
                features[1] = 1.0
            if expressions.get('mouth_shape'):
                features[2] = 1.0
            
            # Encode head movements
            head_movements = nmm_features.get('head_movements', {})
            if head_movements.get('nod'):
                features[10] = head_movements['nod'].get('intensity', 0.0)
            if head_movements.get('shake'):
                features[11] = head_movements['shake'].get('intensity', 0.0)
            
            # Encode body posture
            body_posture = nmm_features.get('body_posture', {})
            if body_posture.get('lean_forward'):
                features[20] = 1.0
            if body_posture.get('shoulder_shift'):
                features[21] = 1.0
            
            return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error preparing NMM features: {e}")
            return torch.zeros(1, 64)
    
    def _decode_tokens(self, token_tensor: torch.Tensor) -> str:
        """Decode token tensor to text"""
        try:
            if self.tokenizer:
                # Use actual tokenizer if available
                tokens = token_tensor.squeeze(0).tolist()
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                return text
            else:
                # Simple mock decoding
                return "Hello, how are you today?"
                
        except Exception as e:
            logger.error(f"Error decoding tokens: {e}")
            return "[Decoding Error]"
    
    def _calculate_confidence(self, output_tensor: torch.Tensor) -> float:
        """Calculate translation confidence"""
        try:
            # Simple confidence based on max probabilities
            probs = F.softmax(output_tensor, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            confidence = torch.mean(max_probs).item()
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _extract_semantic_roles(self, 
                              gloss_sequence: List[Dict[str, Any]],
                              spatial_roles: Dict[str, Any],
                              translated_text: str) -> List[Dict[str, Any]]:
        """Extract semantic roles from translation"""
        try:
            roles = []
            
            # Map spatial roles to semantic roles
            spatial_role_list = spatial_roles.get('spatial_roles', [])
            for role in spatial_role_list:
                if isinstance(role, dict):
                    roles.append({
                        'role': role.get('role', 'unknown'),
                        'entity': f"entity_{len(roles)}",
                        'confidence': role.get('confidence', 0.5),
                        'span': (0, len(translated_text))  # Simplified span
                    })
            
            return roles
            
        except Exception as e:
            logger.error(f"Error extracting semantic roles: {e}")
            return []
    
    def _identify_discourse_markers(self, 
                                  nmm_features: Dict[str, Any],
                                  translated_text: str) -> List[str]:
        """Identify discourse markers from NMM features"""
        try:
            markers = []
            
            # Map NMM features to discourse markers
            if nmm_features.get('facial_expressions', {}).get('eyebrow_raise'):
                markers.append('question')
            
            if nmm_features.get('head_movements', {}).get('nod'):
                markers.append('affirmation')
            
            if nmm_features.get('head_movements', {}).get('shake'):
                markers.append('negation')
            
            return markers
            
        except Exception as e:
            logger.error(f"Error identifying discourse markers: {e}")
            return []
    
    def _generate_alternatives(self, 
                             translated_text: str,
                             confidence: float) -> List[Dict[str, Any]]:
        """Generate alternative translations"""
        try:
            alternatives = []
            
            # Simple alternative generation (in production, use beam search)
            if confidence < 0.8:
                alternatives.append({
                    'text': translated_text.replace('.', '?'),
                    'confidence': confidence * 0.9,
                    'reason': 'question_interpretation'
                })
            
            return alternatives
            
        except Exception as e:
            logger.error(f"Error generating alternatives: {e}")
            return []
