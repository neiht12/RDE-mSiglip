
import torch
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipTextModel, SiglipConfig
import logging

logger = logging.getLogger("IRRA.model")

class MSigLIP(nn.Module):
    def __init__(self, model_name: str, device='cpu', **kwargs):
        super().__init__()
        logger.info(f"Loading Google SigLIP model components separately: {model_name}")
        
        # 1. Load Config gốc
        config = SiglipConfig.from_pretrained(model_name)
        
        # 2. Setup Vision Model 
        vision_config = config.vision_config
        vision_config.attn_implementation = "eager"
        vision_config.output_attentions = True
        
        self.vision_model = SiglipVisionModel.from_pretrained(
            model_name,
            config=vision_config,
            attn_implementation="eager", 
            ignore_mismatched_sizes=True
        )
        
        # 3. Setup Text Model 
        text_config = config.text_config
        text_config.attn_implementation = "eager"
        text_config.output_attentions = True
        
        self.text_model = SiglipTextModel.from_pretrained(
            model_name,
            config=text_config,
            attn_implementation="eager",
            ignore_mismatched_sizes=True
        )
        
        self.vision_model.train()
        self.text_model.train()

        # 4. Lấy các tham số cần thiết
        self.vocab_size = config.text_config.vocab_size
        self.embed_dim = config.vision_config.hidden_size 
        self.context_length = config.text_config.max_position_embeddings
        
        # Setup logit scale
        self.logit_scale = nn.Parameter(torch.tensor(4.6052))
        if hasattr(self.vision_model, "logit_scale"):
             self.logit_scale = self.vision_model.logit_scale
             
        self.logit_bias = None
        if hasattr(self.vision_model, "logit_bias"):
            self.logit_bias = self.vision_model.logit_bias

        logger.info("Successfully loaded components with Eager Attention.")

    def encode_image(self, image):
        
        target_dtype = self.vision_model.dtype
        image = image.to(dtype=target_dtype)
        
        outputs = self.vision_model(
            pixel_values=image,
            output_attentions=True,
            return_dict=True,
            interpolate_pos_encoding=True 
        )
        
        if outputs.attentions is None:
            raise ValueError("Vision model returned attentions=None. 'eager' implementation failed.")

        pooled = outputs.pooler_output      # [Batch, Dim]
        patches = outputs.last_hidden_state # [Batch, Seq_Len, Dim]
        
        # Fake CLS Feature
        x = torch.cat([pooled.unsqueeze(1), patches], dim=1)
        
        # Get attention weights
        raw_atten = outputs.attentions[-1]  # [Batch, Heads, Seq_Len, Seq_Len]
        
        # Average over heads
        raw_atten_mean = raw_atten.mean(dim=1) 
        
        bs, seq_len, _ = raw_atten_mean.shape
        new_seq_len = seq_len + 1 
        
        fake_atten = torch.zeros(
            (bs, new_seq_len, new_seq_len),
            dtype=raw_atten.dtype,
            device=raw_atten.device
        )
        
        fake_atten[:, 1:, 1:] = raw_atten_mean
        
        # Smart CLS Attention
        patch_importance = raw_atten_mean.mean(dim=1)
        patch_importance = patch_importance / (patch_importance.sum(dim=-1, keepdim=True) + 1e-6)
        
        fake_atten[:, 0, 1:] = patch_importance
        fake_atten[:, 0, 0]  = 1.0 
        fake_atten[:, 1:, 0] = 1.0 / seq_len

        return x.float(), fake_atten.float()

    def encode_text(self, text):
        text = torch.clamp(text, min=0, max=self.vocab_size - 1)
        
        outputs = self.text_model(
            input_ids=text,
            output_attentions=True,
            return_dict=True
        )
        
        if outputs.attentions is None:
             raise ValueError("Text model returned attentions=None.")

        pooled = outputs.pooler_output 
        sequence = outputs.last_hidden_state
        
        # Fake CLS
        x = torch.cat([pooled.unsqueeze(1), sequence], dim=1)
        
        # Process Attention
        raw_atten = outputs.attentions[-1]
        raw_atten_mean = raw_atten.mean(dim=1)
        
        bs, seq_len, _ = raw_atten_mean.shape
        new_seq_len = seq_len + 1
        
        fake_atten = torch.zeros(
            (bs, new_seq_len, new_seq_len),
            dtype=raw_atten.dtype,
            device=raw_atten.device
        )
        
        fake_atten[:, 1:, 1:] = raw_atten_mean
        fake_atten[:, 0, 1:] = 1.0 / seq_len
        fake_atten[:, 0, 0]  = 1.0
        fake_atten[:, 1:, 0] = 1.0 / seq_len
        
        return x.float(), fake_atten.float()

    def forward(self, image, text):
        image_feats, atten_i = self.encode_image(image)
        text_feats, atten_t = self.encode_text(text)
        return image_feats, atten_i, text_feats, atten_t

    def load_param(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("hf_model.vision_model"):
                new_key = k.replace("hf_model.vision_model", "vision_model")
                new_state_dict[new_key] = v
            elif k.startswith("hf_model.text_model"):
                new_key = k.replace("hf_model.text_model", "text_model")
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        msg = self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded params (remapped) with message: {msg}")

def build_CLIP_from_openai_pretrained(name: str = "google/siglip-base-patch16-256-multilingual", 
                                  image_size=256, stride_size=16):
    model = MSigLIP(model_name=name)
    
    model_cfg = {
        'embed_dim': model.embed_dim,
        'image_resolution': image_size,
        'context_length': model.context_length,
        'stride_size': stride_size
    }
    
    return model, model_cfg
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    pass