
import torch
import torch.nn as nn
from transformers import SiglipModel, AutoTokenizer, SiglipConfig
import logging

logger = logging.getLogger("IRRA.model")

class MSigLIP(nn.Module):
    def __init__(self, model_name: str, device='cpu', **kwargs):
        super().__init__()
        logger.info(f"Loading Google SigLIP model: {model_name}")
        
        config = SiglipConfig.from_pretrained(model_name)
        # Ép buộc output attention
        config.output_attentions = True
        config.vision_config.output_attentions = True
        config.text_config.output_attentions = True
        
        self.hf_model = SiglipModel.from_pretrained(model_name, config=config)
        self.hf_model.train()
        self.hf_model.gradient_checkpointing_enable()

        self.vocab_size = config.text_config.vocab_size
        
        self.embed_dim = config.vision_config.hidden_size 
        self.context_length = config.text_config.max_position_embeddings
        self.logit_scale = self.hf_model.logit_scale
        if hasattr(self.hf_model, "logit_bias"):
            self.logit_bias = self.hf_model.logit_bias

    @property
    def dtype(self):
        return self.hf_model.vision_model.embeddings.patch_embedding.weight.dtype

    def encode_image(self, image):
        image = image.to(dtype=self.dtype)
        
        # 1. Forward model
        outputs = self.hf_model.vision_model(
            pixel_values=image, 
            interpolate_pos_encoding=True 
        )
        
        pooled = outputs.pooler_output      # [Batch, Dim]
        patches = outputs.last_hidden_state # [Batch, 192, Dim]
        
        # 2. Fake CLS Feature (Nối pooled vào đầu chuỗi)
        # x shape: [Batch, 193, Dim]
        x = torch.cat([pooled.unsqueeze(1), patches], dim=1)
        
        # 3. FIX LỖI 4D -> 3D ATTENTION
        # Raw attention: [Batch, Heads, 192, 192]
        raw_atten = outputs.attentions[-1]
        
        # Bước quan trọng: Gộp các Heads lại bằng cách lấy trung bình
        # New shape: [Batch, 192, 192]
        raw_atten_mean = raw_atten.mean(dim=1) 
        
        bs, seq_len, _ = raw_atten_mean.shape
        new_seq_len = seq_len + 1 # 193
        
        # Tạo Fake Attention Map 3D: [Batch, 193, 193]
        fake_atten = torch.zeros(
            (bs, new_seq_len, new_seq_len),
            dtype=raw_atten.dtype,
            device=raw_atten.device
        )
        
        # A. Copy attention nội bộ các patch
        fake_atten[:, 1:, 1:] = raw_atten_mean
        
        # B. Smart CLS Attention
        # Tính độ quan trọng của patch dựa trên việc các patch khác nhìn vào nó bao nhiêu
        # dim=1 ở đây là dimension "Queries" (các patch khác), ta mean để xem patch cột j nhận bao nhiêu sự chú ý
        patch_importance = raw_atten_mean.mean(dim=1) # [Batch, 192]
        
        # Chuẩn hóa (Softmax nhẹ)
        patch_importance = patch_importance / (patch_importance.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Gán vào hàng 0 (CLS nhìn các patch)
        fake_atten[:, 0, 1:] = patch_importance
        
        # CLS nhìn chính nó
        fake_atten[:, 0, 0]  = 1.0 
        
        # Các patch nhìn CLS (Uniform)
        fake_atten[:, 1:, 0] = 1.0 / seq_len

        return x.float(), fake_atten.float()

    def encode_text(self, text):
        text = torch.clamp(text, min=0, max=self.vocab_size - 1)
        # Forward Text Model
        outputs = self.hf_model.text_model(input_ids=text)
        
        # [QUAN TRỌNG] Lấy Feature gốc (EOS token) do HuggingFace tính sẵn
        # Đây là feature chuẩn nhất để tính Loss, không lo tìm sai index
        pooled = outputs.pooler_output # [Batch, Dim]
        sequence = outputs.last_hidden_state # [Batch, Seq_Len, Dim]
        
        # --- 1. FAKE CLS TOKEN cho TEXT ---
        # Nối pooled vào đầu chuỗi: [Batch, 1 + Seq_Len, Dim]
        x = torch.cat([pooled.unsqueeze(1), sequence], dim=1)
        
        # --- 2. XỬ LÝ ATTENTION MAP (4D -> 3D) ---
        raw_atten = outputs.attentions[-1] # [Batch, Heads, Seq_Len, Seq_Len]
        
        # Gộp các Heads (Average) -> [Batch, Seq_Len, Seq_Len]
        raw_atten_mean = raw_atten.mean(dim=1)
        
        bs, seq_len, _ = raw_atten_mean.shape
        new_seq_len = seq_len + 1
        
        # Tạo Fake Attention Map 3D: [Batch, New_Seq_Len, New_Seq_Len]
        fake_atten = torch.zeros(
            (bs, new_seq_len, new_seq_len),
            dtype=raw_atten.dtype,
            device=raw_atten.device
        )
        
        # Copy attention nội bộ
        fake_atten[:, 1:, 1:] = raw_atten_mean
        
        # Fake CLS Attention (Uniform hoặc Smart)
        # Text: EOS token (cuối) thường quan trọng, nhưng để đơn giản ta dùng Uniform 
        # để module TSE tự học cách lọc từ.
        fake_atten[:, 0, 1:] = 1.0 / seq_len # CLS nhìn các từ
        fake_atten[:, 0, 0]  = 1.0           # CLS nhìn chính nó
        fake_atten[:, 1:, 0] = 1.0 / seq_len # Các từ nhìn CLS
        
        # [QUAN TRỌNG] Ép kiểu float32 trước khi trả về
        return x.float(), fake_atten.float()

    def forward(self, image, text):
        image_feats, atten_i = self.encode_image(image)
        text_feats, atten_t = self.encode_text(text)
        return image_feats, atten_i, text_feats, atten_t

    def load_param(self, state_dict):
        # Hàm này để tương thích code cũ gọi load_param
        # Với HF model, ta thường load trực tiếp từ .from_pretrained nên có thể bỏ qua
        # hoặc implement logic load custom nếu bạn finetune rồi lưu state_dict riêng.
        msg = self.hf_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded params with message: {msg}")
def convert_weights(model: nn.Module):
    """
    Giữ nguyên interface convert_weights.
    Với HuggingFace model, weights thường đã load đúng. 
    Bạn có thể dùng mixed-precision (AMP) khi training thay vì convert thủ công.
    """
    pass
def build_CLIP_from_openai_pretrained(name: str = "google/siglip-base-patch16-256-multilingual", 
                                  image_size=256, stride_size=16):
    """
    Factory function thay thế hàm cũ
    """
    model = MSigLIP(model_name=name)
    
    # Tạo dummy config để code bên ngoài không bị lỗi khi truy cập config
    model_cfg = {
        'embed_dim': model.embed_dim,
        'image_resolution': image_size,
        'context_length': model.context_length,
        'stride_size': stride_size
    }
    
    return model, model_cfg