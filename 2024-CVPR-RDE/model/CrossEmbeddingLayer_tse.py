
import torch
import torch.nn as nn
import torch.nn.functional as F

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def maxk_pool1d_var(x, dim, k, lengths):
    """https://github.com/woodfrog/vse_infty, thanks!"""
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length) if length > 0 else 1 # Safety check
        # Fix index slicing error if length is small
        curr_len = min(length, x.shape[1])
        if curr_len == 0:
            results.append(x[idx, 0, :]) # Fallback
        else:
            max_k_i = maxk(x[idx, :curr_len, :], dim - 1, k).mean(dim - 1)
            results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results

def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)

def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x

class TexualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=768, embed_dim=1024, ratio=0.3):
        super(TexualEmbeddingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.ratio = ratio

    def forward(self, features, text, atten):
        # features: [Batch, 65, 768] (Đã có CLS ở index 0)
        # text: [Batch, 64] (Raw tokens)
        # atten: [Batch, 65, 65]
        
        bs = features.size(0)
        pad_id = 1 
        
        # 1. Tạo Mask từ text gốc [Batch, 64]
        text_mask = (text != pad_id).float() 
        
        # 2. [FIX LỖI SIZE] Thêm Mask cho CLS token ở vị trí 0
        # Tạo cột 1 [Batch, 1] nối vào đầu text_mask -> mask mới [Batch, 65]
        cls_mask = torch.ones((bs, 1), device=text.device)
        mask = torch.cat([cls_mask, text_mask], dim=1) # Size: 65
        
        # 3. Tính lại lengths trên mask mới (đã bao gồm CLS)
        lengths = mask.sum(1).long()
        
        # 4. Tìm vị trí EOS (Token cuối cùng hợp lệ)
        # Vì mask đã tính cả CLS, vị trí cuối cùng là lengths - 1
        eos_indices = (lengths - 1).clamp(min=0)
        
        k = int((atten.size(1) - 2) * self.ratio)
        if k < 1: k = 1
        
        # Clone atten
        atten = atten.clone()
        
        # Masking logic cũ nhưng áp dụng trên size 65
        # Mask EOS (cột)
        atten[torch.arange(bs), :, eos_indices] = -1 
        # Mask SOS/CLS (cột 0)
        atten[torch.arange(bs), :, 0] = -1 
        
        # Lấy row của EOS token
        atten = atten[torch.arange(bs), eos_indices, :] # [Batch, 65]
        
        # 5. Nhân Mask (Giờ cả 2 đều size 65 -> KHỚP!)
        atten = atten * mask
        
        # Top-K pooling
        atten_topK = atten.topk(dim=-1, k=k)[1].unsqueeze(-1).expand(bs, k, features.size(2)) 
        features = torch.gather(input=features, dim=1, index=atten_topK) 
        features = l2norm(features, dim=-1)

        # Tính lengths cho maxk_pool
        # Trừ đi 2 (CLS và EOS) để lấy độ dài thực tế của nội dung
        pool_lengths = [l.item() - 2 if l.item() - 2 > 0 else 1 for l in lengths]
        pool_lengths = torch.Tensor([min(pl, k) for pl in pool_lengths]).to(features.device)
        
        cap_emb = self.linear(features) 
        features = self.mlp(features) + cap_emb
        features = maxk_pool1d_var(features, 1, 1, pool_lengths) 
        
        return features.float()

class VisualEmbeddingLayer(nn.Module):
    # CHANGE: input_dim default 512 -> 768 (SigLIP)
    def __init__(self, input_dim=768, embed_dim=1024, ratio=0.3):
        super(VisualEmbeddingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.ratio = ratio
        
        # --- FIX LỖI: Bỏ .half() ---
        # Để layer tự động theo kiểu dữ liệu input (float32)
        self.fc = nn.Linear(input_dim, embed_dim) 
        
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
    
    def forward(self, base_features, atten):
        # base_features: [Batch, 193, 768] (từ SigLIP wrapper)
        # atten: [Batch, 193, 193] (từ SigLIP wrapper)
        
        # Đảm bảo input là float32 để tránh lỗi matmul
        base_features = base_features.float()
        atten = atten.float()

        k = int((atten.size(1) - 1) * self.ratio) # Trừ đi CLS token
        if k < 1: k = 1
        
        bs = base_features.size(0)
        
        # Clone để an toàn
        atten = atten.clone()
        
        # Mask CLS token (cột 0) để không chọn nó vào top-K local features
        atten[torch.arange(bs), :, 0] = -1 
        
        # Lấy attention của CLS (hàng 0) nhìn các patch
        atten_cls = atten[:, 0, :] # [Batch, Seq_Len]
        
        atten_topK = atten_cls.topk(dim=-1, k=k)[1]
        
        atten_topK = atten_topK.unsqueeze(-1).expand(bs, k, base_features.size(2)) 
        
        # Gather features
        base_features = torch.gather(input=base_features, dim=1, index=atten_topK) 
        base_features = l2norm(base_features, dim=-1) 
        
        # Prepare lengths
        feat_lengths = torch.full((bs,), base_features.size(1), device=base_features.device)
        
        features = self.fc(base_features)
        features = self.mlp(base_features) + features 
        features = maxk_pool1d_var(features, 1, 1, feat_lengths) 

        return features.float()