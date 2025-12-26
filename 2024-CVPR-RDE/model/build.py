
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from model import objectives
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights 
import torch
import torch.nn as nn 
import torch.nn.functional as F
from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer

class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        # [SỬA 1] Dùng hàm builder của SigLIP
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
 
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)
 
        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
 
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        image_feats, atten_i = self.base_model.encode_image(image)
        # [SỬA 2] Lấy index 0 (Fake CLS) -> Luôn đúng
        return image_feats[:, 0, :].float()
      
    def encode_text(self, text):
        text_feats, atten_t = self.base_model.encode_text(text)
        # [SỬA 3] Lấy index 0 (Fake CLS / Pooler Output) -> Luôn đúng, không cần tìm argmax hay eos
        return text_feats[:, 0, :].float()

    def encode_image_tse(self, image):
        x, atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return i_tse_f.float()
 
    def encode_text_tse(self, text):
        x, atten_t = self.base_model.encode_text(text)
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        
        # Forward model
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        
        # [SỬA 4] Lấy Feature Global tại Index 0 (Thay vì logic cũ)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[:, 0, :].float() 
        
        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB

    def forward(self, batch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        
        # [SỬA 5] Sửa tương tự trong hàm forward
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[:, 0, :].float() # <--- Lấy thẳng index 0
        
        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            
        label_hat = batch['label_hat'].to(i_feats.device) 
     
        loss1, loss2 = objectives.compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, batch['pids'], \
                                              label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                                loss_type=self.loss_type,logit_scale=self.logit_scale)
        ret.update({'bge_loss':loss1})
        ret.update({'tse_loss':loss2})
  
        return ret

def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    convert_weights(model)
    return model
