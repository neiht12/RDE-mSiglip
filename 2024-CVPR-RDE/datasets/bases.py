
from torch.utils.data import Dataset
import logging
import torch
import random
import numpy as np
import os
from prettytable import PrettyTable
from transformers import AutoTokenizer

# Giả định bạn có file utils/iotools.py chứa hàm read_image
from utils.iotools import read_image 

def inject_noisy_correspondence(dataset, noisy_rate, noisy_file=None):
    logger = logging.getLogger("RDE.dataset")
    nums = len(dataset)
    dataset_copy = dataset.copy()
    captions  = [i[3] for i in dataset_copy]
    images    = [i[2] for i in dataset_copy]
    image_ids = [i[1] for i in dataset_copy]
    pids      = [i[0] for i in dataset_copy]

    noisy_inx = np.arange(nums)
    if noisy_rate > 0:
        print(noisy_file)
        random.seed(123)
        if os.path.exists(noisy_file):
            logger.info('=> Load noisy index from {}'.format(noisy_file))
            noisy_inx = np.load(noisy_file)
        else:
            inx = np.arange(nums)
            np.random.shuffle(inx)
            c_noisy_inx = inx[0: int(noisy_rate * nums)]
            shuffle_noisy_inx = np.array(c_noisy_inx)
            np.random.shuffle(shuffle_noisy_inx)
            noisy_inx[c_noisy_inx] = shuffle_noisy_inx
            np.save(noisy_file, noisy_inx)

    real_correspondeces = []
    for i in range(nums):
        if noisy_inx[i] == i:
            real_correspondeces.append(1)
        else:
            real_correspondeces.append(0)
        # pid, real_pid, image_id, image_path, text
        tmp = (pids[i], image_ids[i], images[i], captions[noisy_inx[i]])
        dataset[i] = tmp
    
    # Chỉ log 10 phần tử đầu để debug
    logger.info(f"First 10 correspondence flags: {real_correspondeces[0:10]}")
    logger.info('=> Noisy rate: {}, Clean pairs: {}, Noisy pairs: {}, Total pairs: {}'.format(
        noisy_rate, np.sum(real_correspondeces), nums - np.sum(real_correspondeces), nums
    ))

    return dataset, np.array(real_correspondeces)


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("RDE.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 64, # SigLIP default is often 64
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        
        # Init Tokenizer for Google SigLIP
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-256-multilingual")
  
    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]
        
        # Tokenize using HuggingFace logic
        inputs = self.tokenizer(
            caption, 
            padding="max_length", 
            truncation=self.truncate, 
            max_length=self.text_length, 
            return_tensors="pt"
        )
        # Squeeze to remove batch dimension: [1, 64] -> [64]
        caption_tensor = inputs["input_ids"].squeeze(0)

        return pid, caption_tensor


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset, args,
                 transform=None,
                 text_length: int = 64,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.txt_aug = args.txt_aug
        self.img_aug = args.img_aug
        
        self.dataset, self.real_correspondences = inject_noisy_correspondence(dataset, args.noisy_rate, args.noisy_file)
        
        # Init Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-256-multilingual")
        
        # Prepare special tokens for Augmentation
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        # Fallback to UNK if MASK doesn't exist (SentencePiece typically doesn't have [MASK])
        self.mask_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.unk_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        
        # 1. Tokenize
        inputs = self.tokenizer(
            caption, 
            padding="max_length", 
            truncation=self.truncate, 
            max_length=self.text_length, 
            return_tensors="pt"
        )
        caption_tokens = inputs["input_ids"].squeeze(0)

        # 2. Text Augmentation
        if self.txt_aug:
            caption_tokens = self.txt_data_aug(caption_tokens)
        
        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'index': index,
        }

        return ret

    def txt_data_aug(self, tokens):
        # Chuyển tensor sang numpy để xử lý
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.numpy()
            
        # [CẤU HÌNH CHO SIGLIP]
        VOCAB_SIZE = 250002        # Giới hạn từ điển của SigLIP
        MASK_TOKEN = 0             
        PAD_TOKEN = 1              # SigLIP pad_token_id là 1
        
        # Khởi tạo mảng kết quả full Padding (1) thay vì 0
        new_tokens = np.full_like(tokens, PAD_TOKEN)
        aug_tokens = []
        
        for i, token in enumerate(tokens):
            # Chỉ augment các token hợp lệ 
            
            if 1 < token < VOCAB_SIZE:
                prob = random.random()
                
                # Augment với xác suất 20%
                if prob < 0.20:
                    prob /= 0.20
                    
                    # 60% đổi thành MASK token
                    if prob < 0.6:
                        aug_tokens.append(MASK_TOKEN)
                        
                    # 20% đổi thành RANDOM token
                    elif prob < 0.8:
                        # Dùng randint trực tiếp thay vì tạo list(range(...)) để tránh lag
                        # Random từ 2 đến VOCAB_SIZE - 1 (tránh pad/unk)
                        random_token = random.randint(2, VOCAB_SIZE - 1)
                        aug_tokens.append(random_token)
                        
                    else:
                        # 20% còn lại: Xóa token (không append gì cả)
                        pass 
                else:
                    # Giữ nguyên token
                    aug_tokens.append(token)
            else:
                # Token đặc biệt (Padding, v.v...) giữ nguyên
                aug_tokens.append(token)
        
        # Đưa danh sách đã augment vào mảng kết quả
        
        valid_len = min(len(aug_tokens), len(tokens))
        new_tokens[:valid_len] = aug_tokens[:valid_len]
        
        return torch.tensor(new_tokens, dtype=torch.long)