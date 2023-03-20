import os
import sys
import torch
import pickle
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset
from transformers import AutoTokenizer, GPT2Tokenizer



class ClipGPTFlickr8kDataset(Dataset):
   
    def __init__(self, data_path,  prefix_length, lang , normalize_prefix=False):
        self.lang = lang
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        
        # Choose the tokenizer based on the language
        if self.lang == 'english':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        # if self.lang == 'arabic':
            # self.tokenizer = AutoTokenizer.from_pretrained("akhooli/gpt2-small-arabic")
        if self.lang == 'arabic':
            self.tokenizer = AutoTokenizer.from_pretrained("elgeish/gpt2-medium-arabic-poetry")

        # Load the data
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        captions_raw = all_data["captions"]
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()

        # Get the image ids and the captions and the CLIP embeddings
        self.prefixes = all_data["clip_embedding"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]

        # Get the tokens for the captions
    
        self.captions_tokens = []
        self.caption2embedding = []
        max_seq_len = 0
        for caption in captions_raw:
            self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
            self.caption2embedding.append(caption["clip_embedding"])
            max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
        with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
            pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)

        # Get the max sequence length
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))


    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask


    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix


    def __len__(self) -> int:
        return len(self.captions_tokens)



class ClipGPTFlickr8kDatasetBilingual(Dataset):
   
    def __init__(self, data_path = 'data/embeddings/multilingual_CLIP-ViT-B-32_embeddings.pkl',  prefix_length = 10 , normalize_prefix=False):
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')


        # Load the data
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        captions_raw = all_data["captions"]
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()

        # Get the image ids and the captions and the CLIP embeddings
        self.prefixes = all_data["clip_embedding"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions_arabic  = [caption['arabic_caption']  for caption in captions_raw]
        self.captions_english = [caption['english_caption'] for caption in captions_raw]

        # Get the tokens for the captions
    
        self.captions_tokens_arabic = []
        self.captions_tokens_english = []
        self.caption2embedding = []
        max_seq_len_arabic = 0
        max_seq_len_english = 0
        for caption in captions_raw:
            self.captions_tokens_arabic.append(torch.tensor(self.tokenizer.encode(caption['arabic_caption']), dtype=torch.int64))
            self.caption2embedding.append(caption["clip_embedding"])
            max_seq_len_arabic = max(max_seq_len_arabic, self.captions_tokens_arabic[-1].shape[0])
            self.captions_tokens_english.append(torch.tensor(self.tokenizer.encode(caption['english_caption']), dtype=torch.int64))
            max_seq_len_english = max(max_seq_len_english, self.captions_tokens_english[-1].shape[0])
           
        
        with open(f"{data_path[:-4]}_tokens_arabic.pkl", 'wb') as f:
            pickle.dump([self.captions_tokens_arabic, self.caption2embedding, max_seq_len_arabic], f)
        with open(f"{data_path[:-4]}_tokens_english.pkl", 'wb') as f:
            pickle.dump([self.captions_tokens_english, self.caption2embedding, max_seq_len_english], f)

        # Get the max sequence length

        all_len_arabic = torch.tensor([len(self.captions_tokens_arabic[i]) for i in range(len(self))]).float()
        all_len_english = torch.tensor([len(self.captions_tokens_english[i]) for i in range(len(self))]).float()

        self.max_seq_len_arabic = min(int(all_len_arabic.mean() + all_len_arabic.std() * 10), int(all_len_arabic.max()))
        self.max_seq_len_english = min(int(all_len_english.mean() + all_len_english.std() * 10), int(all_len_english.max()))

        self.max_seq_len = min(self.max_seq_len_arabic, self.max_seq_len_english)

        sys.stdout.flush()


    def pad_tokens(self, item: int):
        tokens_arabic = self.captions_tokens_arabic[item]
        tokens_english = self.captions_tokens_english[item]

        padding_arabic = self.max_seq_len - tokens_arabic.shape[0]
        padding_english = self.max_seq_len - tokens_english.shape[0]


        if padding_arabic > 0:
            tokens_arabic = torch.cat((tokens_arabic, torch.zeros(padding_arabic, dtype=torch.int64) - 1))
            self.captions_tokens_arabic[item] = tokens_arabic
        elif padding_arabic < 0:
            tokens_arabic = tokens_arabic[:self.max_seq_len]
            self.captions_tokens_arabic[item] = tokens_arabic
        
        if padding_english > 0:
            tokens_english = torch.cat((tokens_english, torch.zeros(padding_english, dtype=torch.int64) - 1))
            self.captions_tokens_english[item] = tokens_english
        elif padding_english < 0:
            tokens_english = tokens_english[:self.max_seq_len]
            self.captions_tokens_english[item] = tokens_english


        mask_arabic = tokens_arabic.ge(0)
        mask_english = tokens_english.ge(0)
        tokens_arabic[~mask_arabic] = 0
        tokens_english[~mask_english] = 0
        mask_arabic = mask_arabic.float()
        mask_english = mask_english.float()

        mask_arabic = torch.cat((torch.ones(self.prefix_length), mask_arabic), dim=0)  # adding prefix mask
        mask_english = torch.cat((torch.ones(self.prefix_length), mask_english), dim=0)  # adding prefix mask
        return tokens_arabic, mask_arabic, tokens_english, mask_english


    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens_arabic, mask_arabic, tokens_english, mask_english = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)

        return tokens_arabic, mask_arabic, tokens_english, mask_english, prefix


    def __len__(self) -> int:
        return len(self.captions_tokens_arabic)
