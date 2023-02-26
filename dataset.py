import pandas as pd
import torch.utils.data as data
import random
import tools
import vocab
import torch
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import os
import pickle
import sys
from typing import Tuple


def get_loader_f8k(lang='ar', batch_size=1, clip_model='ViT-L/14', caption_len=5,  device = 'cuda'):
    encoder, preprocess = tools.load_clip(model_name=clip_model, device=device)
    merged_captions = pd.read_pickle('data/Flickr8k_text/merged_captions.pkl')
    f8k = Flickr8kDataset(preprocess, merged_captions=merged_captions, lang=lang, caption_len=caption_len)
    data_loader = data.DataLoader(dataset=f8k, batch_size=batch_size, shuffle=True,)
    return encoder, data_loader


def merge_captions_flickr8k():

    english_captions = pd.read_csv('data/Flickr8k_text/captions.txt', sep=',')
    # read arabic captions with column names as image and caption
    arabic_captions = pd.read_csv('data/Flickr8k_text/Flickr8k.arabic.full.txt', sep='\t', names=['image', 'caption'])

    # remove the last 2 characters from image name in arabic captions
    arabic_captions['image'] = arabic_captions['image'].apply(lambda x: x[:-2])

    merger_captions = {
        'image': [],
        'arabic_caption': [],
        'num_arabic_captions': [],
        'english_caption': [],
        'num_english_captions': []
    }

    for image in english_captions['image'].unique():
        # get all english captions for the image
        english_caption = english_captions[english_captions['image'] == image]['caption'].values
        # get all arabic captions for the image
        arabic_caption = arabic_captions[arabic_captions['image'] == image]['caption'].values

        # Add the image name to the dictionary
        merger_captions['image'].append(image)
        # Add the english caption to the dictionary
        merger_captions['english_caption'].append(english_caption)
        # Add the arabic caption to the dictionary
        merger_captions['arabic_caption'].append(arabic_caption)
        # Add the number of english captions to the dictionary
        merger_captions['num_english_captions'].append(len(english_caption))
        # Add the number of arabic captions to the dictionary
        merger_captions['num_arabic_captions'].append(len(arabic_caption))

    # create a dataframe from the dictionary
    merged_captions = pd.DataFrame(merger_captions)
    # save the dataframe to a pickle file
    merged_captions.to_pickle('data/Flickr8k_text/merged_captions.pkl')


class Flickr8kDataset(data.Dataset):
    
    def __init__(self, transform, merged_captions, lang, caption_len=5):
        self.transform = transform
        self.merged_captions = merged_captions
        self.lang = lang
        self.caption_len = caption_len
        
    def __getitem__(self, index):
        image, caption = self.getitem(index)
        image = self.transform(image)
        # caption = tools.tokenize_text(caption)
        #tokenize the caption arabic or english
        caption = vocab.caption_to_indices(caption, lang=self.lang, size=self.caption_len)
        caption = torch.tensor(caption)
        return image, caption

    def __len__(self):
        return len(self.merged_captions)


    def getitem(self, index):
        img_id_index = 0
        arabic_caption_index = 1
        english_caption_index = 3

        # get the image id
        image_id = self.merged_captions.iloc[index, img_id_index]
        # load the image
        image = tools.load_image(image_id)

        # get the arabic captions
        if self.lang == 'ar':
            caption = self.merged_captions.iloc[index, arabic_caption_index]
            # select a random caption out of the 3 captions
            caption = random.choice(caption)

        # get the english captions
        elif self.lang == 'en':
            caption = self.merged_captions.iloc[index, english_caption_index]
            # select a random caption out of the 5 captions
            caption = random.choice(caption)
        
        return image, caption


class ClipGPTFlickr8kDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

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

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

