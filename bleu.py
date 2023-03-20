import os
import sys
import clip
import json
import argparse
import torch
import PIL.Image 
import pandas as pd
from tqdm import tqdm
import skimage.io as io
from nltk.translate.bleu_score import corpus_bleu
from inference_gpt import  beam_search
from transformers import AutoTokenizer, GPT2Tokenizer
from models import ClipCaptionPrefix
import torch.nn.functional as nnf
import numpy as np



#Create a Class to manage the BLEU score
class BLEU:
    def __init__(self,args_path, model_path,use_eval_data = True, n=100, model=None):
        with open(args_path, 'r') as f:
            args_data = json.load(f)
        self.args = argparse.Namespace()
        self.args.__dict__.update(args_data)
        self.lang = self.args.lang
        self.model_path = model_path
        self.eval_prefix = ''
        if use_eval_data:
            self.eval_prefix = 'eval_'
        self.n = n
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.prepare_data_for_bleu()
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        # Load the GPT model Tokenizer
        if self.lang == 'arabic':
            self.tokenizer = AutoTokenizer.from_pretrained("elgeish/gpt2-medium-arabic-poetry")
        if self.lang == 'english':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        #TODO: add other languages
        if model is None:
            self.load_model()
        else:
            self.model = model
        self.model.eval()
        self.model = self.model.to(self.device)

    
    def generate_caption(self, image_path):
        if not image_path.endswith('.jpg'):
            image_path = image_path + '.jpg'
        
        image = io.imread(image_path)
        pil_image = PIL.Image.fromarray(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(self.device, dtype=torch.float32)
            prefix_embed = self.model.clip_project(prefix).reshape(1, self.args.prefix_length, -1)
            generated_text_prefix = beam_search(self.model, self.tokenizer, embed=prefix_embed)
        return generated_text_prefix



    def calculate_bleu(self, n= 100, ngram_weights=(0.25, 0.25, 0.25, 0.25)):
        candidates = []
        references = []
        # check if n is int or list
        if isinstance(n, int):    
            sample_images_paths = self.sample_images_paths[:n]
            captions_per_image = self.captions_per_image[:n]
        if isinstance(n, list):
            sample_images_paths = [self.sample_images_paths[i] for i in n]
            captions_per_image = [self.captions_per_image[i] for i in n]
        for i in tqdm(range(len(sample_images_paths))):
            image_path = self.sample_images_paths[i]
            prediction = self.generate_caption(image_path)
            candidates.append(prediction.split(' '))
            references.append([r.split(' ') for r in captions_per_image[i]])

        self.candidates = candidates
        self.references = references
        self.score = corpus_bleu(references, candidates, ngram_weights) *100
        print(f'The BLEU for Langauge {self.lang} score is {self.score}')
        return self.score


    def prepare_data_for_bleu(self):
        sample_images_dir = f'./{self.eval_prefix}data/images'
        file_path = f'./{self.eval_prefix}data/annotations/{self.lang}_captions.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
        sample_image_captions = [item['caption'] for item in data]
        sample_image_ids = [item['image_id'] for item in data]
        unique_image_ids = list(set(sample_image_ids))
        
        image_ids_occurences = []
        for image_id in unique_image_ids:
            image_ids_occurences.append([i for i, x in enumerate(sample_image_ids) if x == image_id])
        captions_per_image = []
        for image_id_occurence in image_ids_occurences:
            captions_per_image.append([sample_image_captions[i] for i in image_id_occurence])
        sample_images_paths = [os.path.join(sample_images_dir, image_name) for image_name in unique_image_ids]

        self.captions_per_image = captions_per_image
        self.sample_images_paths = sample_images_paths


    def load_model(self):
        model = ClipCaptionPrefix(self.args.prefix_length, lang=self.args.lang, clip_length=self.args.clip_length,
                                   prefix_size=self.args.prefix_size, num_layers=self.args.num_layers)
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model = model





#Create a Class to manage the BLEU score
class BLEUBI:
    def __init__(self,args_path, model_path=None,use_eval_data = True, n=100, model=None):
        #check if args_path is a string or else
        if isinstance(args_path, str):    
            with open(args_path, 'r') as f:
                args_data = json.load(f)
            self.args = argparse.Namespace()
            self.args.__dict__.update(args_data)
        else:
            self.args = args_path
        self.lang_src = 'english'
        self.lang_tgt = 'arabic'
        self.model_path = model_path
        self.eval_prefix = ''
        if use_eval_data:
            self.eval_prefix = 'eval_'
        self.n = n
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.max_seq_len =41

        self.sample_images_paths_src, self.captions_per_image_src = self.prepare_data_for_bleu(self.lang_src)
        self.sample_images_paths_tgt, self.captions_per_image_tgt = self.prepare_data_for_bleu(self.lang_tgt)
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        # Load the GPT model Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        if model is None:
            self.load_model()
        else:
            self.model = model
        self.model.eval()
        self.model = self.model.to(self.device)

    
    def generate_caption(self, image_path, img_index):
        image_path = image_path + '.jpg'
        image = io.imread(image_path)
        pil_image = PIL.Image.fromarray(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(self.device, dtype=torch.float32)
            prefix_embed = self.model.clip_project(prefix).reshape(1, self.args.prefix_length, -1)
            src_captions = [self.captions_per_image_src[img_index][i] for i in range(len(self.captions_per_image_src[img_index]))]
            #choose a random caption from the reference captions
            src_caption = np.random.choice(src_captions)
            generated_text_prefix = self.beam_search(self.model, self.tokenizer, embed=prefix, src_caption=src_caption)
    # def beam_search(model, tokenizer, embed, entry_length=20, top_p=0.8, temperature=1., stop_token= '.', src_caption=None):
        
        print(f'Generated Caption: {generated_text_prefix}')
        print(f'Reference Caption: {self.captions_per_image_tgt[img_index]}')
        print(f'src_caption: {src_caption}')
        return generated_text_prefix



    def calculate_bleu(self, n= 100):
        candidates = []
        references = []
        # check if n is int or list
        if isinstance(n, int):    
            sample_images_paths = self.sample_images_paths_tgt[:n]
            captions_per_image = self.captions_per_image_tgt[:n]
        if isinstance(n, list):
            sample_images_paths = [self.sample_images_paths_tgt[i] for i in n]
            captions_per_image = [self.captions_per_image_tgt[i] for i in n]
        for i in tqdm(range(len(sample_images_paths))):
            image_path = sample_images_paths[i]
            prediction = self.generate_caption(image_path, i)
            candidates.append(prediction.split(' '))
            references.append([r.split(' ') for r in captions_per_image[i]])

        self.score = corpus_bleu(references, candidates) *100
        print(f'The BLEU for Langauge {self.lang_tgt} score is {self.score}')
        return self.score


    def prepare_data_for_bleu(self, lang):
        sample_images_dir = f'./{self.eval_prefix}data/images'
        file_path = f'./{self.eval_prefix}data/annotations/{lang}_captions.json'

        with open(file_path, 'r') as f:
            data = json.load(f)
        sample_image_captions = [item['caption'] for item in data]
        sample_image_ids = [item['image_id'] for item in data]
        unique_image_ids = list(set(sample_image_ids))
        
        image_ids_occurences = []
        for image_id in unique_image_ids:
            image_ids_occurences.append([i for i, x in enumerate(sample_image_ids) if x == image_id])
        captions_per_image = []
        for image_id_occurence in image_ids_occurences:
            captions_per_image.append([sample_image_captions[i] for i in image_id_occurence])
        sample_images_paths = [os.path.join(sample_images_dir, image_name) for image_name in unique_image_ids]

        return sample_images_paths, captions_per_image


    def load_model(self):
        model = ClipCaptionPrefix(self.args.prefix_length, lang=self.lang_src, clip_length=self.args.clip_length,
                                   prefix_size=self.args.prefix_size, num_layers=self.args.num_layers)
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model = model


    def beam_search(self, model, tokenizer, embed, entry_length=20, top_p=0.8, temperature=1., stop_token= '.', src_caption=None):
        '''Beam search for the GPT model.'''
        
        model.eval()
        generated_list = []
        stop_token_index = tokenizer.encode(stop_token)[0]
        filter_value = -float("Inf")
        generated = embed
        tokens = None
        
        src_tokens = tokenizer.encode(src_caption)
        src_tokens = torch.tensor(src_tokens, dtype=torch.int64).to(self.device)
        src_tokens, src_mask = self.pad_tokens(src_tokens)
        src_tokens = src_tokens.unsqueeze(0)
        src_mask = src_mask.unsqueeze(0)
        with torch.no_grad():
            for i in range(entry_length):
                
                #  get the logits for the next token
                                
                outputs =model(src_tokens, generated, src_mask)


                # outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                
                #  take the most likely token and add it to the sequence
                # next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token = torch.argmax(logits, -1)
                next_token_item = next_token.item()
                next_token, next_token_mask = self.pad_tokens(next_token)
                next_token = next_token.unsqueeze(0)
                next_token_mask = next_token_mask.unsqueeze(0)

                src_tokens = next_token
                src_mask = next_token_mask






                # transform the token to embedding
                next_token_embed = model.gpt.transformer.wte(next_token)

                # add the token to the sequence
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                
                # add the embedding to the sequence
                # generated = torch.cat((generated, next_token_embed), dim=1)

                # stop if the stop token is reached
                # if stop_token_index == next_token.item():
                if stop_token_index == next_token_item:
                    break
                if stop_token == tokenizer.decode(tokens.squeeze().cpu().numpy())[-1]:
                    break

            # convert the sequence to text
            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

        return generated_list[0]




    def pad_tokens(self, tokens):
        padding = self.max_seq_len - tokens.shape[0]


        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        


        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = mask.float()

        mask = torch.cat((torch.ones(self.args.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask


