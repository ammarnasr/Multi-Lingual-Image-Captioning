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


#Create a Class to manage the BLEU score
class BLEU:
    def __init__(self,args_path, model_path,use_eval_data = True, n=100):
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
            self.self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        #TODO: add other languages
        self.load_model()
        self.model.eval()
        self.model = self.model.to(self.device)

    
    def generate_caption(self, image_path):
        image_path = image_path + '.jpg'
        image = io.imread(image_path)
        pil_image = PIL.Image.fromarray(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(self.device, dtype=torch.float32)
            prefix_embed = self.model.clip_project(prefix).reshape(1, self.args.prefix_length, -1)
            generated_text_prefix = beam_search(self.model, self.tokenizer, embed=prefix_embed)
        return generated_text_prefix



    def calculate_bleu(self, n= 100):
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

        self.score = corpus_bleu(references, candidates) *100
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






