import os
import sys
import clip
import torch
import pickle
import PIL.Image 
import skimage.io as io
import matplotlib.pyplot as plt
import torch.nn.functional as nnf
from plotting import fix_arabic_text
from models import  ClipCaptionPrefix
from transformers import AutoTokenizer, GPT2Tokenizer
import argparse
import json




def beam_search(model, tokenizer, embed, entry_length=20, top_p=0.8, temperature=1., stop_token= '.'):
    '''Beam search for the GPT model.'''
    
    model.eval()
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    generated = embed
    tokens = None
    
    with torch.no_grad():
        for i in range(entry_length):
            
            #  get the logits for the next token
            outputs = model.gpt(inputs_embeds=generated)
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
            next_token = torch.argmax(logits, -1).unsqueeze(0)


            # transform the token to embedding
            next_token_embed = model.gpt.transformer.wte(next_token)

            # add the token to the sequence
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            
            # add the embedding to the sequence
            generated = torch.cat((generated, next_token_embed), dim=1)

            # stop if the stop token is reached
            if stop_token_index == next_token.item():
                break
            if stop_token == tokenizer.decode(tokens.squeeze().cpu().numpy())[-1]:
                break

        # convert the sequence to text
        output_list = list(tokens.squeeze().cpu().numpy())
        output_text = tokenizer.decode(output_list)
        generated_list.append(output_text)

    return generated_list[0]





class Inference:
    def __init__(self, args_path, model_path=None, model=None):

        #check if args_path is a string or else
        if isinstance(args_path, str):    
            with open(args_path, 'r') as f:
                args_data = json.load(f)
            self.args = argparse.Namespace()
            self.args.__dict__.update(args_data)
        else:
            self.args = args_path
        self.lang = self.args.lang
        self.model_path = model_path
        self.eval_prefix = ''
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        # Load the GPT model Tokenizer
        if self.lang == 'arabic':
            self.tokenizer = AutoTokenizer.from_pretrained("elgeish/gpt2-medium-arabic-poetry")
        if self.lang == 'english':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        if model is None:
            self.load_model()
        else:
            self.model = model
        self.model.eval()
        self.model = self.model.to(self.device)

        
        self.sample_images_dir = './sample_image'
        self.sample_images_paths = [os.path.join(self.sample_images_dir, image_name) for image_name in os.listdir(self.sample_images_dir)]
    

    

    def load_model(self):
        model = ClipCaptionPrefix(self.args.prefix_length, lang=self.args.lang, clip_length=self.args.clip_length,
                                   prefix_size=self.args.prefix_size, num_layers=self.args.num_layers)
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model = model


        
    def generate_caption(self, image_path = None):
        if image_path is None:
            # get a random image from the sample images
            image_path = self.sample_images_paths[torch.randint(0, len(self.sample_images_paths), (1,)).item()]
        if not image_path.endswith('.jpg'):
            image_path = image_path + '.jpg'
    
        image = io.imread(image_path)
        pil_image = PIL.Image.fromarray(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(self.device, dtype=torch.float32)
            prefix_embed = self.model.clip_project(prefix).reshape(1, self.args.prefix_length, -1)
            generated_text_prefix = beam_search(self.model, self.tokenizer, embed=prefix_embed)
        return generated_text_prefix, image_path




