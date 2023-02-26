import clip
import torch
import numpy as np
import transformers
from PIL import Image
from multilingual_clip import pt_multilingual_clip
from clip.simple_tokenizer import SimpleTokenizer
import os
import json
import argparse


clip_tokenizer = SimpleTokenizer()


def load_image(image_id):
    image_path = 'data/Images/' + image_id
    image = Image.open(image_path)
    return image


def tokenize_text(caption):
    t = torch.tensor(clip_tokenizer.encode(caption), dtype=torch.int64)
    t = torch.nn.functional.pad(t, (0, 77 - t.shape[0]), value=0)
    return t


def decode_tokens(tokens):
    t = tokens[tokens != 0].tolist()
    return clip_tokenizer.decode(t)

def load_tokenizer(model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'):
    '''
    Load Text tokenizer for the model
    model_name: name of the model
    '''
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def load_multilingual_clip(model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14', device = 'cuda'):
    '''
    Load the multilingual clip model 
    model_name: name of the model
    '''
    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name).to(device)
    return model

def load_clip(model_name = 'ViT-L/14', device = 'cuda'):
    '''
    Load the clip model and image preprocess pipeline
    model_name: name of the model
    '''
    model, preprocess = clip.load(model_name, device = device)
    return model, preprocess


def load_images(image_paths, preprocess, device = 'cuda'):
    '''
    Load images from the image paths and preprocess them
    image_paths: list of image paths
    preprocess: image preprocess pipeline
    '''
    org_images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        org_images.append(image)
    porcessed_images = torch.cat([preprocess(i).unsqueeze(dim=0) for i in org_images]).to(device)
    return porcessed_images, org_images


def compare_embeddings(logit_scale, img_embs, txt_embs):
    '''
    Compare the embeddings of the images and texts
    logit_scale: scale of the logits
    img_embs: image embeddings
    txt_embs: text embeddings
    '''
    # normalized features
    image_features = img_embs / img_embs.norm(dim=-1, keepdim=True)
    text_features = txt_embs / txt_embs.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text




def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)

