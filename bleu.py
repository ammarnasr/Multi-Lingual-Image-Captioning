import os
import sys
import clip
import json
import torch
import PIL.Image 
import pandas as pd
from tqdm import tqdm
import skimage.io as io
from nltk.translate.bleu_score import corpus_bleu
from inference_gpt import load_model, beam_search
from transformers import AutoTokenizer, GPT2Tokenizer


def prepare_data_for_bleu(file_path, n=100):   
    sample_images_dir = './data/images/'
    with open(file_path, 'r') as f:
        data = json.load(f)
    sample_image_captions = [item['caption'] for item in data]
    sample_image_ids = [item['image_id'] for item in data]
    unique_image_ids = list(set(sample_image_ids))
    unique_image_ids = unique_image_ids[:n]
    
    image_ids_occurences = []
    for image_id in unique_image_ids:
        image_ids_occurences.append([i for i, x in enumerate(sample_image_ids) if x == image_id])
    captions_per_image = []
    for image_id_occurence in image_ids_occurences:
        captions_per_image.append([sample_image_captions[i] for i in image_id_occurence])
    sample_images_paths = [os.path.join(sample_images_dir, image_name) for image_name in unique_image_ids]

    return captions_per_image, sample_images_paths


def generate_caption(image_path, model, preprocess, clip_model, tokenizer ,prefix_length,  lang ,device):
    image = io.imread(image_path)
    pil_image = PIL.Image.fromarray(image)
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        generated_text_prefix = beam_search(model, tokenizer, embed=prefix_embed, entry_length=10)
    return generated_text_prefix



def belu_score(model_path):
        
    if 'english' in model_path:
        lang = 'english'
    if 'arabic' in model_path:
        lang = 'arabic'

    file_path = f'./data/annotations/{lang}_captions.json'
    sample_image_captions, sample_images_paths = prepare_data_for_bleu(file_path)

    # Load the CLIP model
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # Load the GPT model Tokenizer
    if lang == 'arabic':
        tokenizer = AutoTokenizer.from_pretrained("akhooli/gpt2-small-arabic")
    if lang == 'english':
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load the GPT model
    model, prefix_length = load_model(model_path)
    model.eval()
    model = model.to(device)

    candidates = []
    references = []
    for i in tqdm(range(len(sample_images_paths))):
        image_path = sample_images_paths[i]
        prediction = generate_caption(image_path, model,preprocess, clip_model, tokenizer, prefix_length, lang, device)
        candidates.append(prediction.split(' '))
        references.append([r.split(' ') for r in sample_image_captions[i]])

    score = corpus_bleu(references, candidates) *100
    print(f'The BLEU for Langauge {lang} score is {score}')


if __name__ == '__main__':
    #read model path and file path from command line
    model_path = sys.argv[1]
    file_path = sys.argv[2]
    belu_score(model_path, file_path)