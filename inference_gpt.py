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



def load_model(model_path):
    '''load model from path'''
    args_path = model_path.replace('.pt', '_args.pkl')
    with open(args_path, 'rb') as f:
        args = pickle.load(f)
    model = ClipCaptionPrefix(
        prefix_length=args.prefix_length,
        lang = args.lang ,
        clip_length=args.prefix_length_clip,
        prefix_size=512,
        num_layers=args.num_layers,
        mapping_type=args.mapping_type)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model , args.prefix_length


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


def generate_caption(image_path, model, preprocess, clip_model, tokenizer ,prefix_length,  lang ,device):
    
    image = io.imread(image_path)
    pil_image = PIL.Image.fromarray(image)
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        generated_text_prefix = beam_search(model, tokenizer, embed=prefix_embed)

    #display pil_image using plt
    plt.imshow(pil_image)
    plt.axis('off')
    if lang == 'arabic':
        generated_text_prefix = generated_text_prefix#fix_arabic_text(generated_text_prefix)
    print(generated_text_prefix)
    plt.title(generated_text_prefix)
    plt.show()


def main(model_path):
    #Read the language from the model path
    if 'arabic' in model_path:
        lang = 'arabic'
    if 'english' in model_path:
        lang = 'english'
    print(f'The Lang is {lang}')
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


    sample_images_dir = './sample_image'
    sample_images_paths = [os.path.join(sample_images_dir, image_name) for image_name in os.listdir(sample_images_dir)]
    for image_path in sample_images_paths:
        generate_caption(image_path, model,preprocess, clip_model, tokenizer, prefix_length, lang, device)


if __name__ == '__main__':
    #Read the model path from the command line
    model_path = sys.argv[1]
    main(model_path)
