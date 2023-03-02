import json
import clip
import torch
import pickle
import pandas as pd
from PIL import Image
from tqdm import tqdm
import skimage.io as io



def merge_captions_flickr8k():
    '''Merge the english and arabic captions for the flickr8k dataset and save them to a pickle file'''

    # read english captions with column names as image and caption
    english_captions = pd.read_csv('data/Flickr8k_text/Flickr8k.arabic.full.txt', sep=',')
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


def create_captions_json():
    '''
    Create a json file for the arabic and english captions
    '''
    # load the merged captions
    merged_captions = pd.read_pickle('data/Flickr8k_text/merged_captions.pkl')

    # Get the arabic and english captions
    arabic_captions_df = merged_captions[['image', 'arabic_caption']]
    english_captions_df = merged_captions[['image', 'english_caption']]

    # Split the three captions per image into three rows
    arabic_captions_df = arabic_captions_df.explode('arabic_caption')
    english_captions_df = english_captions_df.explode('english_caption')

    #Rename the columns image and arabic_caption to image_id and caption
    arabic_captions_df = arabic_captions_df.rename(columns={'image': 'image_id', 'arabic_caption': 'caption'})
    english_captions_df = english_captions_df.rename(columns={'image': 'image_id', 'english_caption': 'caption'})

    #Convert the dataframe to list of Dictionaries
    arabic_captions = arabic_captions_df.to_dict('records')
    english_captions = english_captions_df.to_dict('records')

    # Save the list of dictionaries to a json file
    with open('./data/annotations/arabic_captions.json', 'w') as f:
        json.dump(arabic_captions, f)
    with open('./data/annotations/english_captions.json', 'w') as f:
        json.dump(english_captions, f)


def create_CLIP_embeddings_for_images(lang, clip_model_type='ViT-B/32', device='cuda'):
    '''
    Create the CLIP embeddings for the images and save them to a pickle file
    '''
    # create the output path
    clip_model_name = clip_model_type.replace('/', '-') # convert ViT-B/32 to ViT-B-32 to be used in the file name
    out_path = f"./data/embeddings/{lang}_CLIP-{clip_model_name}_embeddings.pkl" # path to save the embeddings

    # load the CLIP model
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)# load the CLIP model
    clip_model.eval()

    # load the annotations
    annotations_file = f"./data/annotations/{lang}_captions.json"
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    # create list of dictionaries with the image id, the CLIP embedding and the caption
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        # load the image
        d = data[i]
        img_id = d["image_id"]
        filename = f"./data/images/{img_id}"
        image = io.imread(filename)

        # preprocess the image and encode it with the CLIP model
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        prefix = clip_model.encode_image(image).cpu()

        # add the index , embedding and caption to the dictionary
        d["clip_embedding"] = i 
        all_embeddings.append(prefix)
        all_captions.append(d)

    # save the dictionary to a pickle file
    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)








