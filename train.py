import os 
import sys
import argparse

if __name__ == '__main__':
    # define a parser object
    parser = argparse.ArgumentParser(description='Train a model')

    # add arguments to the parser
    parser.add_argument('--data', type=str, default='data/embeddings/arabic_CLIP-ViT-B-32_embeddings.pkl', help='path to dataset Images Embeddings')
    parser.add_argument('--out_dir', type=str, default='checkpoint', help='path to save the model')
    parser.add_argument('--model_name', type=str, default='arabic_flicker8k_Meduim', help='name of the model')
    parser.add_argument('--prefix_length', type=int, default=20, help='length of the prefix')
    