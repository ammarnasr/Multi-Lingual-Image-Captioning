import torch
from torch.nn import functional as nnf
from torch.utils.data import  DataLoader
from transformers import  AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import sys
import argparse
import json
from typing import  Union
from dataset import ClipGPTFlickr8kDataset, ClipGPTLaion5bArabicDataset
from models import ClipCaptionModel, ClipCaptionPrefix

def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


def train(dataset: ClipGPTFlickr8kDataset, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    return model

#create a class called demo args to store the arguments
class DemoArgs:
    def __init__(self):
        self.data = 'data/oscar_split_ViT-B_32_train_arabic.pkl'
        self.out_dir = './checkpoints'
        self.prefix = 'coco_prefix'
        self.epochs = 10
        self.save_every = 1
        self.prefix_length = 10
        self.prefix_length_clip = 10
        self.bs = 24
        self.only_prefix = True
        self.mapping_type = 'mlp'
        self.num_layers = 8
        self.is_rn = False
        self.normalize_prefix = False

def main(lang):
    args = DemoArgs()
    if lang == 'arabic':
        args.data = './data/oscar_split_ViT-B_32_train_arabic.pkl'
        output_prefix = 'arabic_prefix'
    if lang == 'english':
        args.data = './data/oscar_split_ViT-B_32_train.pkl'
        output_prefix = 'english_prefix'
    dataset = ClipGPTFlickr8kDataset(args.data, args.prefix_length, normalize_prefix=args.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512
    model = ClipCaptionPrefix(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim, num_layers=args.num_layers)
    print("Train only prefix")
    sys.stdout.flush()
    train(dataset, model, args, output_dir=args.out_dir, output_prefix=output_prefix)


def main2():
    args = DemoArgs()
    args.data = './data/laion_part4_ViT-B_32_train.pkl'
    output_prefix = 'arabic_prefix_laion'
    dataset = ClipGPTFlickr8kDataset(args.data, args.prefix_length, normalize_prefix=args.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512
    model = ClipCaptionPrefix(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim, num_layers=args.num_layers)
    print("Train only prefix")
    sys.stdout.flush()
    train(dataset, model, args, output_dir=args.out_dir, output_prefix=output_prefix)

if __name__ == '__main__':
    #read the arguments from the command line
    # lang = sys.argv[1]
    # main(lang)
    main2()
