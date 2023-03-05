import torch
from torch.nn import functional as nnf
from torch.utils.data import  DataLoader
from transformers import  AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import sys
import argparse
import pickle
import json
from typing import  Union
from dataset import ClipGPTFlickr8kDataset
from models import ClipCaptionModel, ClipCaptionPrefix, MappingType
from args import DemoArgs
from bleu import belu_score


def load_model(model_path):
    '''load model from path'''
    epoch_number = int(model_path.split('-')[-1].split('.')[0]) + 1
    args_path = model_path.replace('.pt', '_args.pkl')
    with open(args_path, 'rb') as f:
        args = pickle.load(f)
    model = ClipCaptionPrefix(args.prefix_length, args.lang , clip_length=args.prefix_length_clip, prefix_size=512, num_layers=args.num_layers, mapping_type=args.mapping_type)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model , args, epoch_number


def train(dataset: ClipGPTFlickr8kDataset, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = "", start_epoch = 0):

    device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    # save_config(args)
    for epoch in range(start_epoch, epochs+start_epoch):
        print(f">>> Training epoch {epoch} out of {epochs+start_epoch}")
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
                torch.save(model.state_dict(),os.path.join(output_dir, f"{output_prefix}_latest.pt"),)
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            model_path = os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt")
            args_path = model_path.replace('.pt', '_args.pkl')
            torch.save(model.state_dict(), model_path)
            with open(args_path, 'wb') as f:
                pickle.dump(args, f)

            if args.get_bleu:
                print('Evaluating model on BLEU score')
                belu_score(model_path)
                print('Done')
    return model






def main(model_path = None):
        

    if model_path is None:
        args = DemoArgs()
        start_epoch = 0
        prefix_dim = 640 if args.is_rn else 512
        args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
        model = ClipCaptionPrefix(args.prefix_length, lang=args.lang , clip_length=args.prefix_length_clip, prefix_size=prefix_dim, num_layers=args.num_layers, mapping_type=args.mapping_type)
    else:
        model, args, start_epoch = load_model(model_path)
        prefix_dim = 640 if args.is_rn else 512
    
    dataset = ClipGPTFlickr8kDataset(args.data, args.prefix_length,lang= args.lang, normalize_prefix=args.normalize_prefix)
    train(dataset, model, args, output_dir=args.out_dir, output_prefix=args.output_prefix, start_epoch = start_epoch)


if __name__ == '__main__':
    # check if checkpoint is given as argument
    if len(sys.argv) > 1:
        print("Loading model from checkpoint")
        model_path = sys.argv[1]
        main(model_path)
    else:
        print('Training from scratch')
        main()
