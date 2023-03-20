import os
import sys
import time
import json
import torch
import pickle
import argparse
from tqdm import tqdm
from models import ClipCaptionPrefix
from torch.nn import functional as nnf
from torch.utils.data import  DataLoader
from dataset import ClipGPTFlickr8kDataset
from transformers import  AdamW, get_linear_schedule_with_warmup
from bleu import BLEU

def load_model(args):
    '''
    Load the model from the checkpoint
    :param model_path: path to the model checkpoint
    :return: the model, the arguments and the epoch number
    '''

    #load args from json file
    model_path = args.checkpoint
    epoch_number = int(model_path.split('-')[-1].split('.')[0])
    #create the args path from the model path by replacing the extension and adding _args.json and remove the epoch number
    args_path = model_path.replace('.pt', '_args.json').replace(f'-{epoch_number:03d}', '')
    # check if the args_path exists
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            args_json = json.load(f)
        args.__dict__.update(args_json)
    else:
        # load args from pickle file
        args_path = model_path.replace('.pt', '_args.pkl')
        with open(args_path, 'rb') as f:
            args = pickle.load(f)

    model = ClipCaptionPrefix(args.prefix_length, lang=args.lang, clip_length=args.clip_length, prefix_size=args.prefix_size, num_layers=args.num_layers)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model , args, epoch_number


def train(dataset, model, args , start_epoch = 0, dev_ratio = 0.3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    warmup_steps = args.warmup_steps
    output_dir = args.output_dir
    model_name = args.model_name
    precentage = args.percentage

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    if precentage < 1:
        #adjust the dataloader to the percentage of the dataset
        rand_inds = torch.randperm(len(dataset))[:int(len(dataset) * precentage)]
        sampler = torch.utils.data.SubsetRandomSampler(rand_inds)


    if dev_ratio > 0:
        #split the dataset to train and dev
        train_size = int((1 - dev_ratio) * len(dataset))
        dev_size = len(dataset) - train_size
        train_dataset, dev_dataset = torch.utils.data.random_split(dataset, [train_size, dev_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2, sampler=sampler)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader))

    # Create one Dictionary to track: loss per batch, loss per epoch, time per epoch and time per batch and the BLEU score per epoch
    tracker = {'loss_per_batch': [], 'loss_per_epoch': [], 'time_per_epoch': [], 'time_per_batch': [], 'bleu_per_epoch': [],
               'loss_per_batch_dev': [], 'loss_per_epoch_dev': [], 'time_per_epoch_dev': [], 'time_per_batch_dev': [], 'bleu_per_epoch_dev': []
               }


    for epoch in range(start_epoch, epochs+start_epoch):
        print(f">>> Training epoch {epoch} out of {epochs+start_epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=model_name)
        progress_dev = tqdm(total=len(dev_dataloader), desc=model_name)
        # start the epoch timer
        epoch_start = time.time()
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            # start the batch timer
            batch_start = time.time()
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
            if (idx + 1) % 100 == 0:
                torch.save(model.state_dict(),os.path.join(output_dir, f"{model_name}_latest.pt"),)
            # end the batch timer
            batch_end = time.time()
            # append the batch time to the tracker
            tracker['time_per_batch'].append(batch_end - batch_start)
            # append the batch loss to the tracker
            tracker['loss_per_batch'].append(loss.item())


        if dev_ratio > 0:
            for idx, (tokens, mask, prefix) in enumerate(dev_dataloader):
                # start the batch timer
                batch_start = time.time()
                model.zero_grad()
                tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, dataset.prefix_length - 1: -1]
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
                progress_dev.set_postfix({"loss": loss.item()})
                progress_dev.update()
                # end the batch timer
                batch_end = time.time()
                # append the batch time to the tracker
                tracker['time_per_batch_dev'].append(batch_end - batch_start)
                # append the batch loss to the tracker
                tracker['loss_per_batch_dev'].append(loss.item())





        progress.close()
        progress_dev.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1 + start_epoch:
            model_path = os.path.join(output_dir, f"{model_name}-{epoch:03d}.pt")
            torch.save(model.state_dict(), model_path)
        
        
        # end the epoch timer
        epoch_end = time.time()
        # append the epoch time to the tracker
        tracker['time_per_epoch'].append(epoch_end - epoch_start)
        # append the epoch loss to the tracker
        tracker['loss_per_epoch'].append(loss.item())
        # # calculate the bleu score 
        model_path = os.path.join(output_dir, f"{model_name}-{epoch:03d}.pt")
        #create the args path from the model path by replacing the extension and adding _args.json and remove the epoch number
        args_path = model_path.replace('.pt', '_args.json').replace(f'-{epoch:03d}', '')
        bleu_obj = BLEU(args_path, model_path, model=model)
        bleu = bleu_obj.calculate_bleu()
        # append the bleu score to the tracker
        tracker['bleu_per_epoch'].append(bleu)
        # save the tracker to a pickle file
        tracker_path = os.path.join(output_dir, f"{model_name}_tracker.pkl")
        if start_epoch != 0:
            tracker_path = tracker_path.replace('.pkl', f'-{start_epoch:03d}.pkl')
        with open(tracker_path, 'wb') as f:
            pickle.dump(tracker, f)
    return model


def main(args):
    ck = args.checkpoint
    if ck is None:
        start_epoch = 0
        model = ClipCaptionPrefix(args.prefix_length, lang=args.lang, clip_length=args.clip_length, prefix_size=args.prefix_size, num_layers=args.num_layers)
    else:
        model, args, start_epoch = load_model(args)
    
    dataset = ClipGPTFlickr8kDataset(args.data, args.prefix_length,lang= args.lang, normalize_prefix=args.normalize_prefix)
    train(dataset, model, args, start_epoch = start_epoch)


if __name__ == '__main__':
    # define a parser object
    parser = argparse.ArgumentParser(description='Train a model')

    # add arguments to the parser
    parser.add_argument('--data',             type=str, default='data/embeddings/arabic_CLIP-ViT-B-32_embeddings.pkl', help='path to dataset Images Embeddings')
    parser.add_argument('--output_dir',       type=str, default='checkpoint', help='path to save the model')
    parser.add_argument('--model_name',       type=str, default='arabic_flicker8k_Meduim', help='name of the model')
    parser.add_argument('--prefix_length',    type=int, default=10, help='The Number of prefix tokens to which the CLIP features will be mapped')
    parser.add_argument('--clip_length',      type=int, default=10, help='The Number of CLIP Visual features')
    parser.add_argument('--prefix_size',      type=int, default=512, help='The size of the CLIP Visual features Embeddings')
    parser.add_argument('--batch_size',       type=int, default=32, help='Batch size')
    parser.add_argument('--epochs',           type=int, default=10, help='Number of epochs')
    parser.add_argument('--save_every',       type=int, default=1, help='Save the model every n epochs')
    parser.add_argument('--lang',             type=str, default='arabic', help='The language of the dataset')
    parser.add_argument('--normalize_prefix', type=bool,default=True, help='Normalize the prefix features')
    parser.add_argument('--num_layers',       type=int, default=1, help='Number of layers in the Transformer Mapper')
    parser.add_argument('--mapping_type',     type=str, default='linear', help='The type of mapping between the CLIP features and the prefix features')
    parser.add_argument('--checkpoint',       type=str, default=None, help='The path to the checkpoint to load the model from')
    parser.add_argument('--lr',               type=float, default=0.01, help='The learning rate')
    parser.add_argument('--warmup_steps',     type=int, default=1000, help='The number of warmup steps')
    #add argumnet for the precentage of the dataset to use
    parser.add_argument('--percentage',       type=float, default=1, help='The percentage of the dataset to use')
    
    # parse the arguments
    args = parser.parse_args()

    # print the arguments one by one
    print('The arguments are:')
    for arg in vars(args):
        print(arg,':\t',  getattr(args, arg))

    if args.checkpoint is not None:
        main(args)
    
    else:
        args_path = os.path.join(args.output_dir, f"{args.model_name}_args.json")
        with open(args_path, 'w') as f:
            json.dump(args.__dict__, f)
        print(f'Arguments saved to {args_path}')

        print('+----------------------------------------------------------------------------------------------------------+')
        print('+----------------------------------------------------------------------------------------------------------+')
        print('Starting training..')
        main(args)
