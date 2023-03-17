import os
import sys
import torch
import pickle
import argparse
from tqdm import tqdm
from args import DemoArgs
from models import ClipCaptionPrefix
from torch.nn import functional as nnf
from torch.utils.data import  DataLoader
from dataset import ClipGPTFlickr8kDataset
from transformers import  AdamW, get_linear_schedule_with_warmup

def load_model(model_path):
    '''
    Load the model from the checkpoint
    :param model_path: path to the model checkpoint
    :return: the model, the arguments and the epoch number
    '''
    epoch_number = int(model_path.split('-')[-1].split('.')[0]) + 1
    args_path = model_path.replace('.pt', '_args.pkl')
    with open(args_path, 'rb') as f:
        args = pickle.load(f)
    model = ClipCaptionPrefix(args.prefix_length, args.lang , clip_length=args.prefix_length_clip, prefix_size=512, num_layers=args.num_layers, mapping_type=args.mapping_type)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model , args, epoch_number


def train(dataset, model, args, lr = 2e-5, warmup_steps = 5000, output_dir = ".", output_prefix: str = "", start_epoch = 0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader))

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
        if epoch % args.save_every == 0 or epoch == epochs - 1 + start_epoch:
            model_path = os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt")
            args_path = model_path.replace('.pt', '_args.pkl')
            torch.save(model.state_dict(), model_path)
            with open(args_path, 'wb') as f:
                pickle.dump(args, f)

    return model






def main(model_path = None):
        

    if model_path is None:
        args = DemoArgs()
        start_epoch = 0
        prefix_dim = 640 if args.is_rn else 512
        model = ClipCaptionPrefix(args.prefix_length, lang=args.lang , clip_length=args.prefix_length_clip, prefix_size=prefix_dim, num_layers=args.num_layers)
    else:
        model, args, start_epoch = load_model(model_path)
        prefix_dim = 640 if args.is_rn else 512
    
    dataset = ClipGPTFlickr8kDataset(args.data, args.prefix_length,lang= args.lang, normalize_prefix=args.normalize_prefix)
    train(dataset, model, args, output_dir=args.out_dir, output_prefix=args.output_prefix, start_epoch = start_epoch)


if __name__ == '__main__':
    # define a parser object
    parser = argparse.ArgumentParser(description='Train a model')

    # add arguments to the parser
    parser.add_argument('--data',             type=str, default='data/embeddings/arabic_CLIP-ViT-B-32_embeddings.pkl', help='path to dataset Images Embeddings')
    parser.add_argument('--out_dir',          type=str, default='checkpoint', help='path to save the model')
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
     
    
    # parse the arguments
    args = parser.parse_args()

    # print the arguments one by one
    print('The arguments are:')
    for arg in vars(args):
        print(arg,':\t',  getattr(args, arg))
    
    #check if checkpoint is given as argument
    if args.checkpoint is not None:
        print("Loading model from checkpoint")
        model_path = args.checkpoint
        main(model_path)
    else:
        print('Training from scratch')
        main()
