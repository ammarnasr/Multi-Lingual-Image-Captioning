import dataset
import torch
import torch.nn as nn
import math
import os
from tqdm import tqdm
import sys
import vocab
from models import DecoderRNN
import json
import numpy as np

def init_experiment(experiment_name):
    # load the expriment_parameters.json file as a dictionary using JSON library
    with open('experiment_parameters.json') as f:
        params = json.load(f)

    # print the dictionary in a nice format
    print(json.dumps(params[experiment_name], indent=4, sort_keys=True))

    return params[experiment_name]


def train_model(encoder, decoder, data_loader, loss_function, optimizer, num_epochs=25, print_every=10, save_every=1, model_dir='models' , model_name='model', start_epoch=0, lang='ar'):
    total_step = len(data_loader)
    for epoch in range(start_epoch,num_epochs):
        for i, (images, captions) in tqdm(enumerate(data_loader), total=total_step):
            images = images.to(device)
            captions = captions.to(device)
            decoder.zero_grad()

            # Forward, backward and optimize
            features = encoder.encode_image(images)
            outputs = decoder(features, captions)
            loss = loss_function(outputs.view(-1, vocab_size), captions.view(-1))
            loss.backward()
            optimizer.step()

            # Print log info
            if i % print_every == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                target_indices = captions[0].tolist()
                prdicted_indices = outputs[0].view(-1, vocab_size).argmax(dim=1).tolist()
                print('tragt: ',vocab.indices_to_caption(target_indices, lang=lang))
                print('pred: ',vocab.indices_to_caption(prdicted_indices, lang=lang))
        # Save the model checkpoints
        if epoch % save_every == 0:
            torch.save(encoder.state_dict(), os.path.join(model_dir, '{}-{}-.ckpt'.format(model_name, epoch+1)))
            print('model saved to {}'.format(model_dir))

    return encoder


def get_decoder(embed_size, hidden_size, vocab_size, num_layers, checkpoint='', use_checkpoint=False, start_epoch=0, device='cuda'):
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, device=device)
    if use_checkpoint:
        decoder.load_state_dict(torch.load(checkpoint))
        start_epoch = int(checkpoint.split('-')[-1].split('.')[0])
    return decoder, start_epoch

def get_vocab(lang):
    if lang == 'en':
        vocab_size = vocab.Vocab_Size_english 
    if lang == 'ar':
        vocab_size = vocab.Vocab_Size_arabic 
    return vocab_size



if __name__ == '__main__':
    params = init_experiment(sys.argv[1])
    lang = params['lang']
    batch_size = params['batch_size']
    embed_size = params['embed_size']
    hidden_size = params['hidden_size']
    num_layers = params['num_layers']
    lr = params['lr']
    clip_model = params['clip_model']
    model_dir = params['model_dir']
    num_epochs = params['num_epochs']
    print_every = params['print_every']
    save_every = params['save_every']
    model_name = params['model_name']
    checkpoint = params['checkpoint']
    use_checkpoint = params['use_checkpoint']

    os.makedirs( model_dir , exist_ok=True)
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = get_vocab(lang)
    loss_function = nn.CrossEntropyLoss()

    encoder, data_loader_train = dataset.get_loader_f8k(lang, batch_size, clip_model=clip_model, device=device)
    encoder.eval()    
    decoder, start_epoch = get_decoder(embed_size, hidden_size, vocab_size, num_layers, checkpoint=checkpoint, use_checkpoint=use_checkpoint, device=device)
    decoder.to(device)
    all_params = list(decoder.parameters())  
    optimizer = torch.optim.Adam( params  = all_params , lr = lr  )

    #Training the model
    print('Training the model ...')
    train_model(
        encoder, decoder, data_loader_train, loss_function, optimizer, num_epochs,
        print_every, save_every, model_dir, model_name, start_epoch, lang)



