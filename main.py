import tools
import plotting
import torch
import numpy as np
import sys


if __name__ == '__main__':
    # read in system arguments
    mode  = sys.argv[1]
    
    if mode == 'ex1':
        print('Running Example 1')
        # initialize the parameters
        image_paths = ['sample_image/1000268201_693b08cb0e.jpg', 'sample_image/1001773457_577c3a7d70.jpg']
        texts = [
            'child in a pink dress is climbing up a set of stairs in an entry way',
            'طفلة صغيرة تتسلق إلى مسرح خشبي',
            'A dog and a cat',
            'نساء في السودان',
        ]
        english_captions = [texts[0], texts[2]]
        arabic_captions = [texts[1], texts[3]]
        all_captions = {'English': english_captions, 'Arabic': arabic_captions}

        #set device to cuda if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load the models
        tokenizer = tools.load_tokenizer()
        multi_clip_model = tools.load_multilingual_clip(device='cpu')
        clip_model, preprocess = tools.load_clip(device=device)

        # Load the images
        input_images, org_images = tools.load_images(image_paths, preprocess, device=device)

        # Get the embeddings
        language_embs = {}
        with torch.no_grad():
            image_embs = clip_model.encode_image(input_images).float().cpu()
            for lang, captions in all_captions.items():
                language_embs[lang] = multi_clip_model(captions, tokenizer).float().cpu()


        # CLIP Temperature scaler
        logit_scale = clip_model.logit_scale.exp().float().to('cpu')

        # Compare the embeddings
        language_logits = {}
        for lang, embs in language_embs.items():
            language_logits[lang] = tools.compare_embeddings(logit_scale, image_embs, embs)

        # Plot the heatmap
        for lang, (img_logits, txt_logits) in language_logits.items():
            # Convert Logits into Softmax predictions
            probs = img_logits.softmax(dim=-1).cpu().detach().numpy()
            # Transpose so that each column is the softmax for each picture over the texts
            probs = np.around(probs, decimals=2).T * 100
            print("Language: {}".format(lang))
            plotting.plot_heatmap(probs)

    elif mode == 'ex2':
        print('Running Example 2')