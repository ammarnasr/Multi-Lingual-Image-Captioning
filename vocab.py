import pandas as pd


#Arabic Preprocessing
def arabic_preprocessing():
    merged_captions = pd.read_pickle('data/Flickr8k_text/merged_captions.pkl')
    #count the number of words in all arabic captions
    all_words_arabic = []
    for caption in merged_captions['arabic_caption']:
        for c in caption:
            for word in c.split():
                all_words_arabic.append(word.lower())

    # count unique words in all words arabic
    unique_words_arabic = set(all_words_arabic)
    Vocab_Size_arabic =  len(unique_words_arabic) + 4 # 4 for start, end, unknown and padding tokens

    #create a dictionary of all unique words and their index arabic
    word2idx_arabic = {}
    idx2word_arabic = {}
    for i, word in enumerate(unique_words_arabic):
        word2idx_arabic[word] = i
        idx2word_arabic[i] = word

    #add start and end tokens to the dictionary arabic
    word2idx_arabic['<start>'] = len(word2idx_arabic)
    idx2word_arabic[len(idx2word_arabic)] = '<start>'
    word2idx_arabic['<end>'] = len(word2idx_arabic)
    idx2word_arabic[len(idx2word_arabic)] = '<end>'

    #add unknown token to the dictionary arabic
    word2idx_arabic['<unk>'] = len(word2idx_arabic)
    idx2word_arabic[len(idx2word_arabic)] = '<unk>'

    #add padding token to the dictionary arabic
    word2idx_arabic['<pad>'] = len(word2idx_arabic)
    idx2word_arabic[len(idx2word_arabic)] = '<pad>'
    return word2idx_arabic, idx2word_arabic, Vocab_Size_arabic

#English Preprocessing
def english_preprocessing():
    merged_captions = pd.read_pickle('data/Flickr8k_text/merged_captions.pkl')

    # count the number of words in all english captions
    all_words = []
    for caption in merged_captions['english_caption']:
        for c in caption:
            for word in c.split():
                all_words.append(word.lower())

    # count unique words in all words
    unique_words = set(all_words)
    Vocab_Size =  len(unique_words) + 4 # 4 for start, end, unknown and padding tokens

    #create a dictionary of all unique words and their index
    word2idx = {}
    idx2word = {}
    for i, word in enumerate(unique_words):
        word2idx[word] = i
        idx2word[i] = word

    #add start and end tokens to the dictionary
    word2idx['<start>'] = len(word2idx)
    idx2word[len(idx2word)] = '<start>'
    word2idx['<end>'] = len(word2idx)
    idx2word[len(idx2word)] = '<end>'

    #add unknown token to the dictionary
    word2idx['<unk>'] = len(word2idx)
    idx2word[len(idx2word)] = '<unk>'

    #add padding token to the dictionary
    word2idx['<pad>'] = len(word2idx)
    idx2word[len(idx2word)] = '<pad>'
    return word2idx, idx2word, Vocab_Size

word2idx_arabic, idx2word_arabic, Vocab_Size_arabic = arabic_preprocessing()
word2idx_english, idx2word_english, Vocab_Size_english = english_preprocessing()

#function to convert a caption to a list of indices
def caption_to_indices(caption , lang = 'en', size = 10):
    indices = []
    if lang == 'en':
        w2i = word2idx_english
    if lang == 'ar':
        w2i = word2idx_arabic
    #truncate captions longer than 10 words
    if len(caption.split()) > size:
        caption = ' '.join(caption.split()[:size])
    
    # add start token
    indices.append(w2i['<start>'])
    # add indices of words in caption
    for word in caption.split():
        # if word is not in the dictionary, add the index of the unknown token
        if word.lower() not in w2i:
            indices.append(w2i['<unk>'])
        else:
            indices.append(w2i[word.lower()])

    # pad the rest of the caption with the padding token
    if len(indices) < size + 1:
        indices.extend([w2i['<pad>']] * (size + 1 - len(indices)))
    # add end token
    indices.append(w2i['<end>'])
    return indices

#function to convert a list of indices to a caption
def indices_to_caption(indices, lang = 'en'):
    if lang == 'en':
        i2w = idx2word_english
        w2i = word2idx_english
    if lang == 'ar':
        i2w = idx2word_arabic
        w2i = word2idx_arabic
    caption = ''
    for i in indices:
        # if the index is the padding token, stop
        if i == w2i['<pad>']:
            break
        # if the index is the end token, stop
        if i == w2i['<end>']:
            break
        # if the index is not the start token, add the word to the caption
        if i != w2i['<start>']:
            caption += i2w[i] + ' '
    return caption