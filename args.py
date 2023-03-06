
#create a class called demo args to store the arguments
class DemoArgs:
    def __init__(self):
        self.data = './data/embeddings/english_CLIP-ViT-B-32_embeddings.pkl'
        self.lang = 'english'
        self.out_dir = './checkpoints'
        self.output_prefix = 'english_exp_2'
        self.epochs = 3
        self.save_every = 1
        self.prefix_length = 10
        self.prefix_length_clip = 10
        self.bs = 3
        self.only_prefix = True
        self.mapping_type = 'transformer'
        self.num_layers = 8
        self.is_rn = False
        self.normalize_prefix = False
        self.get_bleu = False
        self.prefix_dim = 640 if self.is_rn else 512
