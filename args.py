
#create a class called demo args to store the arguments
class DemoArgs:
    def __init__(self):
        self.data = './data/embeddings/arabic_CLIP-ViT-B-32_embeddings.pkl'
        self.lang = 'arabic'
        self.out_dir = './checkpoints'
        self.output_prefix = 'arabic_exp_3'
        self.epochs = 1
        self.save_every = 1
        self.prefix_length = 10
        self.prefix_length_clip = 10
        self.bs = 2
        self.only_prefix = True
        self.mapping_type = 'transformer'
        self.num_layers = 8
        self.is_rn = False
        self.normalize_prefix = False
        self.get_bleu = False