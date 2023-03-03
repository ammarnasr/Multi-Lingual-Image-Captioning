
#create a class called demo args to store the arguments
class DemoArgs:
    def __init__(self):
        self.data = './data/embeddings/english_CLIP-ViT-B-32_embeddings.pkl'
        self.lang = 'english'
        self.out_dir = './checkpoints'
        self.output_prefix = 'english_exp_1'
        self.epochs = 30
        self.save_every = 3
        self.prefix_length = 10
        self.prefix_length_clip = 10
        self.bs = 40
        self.only_prefix = True
        self.mapping_type = 'transformer'
        self.num_layers = 8
        self.is_rn = False
        self.normalize_prefix = False

