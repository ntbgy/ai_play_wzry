class TransformerConfig(object):
    from common.env import load_weights
    def __init__(
            self,
            d_model=768,
            n_layers=12,
            heads=12,
            dropout=0.0,
            load_weights=load_weights
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout
        self.load_weights = load_weights
