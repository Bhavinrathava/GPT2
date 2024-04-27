import torch.nn as nn 
from starter import Decoder
import torch.nn.functional as F

class GPT2(nn.Module):
    def __init__(self, vocab_size, d_model_decoder, N_decoder, heads_decoder, dropout_decoder):
        super(GPT2, self).__init__()
        self.vocab_size, self.d_model, self.N_decoder, self.heads, self.dropout = vocab_size, d_model_decoder, N_decoder, heads_decoder, dropout_decoder
    
        self.decoder = Decoder(self.vocab_size, self.d_model, self.N_decoder, self.heads, self.dropout)
        
        # Linear, Softmax is not needed
        self.out = nn.Linear(self.d_model, self.vocab_size)
        self.out.weight = self.decoder.embeddings.weight # tie weights

    def forward(self, x, trg_mask):
        #Pass to decoder : forward(self, trg, e_outputs, src_mask, trg_mask) 
        x = self.decoder(x, trg_mask)
        x = self.out(x)
        return x
    

def train():
    model = GPT2(50257, 728, 2, 3, 0.2)
    print(model)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    
if __name__ == "__main__":
    train()
    
    
    
    # Tying source and target variables -> meaning we are using same vocanbulary for bot input and output
    # can be done by ensuring that the embedding layer serves both as input embedder and output layer 
    # Can be done by tying the weights of the output layer to the embedding layer
    #