import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparams
vocab_size = 65
block_size = 8
batch_size = 4




def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size]for i in ix])
    y = torch.stack([data[i+1:i+1+block_size]for i in ix])
    return x,y



with open('input.txt','r', encoding='utf-8') as f:
    text = f.read()

chars =sorted(list(set(text)))

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)


data = torch.tensor(encode(text),dtype=torch.long)
n = int(len(data)*0.9)
train_data = data[:n]
test_data = data[n:]
print(train_data)



xb,yb = get_batch('train')
print(xb.shape)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embdedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embdedding_table(idx)
        B, T, C = logits.shape
        nlogits = logits.view(B*T,C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(nlogits,targets)
        print(logits.shape) 
        
        test = torch.softmax(logits[:, -1, :], dim=-1)
        
        sampled_indices = torch.multinomial(test, 1)

        for i in range(batch_size):
            print(f"--- Batch item {i} ---")
            
            context_str = decode(idx[i].tolist())
            print(f"Context: {context_str}")
            predicted_char = decode(sampled_indices[i].tolist())
            print(f"Next char prediction: {predicted_char}")
            
        return logits

model = BigramLanguageModel(vocab_size)
output = model(xb,yb)
