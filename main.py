import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
vocab_size = 65
block_size = 8
batch_size = 4
n_embd = 32
max_iters = 10000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
test_data = data[n:]
print(train_data)


xb, yb = get_batch("train")
print(xb.shape)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        if T > block_size:
            pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        else:
            pos_emb = pos_emb.unsqueeze(0).expand(B, T, -1)
        x = tok_emb + pos_emb
        logits = self.lm_head(x)
        if targets == None:
            return logits, None
        B, T, C = logits.shape
        nlogits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(nlogits, targets)

        return logits, loss

    def generate(self, max_new_tokens, idx):
        output = ""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
            output += decode(idx_next[0].cpu().tolist())
        print(output)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx=context, max_new_tokens=500)[0].tolist()))
