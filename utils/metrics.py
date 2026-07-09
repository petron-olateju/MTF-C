import torch
import torch.nn as nn
import torch.nn.functional as F


def accuracy_score(logits, classes):
    logits = logits.detach().cpu()
    probs = F.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)
    acc = pred == classes
    acc = acc.sum() / len(acc)
    return acc

class InterIntraClass_Similarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, y):
        sim = embeddings @ embeddings.T

        # 2. Flatten labels to ensure a 1D tensor shape (batch_size,)
        labels = y.view(-1, 1)

        # 3. Create boolean masks for intra-class and inter-class pairs
        # A True value at [i, j] means sample i and sample j have the same label
        class_mask = (labels == labels.T)

        # 4. Exclude self-similarity diagonal (a sample matched with itself is always 1.0)
        identity_mask = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        intra_mask = class_mask & ~identity_mask
        inter_mask = ~class_mask

        # 5. Extract the similarities using the masks
        intra_sims = sim[intra_mask]
        inter_sims = sim[inter_mask]

        inter_sim = inter_sims.mean().item()
        intra_sim = intra_sims.mean().item()

        return inter_sim, intra_sim