import torch
import torch.nn.functional as F


def accuracy_score(logits, classes):
    logits = logits.detach().cpu()
    probs = F.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)
    acc = pred == classes
    acc = acc.sum() / len(acc)
    return acc
