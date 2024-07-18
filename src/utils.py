import torch
import torch.nn as nn

def collate_fn(data):
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets = [torch.tensor([ord(c) for c in cap]) for cap in captions]
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return images, targets, lengths