from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.signals = data
        self.labels = labels
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        # Return a single sample
        return self.signals[idx], self.labels[idx]