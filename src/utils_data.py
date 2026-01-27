from torch.utils.data import Dataset
from typing import List, Dict

class ListOfDictsDataset(Dataset):
    def __init__(self, data: List[Dict]):
        super().__init__()
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)