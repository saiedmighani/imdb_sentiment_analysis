from torch.utils.data import Dataset
from transformers import AutoTokenizer


class IMDBDataset(Dataset):
    def __init__(self, dataset_split, tokenizer_name='distilbert-base-uncased', max_length=512):
        self.dataset = dataset_split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = float(self.dataset[idx]['label'])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label,
            'length': (encoding['input_ids'] != 0).sum().item()
        }
