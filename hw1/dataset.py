from datasets import load_dataset
import tiktoken
import torch
from torch.utils.data import Dataset

SEQ_LEN = 128

enc = tiktoken.get_encoding("cl100k_base")
VOCAB_SIZE = enc.n_vocab
EOT_TOKEN = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]


def extract_texts(split):
    texts = split["text"]
    return [t for t in texts if t and t.strip()]


def load_datasets(seq_len=SEQ_LEN):
    ds = load_dataset("dkagramanyan/horoscopes_ru")
    train_texts = extract_texts(ds["train"])
    test_texts = extract_texts(ds["test"])
    return TokenDataset(train_texts, seq_len), TokenDataset(test_texts, seq_len)


class TokenDataset(Dataset):
    def __init__(self, texts, seq_len=SEQ_LEN):
        tokens = []
        for text in texts:
            tokens.extend(enc.encode_ordinary(text))
            tokens.append(EOT_TOKEN)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]
