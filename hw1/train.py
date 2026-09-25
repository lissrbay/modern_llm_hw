import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataclasses import dataclass
from dataset import load_datasets, SEQ_LEN, VOCAB_SIZE
from model import LLamaModel, AttnParams
from datasets import load_dataset
import tiktoken

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


@dataclass
class ARGS_DEFAULT:
    DIM: 32
    N_LAYERS: 1
    N_HEADS: 8
    N_KV_HEADS: 2
    BATCH_SIZE: 64
    EPOCHS: 1
    LR: 1e-3
    LOG_EVERY: 20
    LOG_DIR: "runs/llama"

class LlamaLightning(pl.LightningModule):
    def __init__(self, dim, n_layers, n_heads, n_kv_heads, seq_len, lr):
        super().__init__()
        attn_params = AttnParams(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=seq_len,
        )
        self.model = LLamaModel(
            dim=dim,
            attn_params=attn_params,
            decoder_blocks_num=n_layers,
            vocab_size=VOCAB_SIZE,
        )
        self.lr = lr

    def training_step(self, batch):
        x, y = batch
        _, loss = self.model(x, y)
        self.log("loss/train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        _, loss = self.model(x, y)
        self.log("loss/test", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def main(args = ARGS_DEFAULT):
    print("Loading and tokenizing dataset...")
    train_dataset, test_dataset = load_datasets(seq_len=SEQ_LEN)
    print(f"Train samples: {len(train_dataset)}, test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE)

    model = LlamaLightning(dim=args.DIM,
                           n_layers=args.N_LAYERS, 
                           n_heads=args.N_HEADS, 
                           n_kv_heads=args.N_KV_HEADS, 
                           seq_len=SEQ_LEN,
                           lr=args.LR
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    logger = TensorBoardLogger(save_dir=args.LOG_DIR, name="")
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.LOG_DIR,
        filename="llama",
        save_top_k=1,
        monitor="loss/test",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_cb],
        log_every_n_steps=args.LOG_EVERY,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    main()
