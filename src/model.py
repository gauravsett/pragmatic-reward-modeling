import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import cosine_similarity

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        return_dict=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_base_model(tokenizer, model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        pad_token_id=tokenizer.eos_token_id,
        return_dict=True
    )
    return model

def get_embeddings(texts, model, tokenizer, device="cpu", load=None, save=None):
    if load:
        embeddings = torch.load(load).to(device)
        return embeddings
    embeddings = torch.empty((len(texts), model.config.hidden_size)).to(device)
    n = len(texts)
    for i in tqdm(range(n), total=n):
        input_ids = tokenizer(texts[i], return_tensors="pt").input_ids
        if input_ids.shape[1] > 512:
            input_ids = input_ids[:, :512]
        with torch.no_grad():
            output = model(input_ids, output_hidden_states=True)
        embeddings[i] = output.hidden_states[-1][0, -1, :]
    if save:
        torch.save(embeddings, save)
    return embeddings

def measure_similarity(embeddings, data):
    indices_contemporary = data.loc[data["year"] == 2020].index.tolist()
    indices_historical = data.loc[data["year"] < 2020].index.tolist()
    embeddings_historical = embeddings[indices_historical]
    embeddings_contemporary = embeddings[indices_contemporary]
    similarities = cosine_similarity(embeddings_contemporary, embeddings_historical)
    return similarities

def compute_reward(similarities, k=10):
    top_k_indices = np.argsort(similarities, axis=1)[:, -k:]
    rows, _ = np.ogrid[:similarities.shape[0], :k]
    similarities[:] = 0
    similarities[rows, top_k_indices] = 1
    rewards = np.sum(similarities, axis=0)
    return rewards

def prepare_data(data, rewards):
    data = data.loc[data["year"] < 2020].reset_index(drop=True)
    data["reward"] = rewards
    data = data[["text", "reward"]]
    return data

class PaperDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.data = data["text"].tolist()
        self.rewards = torch.tensor(data["reward"].tolist())
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        if self.tokenizer:
            encoding = self.tokenizer(
                text, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
            )
            input_ids = encoding["input_ids"].squeeze().int()
            attention_mask = encoding["attention_mask"].squeeze()
            label = self.rewards[index]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label,
            }

def load_data(data, tokenizer, batch_size=128):
    dataset = PaperDataset(data, tokenizer)
    train_size = int(0.7 * len(dataset))
    test_size = int(0.2 * len(dataset))
    val_size = len(dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size, val_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, val_loader