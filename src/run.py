import json
import os
import re
import urllib

import lxml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sklearn
import torch
import tqdm
from bs4 import BeautifulSoup
from langdetect import detect
from lxml import html
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GPTNeoXForSequenceClassification)
from trl import (AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer,
                 create_reference_model)
from trl.core import respond_to_batch

from data import load_survey, fetch_paper_ids, fetch_abstracts
from model import load_tokenizer, load_base_model, get_embeddings, measure_similarity, compute_reward, prepare_data, load_data
from time import time


# Data

t_0 = time()
questions, options, responses = load_survey()
ids = fetch_paper_ids(questions, load=True)
data = fetch_abstracts(ids, load=True)

print(f"Data loaded in {time() - t_0:.2f} seconds")


# Model

t_0 = time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "EleutherAI/pythia-70m-deduped"
tokenizer = load_tokenizer(model_name)
base_model = load_base_model(tokenizer, model_name).to(device)

print(f"Model loaded in {time() - t_0:.2f} seconds")

t_0 = time()
texts = data["text"].values.tolist()
embeddings = get_embeddings(texts, base_model, tokenizer, device=device, load=True)
similarities = measure_similarity(embeddings, data)
rewards = compute_reward(similarities)

print(f"Reward computed in {time() - t_0:.2f} seconds")

t_0 = time()
data = prepare_data(data, rewards)
batch_size = 128
train_loader, test_loader, val_loader = load_data(
    data, tokenizer, batch_size=batch_size
)

print(f"Data prepared in {time() - t_0:.2f} seconds")


# Train

r_model = GPTNeoXForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,
    pad_token_id=tokenizer.pad_token_id,
).to(device)

import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

logging.basicConfig(filename="/mnt/ssd-cluster/training.log", level=logging.INFO)

r_model = GPTNeoXForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,
    pad_token_id=tokenizer.pad_token_id,
).to(device)

def evaluate(model, val_dataloader):
    model.eval()
    val_loss = []
    val_acc = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss.append(outputs.loss.item())
            preds = torch.argmax(outputs.logits, dim=1)
            acc = accuracy_score(labels.cpu(), preds.cpu())
            val_acc.append(acc)
    return sum(val_loss)/len(val_loss), sum(val_acc)/len(val_acc)

def train_r(model, train_dataloader, val_dataloader, lr=1e-3, n_epochs=10, save=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    train_loss_values = []
    val_loss_values = []
    for epoch in tqdm(range(n_epochs), total=n_epochs):
        model.train()
        train_loss = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        avg_train_loss = sum(train_loss) / len(train_loss)
        train_loss_values.append(avg_train_loss)
        val_loss, val_acc = evaluate(model, val_dataloader)
        val_loss_values.append(val_loss)
        logging.info(f"Epoch: {epoch}, Training Loss: {avg_train_loss}, Validation Loss: {val_loss}, Validation Acc: {val_acc}")
        scheduler.step()
    if save:
        torch.save(model.state_dict(), save)
    # plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_values, label='Training Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.savefig("/mnt/ssd-cluster/plots/training-validation-loss.png")
    return model
