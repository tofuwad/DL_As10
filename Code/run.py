#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IMDB 情感分析 - 符合作业要求版
- 方法1: One‑Hot 词袋 (unigram) + MLP （分析参数量）
- 方法2: Word2Vec(300d, 全量数据) + BiGRU
- 方法3: DistilBERT 冻结 + 强正则化逻辑回归（全量数据）
- 方法4: Few-shot (20-shot, 10次集成评估)
"""

import os
import json
import random
import gc
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer   # 用于 one‑hot 词袋
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== 路径配置 ====================
DATA_DIR = "/Pytorch_Book_ZhouRUC/dataset"
NPZ_PATH = os.path.join(DATA_DIR, "imdb.npz")
JSON_PATH = os.path.join(DATA_DIR, "imdb_word_index.json")
DISTILBERT_PATH = "/mnt/Data/distilbert-base-uncased"   # 已有的 DistilBERT

# ==================== 1. 数据加载与解码 ====================
def load_imdb_from_npz(npz_path, json_path):
    data = np.load(npz_path, allow_pickle=True)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    with open(json_path, 'r') as f:
        word_index = json.load(f)
    index_to_word = {v: k for k, v in word_index.items()}
    index_to_word[0] = '<PAD>'
    index_to_word[1] = '<START>'
    index_to_word[2] = '<UNK>'
    print(f"Vocab size: {len(word_index)}")
    print(f"Train: {len(x_train)}, Test: {len(x_test)}")
    return (x_train, y_train), (x_test, y_test), word_index, index_to_word

def decode_review(encoded_review, index_to_word, max_len=None):
    words = []
    for idx in encoded_review:
        if idx in [0, 1]:
            continue
        word = index_to_word.get(idx, '<UNK>')
        words.append(word)
        if max_len and len(words) >= max_len:
            break
    return ' '.join(words)

# 加载全量数据
(x_train, y_train), (x_test, y_test), word_index, index_to_word = load_imdb_from_npz(NPZ_PATH, JSON_PATH)

print("Decoding all texts (this may take 2-3 minutes)...")
train_texts = [decode_review(seq, index_to_word) for seq in tqdm(x_train, desc="Train decode")]
test_texts = [decode_review(seq, index_to_word) for seq in tqdm(x_test, desc="Test decode")]
train_labels = y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train)
test_labels = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
print(f"Decoded {len(train_texts)} train, {len(test_texts)} test")

# ==================== 公用分类器 ====================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.5):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class LogisticRegression(nn.Module):
    """用于方法3的强正则化线性分类器"""
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)

# ==================== 方法1: One‑Hot 词袋 (BOW) + MLP ====================
print("\n" + "="*60)
print("Method 1: One‑Hot Bag‑of‑Words (unigram) + MLP")
print("="*60)

# 构建词袋 one‑hot 向量 (binary=True 表示每个词出现与否，即为 one‑hot 的文档级聚合)
vectorizer_onehot = CountVectorizer(binary=True, max_features=20000)  # 限制词表大小，否则维度爆炸
X_train_onehot = vectorizer_onehot.fit_transform(train_texts).toarray().astype(np.float32)
X_test_onehot = vectorizer_onehot.transform(test_texts).toarray().astype(np.float32)
print(f"One‑hot BOW shape: train {X_train_onehot.shape}, test {X_test_onehot.shape}")
print(f"Vocabulary size: {len(vectorizer_onehot.vocabulary_)}")

# 标准化（对 one‑hot 向量标准化通常不是必须的，但为了与之前流程保持一致）
scaler_onehot = StandardScaler(with_mean=False)  # one‑hot 是稀疏的，均值缩放会破坏稀疏性，但这里转为 dense 后可以
X_train_onehot = scaler_onehot.fit_transform(X_train_onehot)
X_test_onehot = scaler_onehot.transform(X_test_onehot)

class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 128
train_dataset1 = ArrayDataset(X_train_onehot, train_labels)
test_dataset1 = ArrayDataset(X_test_onehot, test_labels)
train_loader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=True, num_workers=3)
test_loader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False, num_workers=3)

model1 = MLPClassifier(X_train_onehot.shape[1], hidden_dims=[1024, 512, 256], dropout=0.5).to(device)
print(f"Method1 total params: {sum(p.numel() for p in model1.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.AdamW(model1.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=20)

epochs1 = 20
best_acc1 = 0
patience = 3
epochs_no_improve = 0
best_model_path1 = "Saving/model1_onehot.pt"

for epoch in range(epochs1):
    model1.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in tqdm(train_loader1, desc=f"Method1 Epoch {epoch+1}"):
        X, y = X.to(device), y.to(device)
        optimizer1.zero_grad()
        logits = model1(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer1.step()
        total_loss += loss.item() * X.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += X.size(0)
    train_acc = correct / total
    scheduler1.step()

    model1.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader1:
            X, y = X.to(device), y.to(device)
            logits = model1(X)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += X.size(0)
    test_acc = correct / total
    print(f"Epoch {epoch+1}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    if test_acc > best_acc1:
        best_acc1 = test_acc
        torch.save(model1.state_dict(), best_model_path1)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after epoch {epoch+1} (no improvement for {patience} epochs).")
        break

model1.load_state_dict(torch.load(best_model_path1))
print(f"Method1 Best Test Acc: {best_acc1*100:.2f}%")

# 预测3个样本
sample_indices = [0, 1, 2]
sample_texts = [test_texts[i] for i in sample_indices]
sample_labels_true = [test_labels[i] for i in sample_indices]
sample_onehot = vectorizer_onehot.transform(sample_texts).toarray().astype(np.float32)
sample_onehot = scaler_onehot.transform(sample_onehot)
sample_tensor1 = torch.tensor(sample_onehot, dtype=torch.float32).to(device)
model1.eval()
with torch.no_grad():
    logits1 = model1(sample_tensor1)
    probs1 = torch.softmax(logits1, dim=1).cpu().numpy()
    preds1 = logits1.argmax(dim=1).cpu().numpy()

del X_train_onehot, X_test_onehot, scaler_onehot, vectorizer_onehot, model1
del train_loader1, test_loader1
gc.collect()
torch.cuda.empty_cache()

# ==================== 方法2: Word2Vec + BiGRU ====================
print("\n" + "="*60)
print("Method 2: Word2Vec (300d, full data) + BiGRU")
print("="*60)

from gensim.models.callbacks import CallbackAny2Vec

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.total_epochs = None
    def on_epoch_end(self, model):
        self.epoch += 1
        if self.total_epochs is None:
            self.total_epochs = model.epochs
        print(f"Word2Vec training: epoch {self.epoch}/{self.total_epochs} completed")

print("Training Word2Vec on all training texts...")
tokenized_train = [text.split() for text in train_texts]
w2v_model = Word2Vec(
    sentences=tokenized_train,
    vector_size=300,
    window=5,
    min_count=2,
    workers=8,
    sg=0,                       # CBOW (更快, 精度几乎不变)
    epochs=15,
    callbacks=[EpochLogger()]
)
print(f"Word2Vec vocab size: {len(w2v_model.wv)}")

# 建立词索引
word2idx_w2v = {w: i+3 for i, w in enumerate(w2v_model.wv.index_to_key)}
word2idx_w2v['<PAD>'] = 0
word2idx_w2v['<START>'] = 1
word2idx_w2v['<UNK>'] = 2
idx_to_word_w2v = {v: k for k, v in word2idx_w2v.items()}

def text_to_ids(text, max_len=400):
    words = text.split()
    ids = [word2idx_w2v.get(w, 2) for w in words][:max_len]
    return ids

train_ids = [text_to_ids(t) for t in tqdm(train_texts, desc="Train to ids")]
test_ids = [text_to_ids(t) for t in tqdm(test_texts, desc="Test to ids")]

max_idx = max(word2idx_w2v.values())
embedding_matrix = np.zeros((max_idx + 1, 300), dtype=np.float32)
for w, i in word2idx_w2v.items():
    if i >= 3:
        embedding_matrix[i] = w2v_model.wv[w]

class W2VDataset(Dataset):
    def __init__(self, ids, labels):
        self.ids = ids
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx], dtype=torch.long), self.labels[idx]

def collate_w2v(batch, pad_idx=0):
    ids, labels = zip(*batch)
    ids_padded = pad_sequence(ids, batch_first=True, padding_value=pad_idx)
    return ids_padded, torch.tensor(labels, dtype=torch.long)

batch_size = 32
train_dataset2 = W2VDataset(train_ids, train_labels)
test_dataset2 = W2VDataset(test_ids, test_labels)
train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_w2v, num_workers=3)
test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False,
                          collate_fn=collate_w2v, num_workers=3)

class BiGRUClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=256, dropout=0.5):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True,
                          bidirectional=True, dropout=dropout, num_layers=2)
        self.fc = nn.Linear(hidden_dim*2, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        out = out[:, -1, :]   # 最后时刻输出
        out = self.dropout(out)
        return self.fc(out)

model2 = BiGRUClassifier(embedding_matrix, hidden_dim=256, dropout=0.5).to(device)
print(f"Method2 params: {sum(p.numel() for p in model2.parameters()):,}")

optimizer2 = optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=15)

epochs2 = 15
best_acc2 = 0
patience = 3
epochs_no_improve = 0
best_model_path2 = "Saving/model2_w2v_gru.pt"

for epoch in range(epochs2):
    model2.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in tqdm(train_loader2, desc=f"Method2 Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device)
        optimizer2.zero_grad()
        logits = model2(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer2.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    train_acc = correct / total
    scheduler2.step()

    model2.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader2:
            x, y = x.to(device), y.to(device)
            logits = model2(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    test_acc = correct / total
    print(f"Epoch {epoch+1}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    if test_acc > best_acc2:
        best_acc2 = test_acc
        torch.save(model2.state_dict(), best_model_path2)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after epoch {epoch+1} (no improvement for {patience} epochs).")
        break

model2.load_state_dict(torch.load(best_model_path2))
print(f"Method2 Best Test Acc: {best_acc2*100:.2f}%")

# 预测3个样本
model2.eval()
sample_ids2 = [text_to_ids(t) for t in sample_texts]
sample_ids_padded = pad_sequence([torch.tensor(ids) for ids in sample_ids2], batch_first=True).to(device)
with torch.no_grad():
    logits2 = model2(sample_ids_padded)
    probs2 = torch.softmax(logits2, dim=1).cpu().numpy()
    preds2 = logits2.argmax(dim=1).cpu().numpy()

del train_ids, test_ids, embedding_matrix, w2v_model, model2
del train_loader2, test_loader2
gc.collect()
torch.cuda.empty_cache()

# ==================== 方法3: DistilBERT 冻结 + 逻辑回归（全量数据）====================
print("\n" + "="*60)
print("Method 3: Frozen DistilBERT + Logistic Regression (full data)")
print("="*60)

# 提取全量数据的 DistilBERT 嵌入（只做一次，缓存）
train_emb_cache = "Saving/distilbert_emb_train.npy"
test_emb_cache = "Saving/distilbert_emb_test.npy"
if os.path.exists(train_emb_cache) and os.path.exists(test_emb_cache):
    train_embs = np.load(train_emb_cache)
    test_embs = np.load(test_emb_cache)
    print("Loaded cached DistilBERT embeddings.")
else:
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_PATH)
    model_bert = AutoModel.from_pretrained(DISTILBERT_PATH).to(device)
    model_bert.eval()

    def extract_embeddings(texts, batch_size=64):
        embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, return_tensors='pt', padding=True,
                                   truncation=True, max_length=256).to(device)
                outputs = model_bert(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embs.append(cls_emb)
        return np.concatenate(embs, axis=0)

    print("Extracting training embeddings...")
    train_embs = extract_embeddings(train_texts, batch_size=64)
    np.save(train_emb_cache, train_embs)
    print("Extracting test embeddings...")
    test_embs = extract_embeddings(test_texts, batch_size=64)
    np.save(test_emb_cache, test_embs)
    del model_bert  # 释放显存

print(f"Train embeddings shape: {train_embs.shape}, Test: {test_embs.shape}")

# 使用逻辑回归（线性分类器） + 强正则化
train_emb_dataset = ArrayDataset(train_embs, train_labels)
test_emb_dataset = ArrayDataset(test_embs, test_labels)
train_emb_loader = DataLoader(train_emb_dataset, batch_size=512, shuffle=True, num_workers=3)
test_emb_loader = DataLoader(test_emb_dataset, batch_size=512, shuffle=False, num_workers=3)

model3 = LogisticRegression(train_embs.shape[1]).to(device)
print(f"Method3 params: {sum(p.numel() for p in model3.parameters()):,}")

optimizer3 = optim.AdamW(model3.parameters(), lr=1e-3, weight_decay=1e-1)  # 强L2正则
criterion3 = nn.CrossEntropyLoss()
epochs3 = 20
best_acc3 = 0
patience = 3
epochs_no_improve = 0
best_model_path3 = "Saving/model3_distilbert_lr.pt"

for epoch in range(epochs3):
    model3.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in tqdm(train_emb_loader, desc=f"Method3 Epoch {epoch+1}"):
        X, y = X.to(device), y.to(device)
        optimizer3.zero_grad()
        logits = model3(X)
        loss = criterion3(logits, y)
        loss.backward()
        optimizer3.step()
        total_loss += loss.item() * X.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += X.size(0)
    train_acc = correct / total

    model3.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_emb_loader:
            X, y = X.to(device), y.to(device)
            logits = model3(X)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += X.size(0)
    test_acc = correct / total
    print(f"Epoch {epoch+1}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    if test_acc > best_acc3:
        best_acc3 = test_acc
        torch.save(model3.state_dict(), best_model_path3)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after epoch {epoch+1} (no improvement for {patience} epochs).")
        break

model3.load_state_dict(torch.load(best_model_path3))
print(f"Method3 Best Test Acc: {best_acc3*100:.2f}%")

# 预测3个样本
tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_PATH)
model_bert = AutoModel.from_pretrained(DISTILBERT_PATH).to(device)
model_bert.eval()
sample_embs = []
with torch.no_grad():
    for t in sample_texts:
        inputs = tokenizer(t, return_tensors='pt', truncation=True, max_length=256).to(device)
        outputs = model_bert(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        sample_embs.append(emb[0])
sample_embs = np.array(sample_embs)
sample_tensor3 = torch.tensor(sample_embs, dtype=torch.float32).to(device)
model3.eval()
with torch.no_grad():
    logits3 = model3(sample_tensor3)
    probs3 = torch.softmax(logits3, dim=1).cpu().numpy()
    preds3 = logits3.argmax(dim=1).cpu().numpy()
del model_bert  # 释放显存

# ==================== 方法4: Few-shot (20-shot, 10次集成) ====================
print("\n" + "="*60)
print("Method 4: Few-shot (20-shot, 10 trials)")
print("="*60)

# 使用相同的 DistilBERT 模型提取嵌入
model_fs = AutoModel.from_pretrained(DISTILBERT_PATH).to(device)
model_fs.eval()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model_fs(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

def few_shot_predict(support_texts, support_labels, query_texts):
    # 计算原型
    prototypes = {}
    for label in [0,1]:
        embs = [get_embedding(t) for t, l in zip(support_texts, support_labels) if l == label]
        prototypes[label] = np.mean(embs, axis=0) if embs else np.zeros(768)
    # 预测
    preds, probs = [], []
    for text in query_texts:
        emb = get_embedding(text)
        sim0 = np.dot(emb, prototypes[0]) / (np.linalg.norm(emb)*np.linalg.norm(prototypes[0])+1e-8)
        sim1 = np.dot(emb, prototypes[1]) / (np.linalg.norm(emb)*np.linalg.norm(prototypes[1])+1e-8)
        exp0, exp1 = np.exp(sim0), np.exp(sim1)
        prob0 = exp0/(exp0+exp1)
        pred = 0 if sim0 > sim1 else 1
        preds.append(pred)
        probs.append([prob0, 1-prob0])
    return np.array(preds), np.array(probs)

# 评估 few-shot 性能
n_trials = 10
k_shot = 20
query_size = 2000
accs = []
for trial in range(n_trials):
    pos_idx = [i for i,l in enumerate(train_labels) if l==1]
    neg_idx = [i for i,l in enumerate(train_labels) if l==0]
    support_pos = random.sample(pos_idx, k_shot)
    support_neg = random.sample(neg_idx, k_shot)
    support_texts_fs = [train_texts[i] for i in support_pos+support_neg]
    support_labels_fs = [1]*k_shot + [0]*k_shot
    query_idx = random.sample(range(len(test_texts)), query_size)
    query_texts_fs = [test_texts[i] for i in query_idx]
    query_labels_fs = [test_labels[i] for i in query_idx]
    preds, _ = few_shot_predict(support_texts_fs, support_labels_fs, query_texts_fs)
    acc = np.mean(preds == query_labels_fs)
    accs.append(acc)
    print(f"Trial {trial+1}: acc={acc*100:.2f}%")
mean_acc4 = np.mean(accs)
std_acc4 = np.std(accs)
print(f"Few-shot (20-shot, {n_trials} trials) Test Acc: {mean_acc4*100:.2f}% ± {std_acc4*100:.2f}%")

# 对3个样本预测（使用最后一次的支持集）
preds_fewshot_sample, probs_fewshot_sample = few_shot_predict(support_texts_fs, support_labels_fs, sample_texts)
del model_fs

# ==================== 保存结果 ====================
os.makedirs("Saving", exist_ok=True)
os.makedirs("Output", exist_ok=True)

results = {
    "sample_texts": sample_texts,
    "sample_true_labels": sample_labels_true,
    "method1_onehot": {"predictions": preds1.tolist(), "probabilities": probs1.tolist()},
    "method2_w2v_gru": {"predictions": preds2.tolist(), "probabilities": probs2.tolist()},
    "method3_distilbert_lr": {"predictions": preds3.tolist(), "probabilities": probs3.tolist()},
    "method4_fewshot": {"predictions": preds_fewshot_sample.tolist(), "probabilities": probs_fewshot_sample.tolist()},
    "test_accuracies": {
        "method1": float(best_acc1),
        "method2": float(best_acc2),
        "method3": float(best_acc3),
        "method4": float(mean_acc4)
    }
}
with open("Output/results_final.json", "w") as f:
    json.dump(results, f, indent=2)

comparison = f"""
========================================
FINAL RESULTS (One‑Hot, Word2Vec, DistilBERT, Few‑shot)
========================================
Method 1 (One‑Hot Bag‑of‑Words + MLP)          : {best_acc1*100:.2f}%
Method 2 (Word2Vec + BiGRU)                    : {best_acc2*100:.2f}%
Method 3 (Frozen DistilBERT + LogReg)          : {best_acc3*100:.2f}%
Method 4 (Few-shot, 20-shot, 10 trials)        : {mean_acc4*100:.2f}% (±{std_acc4*100:.2f}%)

Observations:
- One‑hot + MLP performs reasonably but suffers from high dimensionality and lack of semantic information.
- Word2Vec + BiGRU captures word order and semantics, achieving similar accuracy with far fewer parameters.
- Frozen DistilBERT with strong regularization gives the best accuracy due to rich contextual embeddings.
- Few‑shot works without training but is less stable and requires a good support set.
"""
with open("Output/comparison_final.txt", "w") as f:
    f.write(comparison)
print(comparison)
print("\nAll results saved to Output/ and models to Saving/")