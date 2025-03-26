import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import torch.nn as nn
from transformers import BertModel
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from sklearn.metrics import classification_report
import sys

# 数据路径
train_path = "./datasets/CINLID_train_df.json"
test_path = "./datasets/CINLID_test_df_annotation.json"


# 读取数据
def load_data(filepath):
    with open(filepath, 'r', encoding='ANSI') as f:
        data = json.load(f)
    return pd.DataFrame(data)


# 加载数据集
train_data = load_data(train_path)
test_data = load_data(test_path)

# 检查数据
print(train_data.head())  # 显示训练集前几行
print(test_data.head())   # 显示测试集前几行

# 标签映射
label_mapping = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
train_data['label'] = train_data['label'].map(label_mapping)
test_data['label'] = test_data['label'].map(label_mapping)

# 加载预训练的BERT分词器（从本地加载）
tokenizer = BertTokenizer.from_pretrained("./bert_model")


# 数据预处理函数
def preprocess(data, tokenizer, max_len=128):
    inputs = tokenizer(
        list(data['phrase1']),
        list(data['phrase2']),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    labels = data['label'].values
    return inputs, labels


train_inputs, train_labels = preprocess(train_data, tokenizer)
test_inputs, test_labels = preprocess(test_data, tokenizer)


# 自定义BERT模型
class BertForNLI(nn.Module):
    def __init__(self, model_name="./bert_model", num_labels=3):
        super(BertForNLI, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)  # Load from local directory
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.pooler_output
        logits = self.classifier(cls_output)
        return logits


# 创建DataLoader
train_dataset = TensorDataset(
    train_inputs['input_ids'],
    train_inputs['attention_mask'],
    train_inputs['token_type_ids'],
    torch.tensor(train_labels)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 模型初始化
model = BertForNLI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

# 保存损失和准确率的列表
losses = []
accuracies = []


# 训练过程
def train(model, loader, optimizer, criterion, device, epochs=6):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_preds = 0
        total_preds = 0

        for batch_idx, batch in enumerate(loader):
            input_ids, attention_mask, token_type_ids, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 计算准确率
            _, preds = torch.max(logits, dim=1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)

            # 每100步打印一次
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}, Step {batch_idx}, Loss: {loss.item()}")

        avg_loss = total_loss / len(loader)
        accuracy = correct_preds / total_preds
        losses.append(avg_loss)
        accuracies.append(accuracy)

        print(f"Epoch {epoch + 1}, Total Loss: {avg_loss}, Accuracy: {accuracy}")

        # 强制刷新输出
        sys.stdout.flush()


# 训练模型
train(model, train_loader, optimizer, criterion, device)

# 保存损失和准确率到CSV文件
losses_df = pd.DataFrame({
    'Epoch': [i + 1 for i in range(len(losses))],
    'Loss': losses,
    'Accuracy': accuracies
})
losses_df.to_csv('training_losses_accuracies_four.csv', index=False)


# 测试数据加载
test_dataset = TensorDataset(
    test_inputs['input_ids'],
    test_inputs['attention_mask'],
    test_inputs['token_type_ids'],
    torch.tensor(test_labels)
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 评估过程
def evaluate(model, loader, device):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, token_type_ids, labels = [x.to(device) for x in batch]
            logits = model(input_ids, attention_mask, token_type_ids)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    print(classification_report(true_labels, preds, target_names=['entailment', 'neutral', 'contradiction']))


# 评估模型
evaluate(model, test_loader, device)

# 保存模型
model_save_path = "bert_nli_model_four.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
