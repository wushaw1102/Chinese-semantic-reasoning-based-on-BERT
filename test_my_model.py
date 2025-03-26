from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
import torch
import torch.nn as nn
from transformers import BertModel

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
# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForNLI()
model.load_state_dict(torch.load('bert_nli_model.pth'))
model.to(device)
model.eval()

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained("./bert_model")

# 加载测试集数据
test_data_path = './datasets/CINLID_test_df_annotation.json'

# # 使用 Pandas 读取测试集
import pandas as pd
test_df = pd.read_json(test_data_path, encoding='ANSI')

# 随机选取100条数据
test_sample = test_df.sample(n=100)

# 转换为模型输入格式
def encode_data(df):
    inputs = tokenizer(df['phrase1'].tolist(), df['phrase2'].tolist(), padding=True, truncation=True, return_tensors="pt")
    labels = df['label'].map({'entailment': 0, 'neutral': 1, 'contradiction': 2}).tolist()
    return inputs, labels

inputs, true_labels = encode_data(test_sample)

# 推理和评估
predictions = []
with torch.no_grad():
    for i in range(len(inputs['input_ids'])):
        input_ids = inputs['input_ids'][i].unsqueeze(0).to(device)
        attention_mask = inputs['attention_mask'][i].unsqueeze(0).to(device)
        token_type_ids = inputs['token_type_ids'][i].unsqueeze(0).to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        predicted_label = torch.argmax(logits, dim=1).item()

        predictions.append(predicted_label)

# 计算准确率
accuracy = accuracy_score(true_labels, predictions)

# 输出每个预测的结果与正确性
for i in range(len(test_sample)):
    print(f"Input: {test_sample.iloc[i]['phrase1']} - {test_sample.iloc[i]['phrase2']}")
    print(f"Predicted: {['entailment', 'neutral', 'contradiction'][predictions[i]]} | True: {['entailment', 'neutral', 'contradiction'][true_labels[i]]}")
    print(f"Correct: {'true' if predictions[i] == true_labels[i] else 'false'}\n")

# 输出准确率
print(f"Accuracy: {accuracy:.4f}")
