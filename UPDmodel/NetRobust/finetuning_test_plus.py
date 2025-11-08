import pandas as pd
import torch
from transformers import EsmTokenizer, EsmModel, Trainer, TrainingArguments
from accelerate import Accelerator

# 初始化设备和加速器
accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device

# 加载数据
train_df = pd.read_csv("train_data_plus.csv")
valid_df = pd.read_csv("valid_data_plus.csv")

# 初始化分词器
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

# 数据集类
class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, target_sequences, scores, tokenizer):
        self.sequences = sequences
        self.target_sequences = target_sequences
        self.scores = scores
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target_seq = self.target_sequences[idx]
        score = self.scores[idx]
        
        encoding = self.tokenizer(seq, target_seq, return_tensors='pt', padding='max_length', truncation=True, max_length=100)
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['labels'] = torch.tensor(score, dtype=torch.float)
        return encoding

# 创建数据集对象
train_dataset = PeptideDataset(train_df['sequence'].tolist(), train_df['target_sequence'].tolist(), train_df['score'].tolist(), tokenizer)
valid_dataset = PeptideDataset(valid_df['sequence'].tolist(), valid_df['target_sequence'].tolist(), valid_df['score'].tolist(), tokenizer)

# 模型定义
class PeptideScorePredictor(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.dense = torch.nn.Linear(base_model.config.hidden_size, 1)  # 预测单个分数

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        score = self.dense(pooled_output).squeeze(-1)

        if labels is not None:
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(score, labels)
            return loss, score
        else:
            return score

# 加载预训练模型
def load_model_from_checkpoint(checkpoint_path=None):
    base_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = PeptideScorePredictor(base_model)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model

# 设置模型检查点路径
# checkpoint_path = None  # 如果没有预训练模型文件，则设置为 None
checkpoint_path = 'results_plus/checkpoint.pth' # 如果没有预训练模型文件，则设置为 None
model = load_model_from_checkpoint(checkpoint_path)
model.to(device)

# 训练参数
training_args = TrainingArguments(
    output_dir='./results_plus',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    fp16=True,  # 启用混合精度训练
)

# 自定义Trainer以处理返回的损失和输出
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, labels=labels)
        loss = outputs[0]
        return (loss, outputs[1]) if return_outputs else loss

# 训练器
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=None,  # 取消默认数据collator
)

# 数据加载器
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
eval_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, pin_memory=True)

# 使用加速器包装模型和数据加载器
model, train_dataloader, eval_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader)

# 开始训练
# trainer.train()

# # 保存模型
# model_save_path = './results_plus/checkpoint.pth'
# torch.save(model.state_dict(), model_save_path)

# 评估模型
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 预测函数
def predict_sequence_score(sequence, target_sequence):
    inputs = tokenizer(sequence, target_sequence, return_tensors='pt', padding='max_length', truncation=True, max_length=100)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.item()

# 示例预测
sequence = "PEPTIDESEQ"
target_sequence = "TARGETSEQ"
predicted_score = predict_sequence_score(sequence, target_sequence)
print(f"Predicted score for sequence '{sequence}' and target sequence '{target_sequence}': {predicted_score}")
