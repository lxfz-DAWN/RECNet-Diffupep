import yaml
import torch
import torch.nn as nn

from stripedhyena.utils import dotdict
from stripedhyena.model import StripedHyena
from stripedhyena.ESMCembeding import ESMCtokenizer




class HyenaClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hyena = StripedHyena(config)
        
        # 分类头相关参数
        self.pooling_method = "first"
        self.dropout_prob = 0.1
        self.num_classes = 2
        self.hidden_size = 1152

        # 分类头结构
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 2048),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(2048, self.num_classes)
        )
        # # 初始化分类层权重
        # nn.init.xavier_normal_(self.classifier.weight)
        # self.classifier.bias.data.zero_()       
        
    def forward(self, input_ids, padding_mask=None, labels=None):
        # 获取序列特征
        features, _ = self.hyena(input_ids, padding_mask=padding_mask)
        
        # 池化处理
        if self.pooling_method == "first":
            pooled = features[:, 0, :]  # 取第一个token
        elif self.pooling_method == "mean":
            if padding_mask is not None:
                # 考虑padding的均值计算
                seq_lengths = padding_mask.sum(dim=1).unsqueeze(1)
                pooled = (features * padding_mask.unsqueeze(-1)).sum(dim=1) / seq_lengths
            else:
                pooled = features.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")
        
        # 分类预测
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": pooled
        }        


if __name__ == "__main__":

    config_path = "/home/users/hcdai/AI-peptide/Seq2Score/Henya/S2Shy/hyena/hyena_config.yml"

    with open(config_path, "r") as f:
        config = dotdict(yaml.safe_load(f), Loader = yaml.FullLoader)

    model = StripedHyena(config)
    model.to("cuda")

    tokenizer = model.tokenizer

    # print(model)

    seq = "GGGGSVAVENAL"

    print(f"seq:{seq}")

    seq = tokenizer.tokenize(seq)

    print(f"tokenized seq:{seq}")

    last = model(seq)

    print(f":{last[0].shape}")