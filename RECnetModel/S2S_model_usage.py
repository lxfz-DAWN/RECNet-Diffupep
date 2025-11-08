import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

# 定义模型类
class AttentionRegressionNet(torch.nn.Module):
    def __init__(self, input_dim=1024, num_heads=8, num_blocks=3):
        super(AttentionRegressionNet, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.Linear = torch.nn.Linear(1152, 1024)

        # 定义 Transformer 编码器层
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_dim,
                                                         nhead=num_heads,
                                                         dim_feedforward=input_dim * 4,
                                                         activation="relu",
                                                         batch_first=True,
                                                         norm_first=True)
        # 定义 Transformer 编码器
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

        self.global_avg_pool = torch.nn.AdaptiveAvgPool3d((1, 1, input_dim))
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.005),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.005),
            torch.nn.Linear(4096, 62)
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.005),
            torch.nn.Linear(4096, 4096),
            torch.nn.Dropout(p=0.005),
            torch.nn.Linear(4096, 2)
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, labels=None, TF=None, target=['Ros', 'TF']):
        x = self.Linear(x)
        # 调整输入形状以适应 Transformer 编码器的输入要求 [seq_len, batch_size, input_dim]
        batch_size, _, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, -1)

        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)

        # 调整形状回原来的 [batch_size, 1, seq_len, input_dim]
        x = x.unsqueeze(1)

        x = self.global_avg_pool(x).squeeze(1).squeeze(1).squeeze(1)

        x_1 = self.fc1(x)
        x_2 = self.fc2(x)
        x_2 = self.softmax(x_2)

        return x_1, x_2


# 定义数据集类
class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self,
                 df,
                 score_columns,
                 esmc_hidden_column,
                 TF,
                 target=["Ros", "TF"]):
        """
        初始化数据集类。
        :param df: 包含数据的 Pandas DataFrame
        :param score_columns: 包含所有 score 列名的列表
        :param esmc_hidden_column: 包含 ESMC 隐藏层特征文件路径的列名
        """
        self.df = df
        self.score_columns = score_columns
        self.esmc_hidden_column = esmc_hidden_column
        self.TF = TF
        self.target = target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        获取指定索引的数据。
        :param idx: 数据索引
        :return: 包含输入特征和回归目标的字典
        """
        # 读取所有 score 列的数据
        scores = self.df.loc[idx, self.score_columns].values
        scores = scores.astype(float)
        scores = torch.tensor(scores, dtype=torch.bfloat16)

        # 读取 ESMC 隐藏层特征文件路径
        path = self.df.loc[idx, self.esmc_hidden_column]
        ESMC_Hidden = torch.load(path, weights_only=True).to('cpu')
        ESMC_Hidden = ESMC_Hidden[-1, :, :, :]

        # 读取TF对应的label
        TF = self.df.loc[idx, self.TF]
        if isinstance(TF, (int, float, np.float64)):
            TF = np.array([TF]).astype(float)
        else:
            TF = TF.values.astype(float)
        TF = torch.tensor(TF, dtype=torch.bfloat16)

        data = {'x': ESMC_Hidden, "labels": scores, "TF": TF, 'target': self.target}
        return data


# 自定义 collate 函数
def custom_collate_fn(batch):
    new_batch = {}
    for key in batch[0].keys():
        if key == 'target':
            new_batch[key] = [item[key] for item in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            new_batch[key] = torch.utils.data.dataloader.default_collate([item[key] for item in batch])
        elif isinstance(batch[0][key], (int, float, np.float64)):
            values = [item[key] for item in batch]
            if any(np.isnan(val) for val in values):
                values = [0 if np.isnan(val) else val for val in values]
            new_batch[key] = torch.tensor(values)
        else:
            new_batch[key] = [item[key] for item in batch]
    return new_batch


# 加载模型
def load_model(checkpoint_path, device):
    model = AttentionRegressionNet()
    state_dict = torch.load(checkpoint_path, map_location=device)
    # 移除 'module.' 前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # 移除 'module.'
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    model.to(device).to(torch.bfloat16)
    model.eval()
    return model


# 调用模型进行预测
def predict(model, dataloader, device):
    predictions_ros = []
    predictions_tf = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['x'].to(device)
            labels = batch['labels'].to(device)
            TF = batch['TF'].to(device)
            target = batch['target']

            x_1, x_2 = model(inputs, labels=labels, TF=TF, target=target)
            # 将 BFloat16 转换为 Float32
            x_1 = x_1.float()
            x_2 = x_2.float()
            predictions_ros.extend(x_1.cpu().numpy())
            predictions_tf.extend(x_2.cpu().numpy())

    return predictions_ros, predictions_tf


# 主函数
def main():
    # 设备设置
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    output_ESMC = pd.read_csv("/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_2.0/dataset/dataset/ESMC_Embedding_512/generated_EK_withpath_train_512.csv")
    score_columns = [
        'scores_total_score', 'scores_dslf_fa13',
        'scores_fa_atr', 'scores_fa_dun', 'scores_fa_elec',
        'scores_fa_intra_rep', 'scores_fa_intra_sol_xover4', 'scores_fa_rep',
        'scores_fa_sol', 'scores_hbond_bb_sc', 'scores_hbond_lr_bb',
        'scores_hbond_sc', 'scores_hbond_sr_bb', 'scores_linear_chainbreak',
        'scores_lk_ball_wtd', 'scores_omega', 'scores_overlap_chainbreak',
        'scores_p_aa_pp', 'scores_pro_close', 'scores_rama_prepro',
        'scores_ref', 'scores_yhh_planarity',
        'pack_total_score', 'pack_complex_normalized', 'pack_dG_cross',
        'pack_dG_cross/dSASAx100', 'pack_dG_separated',
        'pack_dG_separated/dSASAx100', 'pack_dSASA_hphobic',
        'pack_dSASA_int', 'pack_dSASA_polar', 'pack_delta_unsatHbonds',
        'pack_dslf_fa13', 'pack_fa_atr', 'pack_fa_dun', 'pack_fa_elec',
        'pack_fa_intra_rep', 'pack_fa_intra_sol_xover4', 'pack_fa_rep',
        'pack_fa_sol', 'pack_hbond_E_fraction', 'pack_hbond_bb_sc',
        'pack_hbond_lr_bb', 'pack_hbond_sc', 'pack_hbond_sr_bb',
        'pack_hbonds_int', 'pack_lk_ball_wtd', 'pack_nres_all',
        'pack_nres_int', 'pack_omega', 'pack_p_aa_pp', 'pack_packstat',
        'pack_per_residue_energy_int', 'pack_pro_close', 'pack_rama_prepro',
        'pack_ref', 'pack_sc_value', 'pack_side1_normalized',
        'pack_side1_score', 'pack_side2_normalized', 'pack_side2_score',
        'pack_yhh_planarity']

    # 标准化数据
    scaler = StandardScaler()
    output_ESMC[score_columns] = scaler.fit_transform(output_ESMC[score_columns])

    # 创建数据集和数据加载器
    dataset = PeptideDataset(df=output_ESMC,
                             score_columns=score_columns,
                             esmc_hidden_column='ESMC_Embedding_Path',
                             TF='TF',
                             target=['Ros', 'TF'])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    # 加载模型
    checkpoint_path = '/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_2.0/checkpoint/20250219-21.58/model/512/attn_baseline_checkpoint_epoch_3.pth'  # 替换为你实际的模型路径
    model = load_model(checkpoint_path, device)

    # 进行预测
    predictions_ros, predictions_tf = predict(model, dataloader, device)

    # 创建 DataFrame
    ros_columns = [f'Ros_{i}' for i in range(len(predictions_ros[0]))]
    tf_columns = [f'TF_{i}' for i in range(len(predictions_tf[0]))]

    ros_df = pd.DataFrame(predictions_ros, columns=ros_columns)
    tf_df = pd.DataFrame(predictions_tf, columns=tf_columns)

    result_df = pd.concat([ros_df, tf_df], axis=1)

    # 保存为 CSV 文件
    result_df.to_csv('/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_2.0/output_valid.csv', index=False)

    print("预测结果已保存到 predictions.csv")


if __name__ == "__main__":
    main()