from torch.utils.data import DataLoader,Dataset
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
from accelerate import Accelerator

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device
print("device:",torch.cuda.is_available())


def ESMC_embedding(sequence:str):
   '''调用ESMC模型进行embedding
   
   输入：蛋白质序列
   输出：embedding后的隐层特征（Size = [36, 1, 68, 1152]）
   '''
   protein = ESMProtein(sequence=sequence)
   client = ESMC.from_pretrained("esmc_600m").to(device) # or "cpu"
   protein_tensor = client.encode(protein)
   logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True))
   return logits_output.hidden_states

def show_plot(iteration,loss,label):
    plt.figure(figsize=(10, 6))  # 设置图表大小
    plt.plot(iteration, loss, color='blue', linewidth=2)  # 绘制损失曲线
    plt.title(label, fontsize=16)  # 设置标题
    plt.legend()  # 添加图例
    plt.grid(True)  # 显示网格
    plt.tight_layout()  # 自适应布局
    plt.show()  # 显示图表

# 读取数据
class ProteinPairsDataset(Dataset):
    
    def __init__(self, data, NormalizedSequenceLength, Seq2Tensor_path = './Seq2Tensor', transform=None):
        """
        初始化数据集。

        参数:
            data (DataFrame): 包含序列对及其标签的 DataFrame。
            Seq2Tensor_path (str): 存储张量数据的路径。
        """
        self.data = data  
        self.NormalizedSequenceLength = NormalizedSequenceLength
        self.Seq2Tensor_path = Seq2Tensor_path
        self.ProteinPairsDataTensors = set([path.split('/')[-1].rsplit('.', 1)[0] for path in glob.glob(f"{Seq2Tensor_path}/*pt")]) # 加载已经保存好的Emdeding数据
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def seq2tensor(self, seq):
        # if seq in self.ProteinPairsDataTensors:
        #     return torch.load(f"{self.Seq2Tensor_path}/{seq}.pt")
        # else:
        #     seq_tensor = ESMC_embedding(seq)
        #     try:
        #         torch.save(seq_tensor, f"{self.Seq2Tensor_path}/{seq}.pt")
        #     except:
        #         print(f"Error: {seq} cannot be saved.")
        #     self.ProteinPairsDataTensors.add(seq)
        #     return seq_tensor
        return ESMC_embedding(seq).to(device)
    
    def __len__(self):
        """返回数据集的大小"""
        return len(self.data)
        
    def __getitem__(self, idx):
        """
        获取特定索引的样本。

        参数:
            idx (int): 索引值

        返回:
            dict: 包含序列对及其标签的字典
        """
        sequence1 = self.data.iloc[idx]['sequence1']  # 根据列名获取序列1
        sequence2 = self.data.iloc[idx]['sequence2']  # 根据列名获取序列2
        label = self.data.iloc[idx]['label']  # 根据列名获取标签

        # 将序列转成张量，使用seq2tensor
        sequence1_tensor = self.seq2tensor(sequence1)
        sequence2_tensor = self.seq2tensor(sequence2)

        if self.transform is not None:
            sequence1_tensor = self.transform(sequence1_tensor, self.NormalizedSequenceLength)
            sequence2_tensor = self.transform(sequence2_tensor, self.NormalizedSequenceLength)        

        return {"sequence1": sequence1_tensor, "sequence2": sequence2_tensor, "label": label}

def transform(sequence_tensor, NormalizedSequenceLength = 50):
    """
    判断序列张量是否合规，否则进行变换
    """
    # 输入的形状为[ESM_size, 1, sequence_length, embedding_dim]
    # 输出的形状为[ESM_size, 1, NormalizedSequenceLength, embedding_dim]
    if sequence_tensor.shape[2] > NormalizedSequenceLength:
        sequence_tensor = sequence_tensor[:, :, :NormalizedSequenceLength, :]
    elif sequence_tensor.shape[2] < NormalizedSequenceLength:
        padding_tensor = torch.zeros(sequence_tensor.shape[0], sequence_tensor.shape[1], NormalizedSequenceLength - sequence_tensor.shape[2], sequence_tensor.shape[3])
        sequence_tensor = torch.cat((sequence_tensor, padding_tensor), dim=2)
    return sequence_tensor



# 加载数据
def load_data(ProteinPairsData_csv_path, Seq2Tensor_path, NormalizedSequenceLength =50, batch_size=32, val_size=0.1, test_size=0.1):
    """
    加载数据并划分为训练集、验证集和测试集。

    参数:
        ProteinPairsData_csv_path (str): CSV 文件路径。
        batch_size (int): 每个批次的大小。
        val_size (float): 验证集的比例（0-1之间的浮点数）。
        test_size (float): 测试集的比例（0-1之间的浮点数）。

    返回:
        tuple: 训练集, 验证集和测试集的 DataLoader。
    """
    # 读取 CSV 文件
    data = pd.read_csv(ProteinPairsData_csv_path)

    # 划分训练集和临时集（临时集中将会包含验证集和测试集）
    train_data, temp_data = train_test_split(data, test_size=(val_size + test_size), random_state=42)

    # 计算临时集中验证集和测试集的比例
    temp_val_size = val_size / (val_size + test_size)
    
    # 划分验证集和测试集
    val_data, test_data = train_test_split(temp_data, test_size=temp_val_size, random_state=42)

    # 创建数据集实例
    train_dataset = ProteinPairsDataset(data = train_data,
                                        NormalizedSequenceLength = NormalizedSequenceLength,
                                        Seq2Tensor_path = Seq2Tensor_path,
                                        transform = transform)
    val_dataset = ProteinPairsDataset(data = val_data,
                                      NormalizedSequenceLength = NormalizedSequenceLength,
                                      Seq2Tensor_path = Seq2Tensor_path,
                                      transform = transform)
    test_dataset = ProteinPairsDataset(data = test_data,
                                       NormalizedSequenceLength = NormalizedSequenceLength,
                                       Seq2Tensor_path = Seq2Tensor_path,
                                       transform = transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



def save_model(model, optimizer, epoch, file_path):
    """
    保存模型及其状态

    参数:
        model: 要保存的模型
        optimizer: 优化器
        epoch: 当前训练的epoch
        file_path: 保存的文件路径
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)

def load_model(model, optimizer, file_path):
    """
    加载模型及其状态

    参数:
        model: 要加载的模型
        optimizer: 优化器
        file_path: 加载的文件路径
    """
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']  # 返回加载的epoch