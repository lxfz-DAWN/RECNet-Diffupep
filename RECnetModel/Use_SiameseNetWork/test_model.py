# import torch

# def test_model(model, loss_criterion, test_loader):
#     """
#     测试模型并计算测试损失和准确率。

#     参数:
#         model: 要测试的模型
#         test_loader: 用于测试的数据加载器
#         criterion: 损失函数

#     返回:
#         float: 测试损失
#         float: 测试准确率
#     """
#     model.eval()  # 设置模型为评估模式
#     criterion = loss_criterion  # 损失函数
#     test_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():  # 评估时不需要反向传播
#         for batch in test_loader:
#             sequence1 = batch["sequence1"]
#             sequence2 = batch["sequence2"]
#             labels = batch["label"]

#             # 前向传播
#             output1, output2 = model(sequence1, sequence2)
            
#             # 计算损失
#             loss = criterion(output1, output2, labels)
#             test_loss += loss.item()

#             # 计算准确率，这里假设通过某种方式获取每个样本的预测结果
#             # 需要定义一个方法计算预测结果，比如使用阈值来决定正负样本
#             predicted = (torch.nn.functional.cosine_similarity(output1, output2) > 0.5).float()  # 示例
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     avg_test_loss = test_loss / len(test_loader)
#     accuracy = correct / total * 100  # 转换为百分比

#     print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%')
#     return avg_test_loss, accuracy


import torchsummary

