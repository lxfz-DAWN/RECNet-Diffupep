# 环境配置
运行各个.py文件时，import缺啥，直接pip install啥即可

# 微调ESM2模型
[finetuning_test.py](finetuning_test.py) 实现了一个序列输入，输出9个打分
[finetuning_test_plus.py](finetuning_test_plus.py) 实现了两个序列输入（其中一个原序列、另一个目标对接序列），输出1个打分
当训练一个新模型时，设置checkpoint_path = None，取消“开始训练”与“保存模型”的注释，注释掉“示例预测”（注意如果第一次使用某个模型，在运行时会自动下载）
从一个已经训练了一部分的模型载入并继续训练时，需设置checkpoint_path模型检查点路径，取消“开始训练”与“保存模型”的注释，注释掉“示例预测”
不训练模型，只是验证模型时，需设置checkpoint_path模型检查点路径，注释掉“开始训练”与“保存模型”，可以用“评估模型”来验证验证集，也可以用“示例预测”来通过序列得到分数。
注意，测试时使用的是facebook/esm2_t6_8M_UR50D，是一个很小的、表现很差的模型，只是用于测试。实际应用中将使用esm2_t36_3B_UR50D()或 esm2_t48_15B_UR50D()这样的大模型

[random_generate_seq_num.py](random_generate_seq_num.py)用于测试时生成一些随机数据到csv

[protein_language_modeling.py)](protein_language_modeling.py)与[protein_language_modeling.ipynb](protein_language_modeling.ipynb)是官方给的esm微调教程，可以阅读参考

[esm-main](esm-main)是esm的github库

[esm_test_predict.py](esm_test_predict.py)和[esm_test.py](esm_test.py)是使用esm原模型的一个测试，可以用于学习

# 强化学习辅助变异测试
[mutation_test_plus.py](mutation_test_plus.py) 是一个简单的强化学习辅助变异测试，其中策略网络使用了esm2

