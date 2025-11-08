## welcome to RunRosetta

## Abstract

RunRosetta 是一个用于将收集的蛋白互作数据重新进行RosettaDocking，并重新打分的工具。

## Introduction

- [RunRosetta.py](RunRosetta.py) 程序主文件，用于将收集的蛋白互作数据重新进行RosettaDocking，并重新打分的工具。
- [config.json](config/config.json) 配置文件，用于指定RunRosetta.py运行所需的输入文件路径和相关参数。
- [input]: 输入数据来源，储存数据库pdb文件
- [output]: 输出数据，储存RosettaDocking结果文件
- [error_log.md](error_log.md) : 记录运行过程中报错的pdb文件以及操作
- [output_result]: 储存各pdb文件的RosettaDocking结果文件
- 
## To run RunRosetta.py:

- choose node17 as the working node：`ssh node17`
- `cd /home/users/hcdai/AI-peptide/RunRosetta`
- terminal run `/home/users/hcdai/miniconda3/envs/Chai-1/bin/python /home/users/hcdai/AI-peptide/RunRosetta/RunRosetta.py`
- password: `QH23e9!P`
