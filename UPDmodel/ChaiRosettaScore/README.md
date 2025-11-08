# welcome to ChaiRosettaScore

## Introduction

- [ChaiRosettaScore.py](ChaiRosettaScore.py) 程序主文件，用于计算ChaiRosettaScore。
- [config.json](config.json) 配置文件，用于指定ChaiRosettaScore所需的输入文件路径和相关参数
- [setting_options.ini](setting_options.ini) rosetta相关参数配置文件，用于指定rosetta相关参数
- [ChaiRosettaScore.log](ChaiRosettaScore.log) 程序运行日志文件，用于记录程序运行过程中的信息
- [OUTPUT](OUTPUT) 输出保存文件夹，用于保存ChaiRosettaScore计算结果，即每次运行后从output中复制文件OUTPUT到当前位置。
- [out_path.sh](out_path.sh) 用于转移输出文件夹和log文件到OUTPUT文件夹。
- [README.md](README.md) 程序说明文件，本文件。


## To run ChaiRosettaScore.py:

- choose node17 as the working node
- `cd /home/users/hcdai/AI-peptide/ChaiRosettaScore`
- terminal run `/home/users/hcdai/miniconda3/envs/Chai-1/bin/python /home/users/hcdai/AI-peptide/ChaiRosettaScore/ChaiRosettaScore.py`
- /home/users/hcdai/miniconda3/envs/Chai-1/bin/python /home/users/hcdai/AI-peptide/ChaiRosettaScore_par/ChaiRosettaScore_par.py --config_num 3
- run `sh out_path.sh` to transfer output files to OUTPUT folder.

## 运行记录
- 2024/11/05 测试文献中的数据集

```
@article{wuPancoronavirusPeptideInhibitor2023,
  title = {A Pan-Coronavirus Peptide Inhibitor Prevents {{SARS-CoV-2}} Infection in Mice by Intranasal Delivery},
  author = {Wu, Lili and Zheng, Anqi and Tang, Yangming and Chai, Yan and Chen, Jiantao and Cheng, Lin and Hu, Yu and Qu, Jing and Lei, Wenwen and Liu, William Jun and Wu, Guizhen and Zeng, Shaogui and Yang, Hang and Wang, Qihui and Gao, George Fu},
  year = {2023},
  journal = {Science China Life Sciences},
  pages = {2201--2213},
  issn = {1674-7305, 1869-1889},
  doi = {10.1007/s11427-023-2410-5},
}
```