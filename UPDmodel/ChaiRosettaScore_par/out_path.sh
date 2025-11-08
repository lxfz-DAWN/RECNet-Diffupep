# 转移输出文件至OUTPUT文件夹

#!/bin/bash

cd /home/users/hcdai/AI-peptide/ChaiRosettaScore

# 在OUTPUT文件夹下创建以时间戳命名的文件夹
timestamp=$(date +%Y%m%d_%H%M%S)

mike_dir="/home/users/hcdai/AI-peptide/ChaiRosettaScore/OUTPUT/$timestamp"
mkdir $mike_dir

# 移动output文件夹至目标目录
mv /home/users/hcdai/AI-peptide/ChaiRosettaScore/output $mike_dir

# 移动log文件夹中的所有log文件至目标目录
mv /home/users/hcdai/AI-peptide/ChaiRosettaScore/logs/*.log $mike_dir
# mv /home/users/hcdai/AI-peptide/ChaiRosettaScore/ChaiRosettaScore.log $mike_dir
# mv /home/users/hcdai/AI-peptide/ChaiRosettaScore/ROSETTA_CRASH.log $mike_dir

# 移动输出的csv文件至目标目录
mv /home/users/hcdai/AI-peptide/ChaiRosettaScore/output_result.csv $mike_dir