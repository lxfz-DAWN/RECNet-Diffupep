# Set the workingdir to the 'Seq2Score_1.0'
'''
Usage: python To_save_ESMC_tensor route \
           --input_csv input_csv_route \
           --output_csv output_csv_route \
           --pt_dir ESMC_embedding_dir\

'''
import os
os.chdir('/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0')
import sys
sys.path.append('./S2S_1.0')
import argparse
import csv
import shutil
from S2S_utils import ESMC_func, ESMC_saving

def main():
    parser = argparse.ArgumentParser(description='Dealing with seqs, turn them into ESMC tensor')
    
    parser.add_argument('--input_csv', required=True, help='standard_csv_ori/')
    parser.add_argument('--output_csv', required=True, help='standard_ csv_withpath/')
    parser.add_argument('--pt_dir', required=True, help='ESMC_embedding/inputname,merge or split?')
    parser.add_argument('--temp_dir', default = '/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/dataset/ESMC_embedding/temp', help='ESMC_embedding/inputname,merge or split?')
    parser.add_argument('--ligand_col_name', type=str, default='ligand_sequence', help='ligand_col_name')
    parser.add_argument('--receptor_col_name', type=str, default='receptor_sequence', help='receptor_col_name')
    
    args = parser.parse_args()
    
    try:
    # 尝试创建文件夹
        os.mkdir(args.temp_dir)
        print(f"文件夹 {args.temp_dir} 创建成功。")
    except FileExistsError:
        print(f"文件夹 {args.temp_dir} 已存在，无需创建。")
        
    temp_path = args.temp_dir + '/temp.csv'
    open(temp_path, 'w')
    ESMC_saving(input_csv = args.input_csv,
                output_csv = temp_path,
                pt_dir = args.pt_dir,
                ligand_col_name = args.ligand_col_name,
                receptor_col_name = args.receptor_col_name,
                mode = 'merge'
    )
    ESMC_saving(input_csv = temp_path,
                output_csv = args.output_csv,
                pt_dir = args.pt_dir,
                ligand_col_name = args.ligand_col_name,
                receptor_col_name = args.receptor_col_name,
                mode = 'split'
    )
    
    # 检查文件夹是否存在
    if os.path.exists(args.temp_dir):
        try:
            # 删除文件夹及其所有内容
            shutil.rmtree(args.temp_dir)
            print(f"文件夹 {args.temp_dir} 已成功删除。")
        except Exception as e:
            print(f"删除文件夹 {args.temp_dir} 时出现错误: {e}")
    else:
        print(f"指定的文件夹 {args.temp_dir} 不存在。")
        
if __name__ == '__main__':
    main()

# test
# python /home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/scripts/To_save_ESMC_tensor.py --input_csv /home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/dataset/standard_csv_ori/test.csv --output_csv /home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/dataset/standard_csv_withpath/test.csv --pt_dir /home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/dataset/ESMC_embedding/test
# python /home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/scripts/To_save_ESMC_tensor.py --input_csv /home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/dataset/standard_csv_ori/generated_EK.csv --output_csv /home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/dataset/standard_csv_withpath/generated_EK_withpath.csv --pt_dir /home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/dataset/ESMC_embedding/generated_EK_withpath
# python /home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/scripts/To_save_ESMC_tensor.py --input_csv /home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/dataset/standard_csv_ori/TF_ori.csv --output_csv /home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/dataset/standard_csv_withpath/TF_ori.csv --pt_dir /home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/dataset/ESMC_embedding/TF_ori_withpath

