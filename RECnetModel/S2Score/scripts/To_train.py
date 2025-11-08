import os
os.chdir('/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0')
import sys
sys.path.append('./S2S_1.0')
import argparse
import shutil
from S2S_utils import *
from S2S_model import *
from S2S_train import *
from sklearn.model_selection import train_test_split

# /home/users/hcdai/miniconda3/envs/ESMC/bin/accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 /home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/trying/20250217_loss_too_big/dataset/generated_EK_withpath_train_try.csv
def main(): 
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    
    database_path = '/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/dataset/standard_csv_withpath/generated_EK_withpath_train.csv' # 本次训练数据集的路径
    checkpoint_path = None # 模型检查点保存路径，没有的话写None
    output_path = '/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/checkpoint/20250217_generated_EK1_times1' # 模型检查点和权重保存路径
    logging_dir = '/home/users/hcdai/AI-peptide/Seq2Score/Seq2Score_1.0/log/20250214_TF_ori' # 日志保存路径
    label_cut = [62]
    train_target = ['RosettaScore']
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
    
    model = load_model_from_checkpoint(checkpoint_path = checkpoint_path,
                                       device = device)
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=20,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        fp16=True,  # 启用混合精度训练
        metric_for_best_model='eval_runtime'
    )
    
    s2s_loss = S2S_Loss(label_cut, train_target)
    # all_parameters = list(model.parameters()) + [s2s_loss.log_var_rosetta]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    database = pd.read_csv(database_path)
    train_df, valid_df = train_test_split(database, test_size=0.1, random_state=42) # 一次训练只进行一次划分，避免数据泄露
    for epoch in range(training_args.num_train_epochs):
        # 重置索引，保证索引连续，避免出现KeyError
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)

        train_dataset = PeptideDataset(df = train_df,
                                   score_columns = score_columns,
                                   merge_hidden = 'ESMC_Embedding_Path',
                                   ligand_hidden = 'ESMC_Ligand_Path',
                                   receptor_hidden = 'ESMC_Receptor_Path',
                                   )

        valid_dataset = PeptideDataset(df = valid_df,
                                   score_columns = score_columns,
                                   merge_hidden = 'ESMC_Embedding_Path',
                                   ligand_hidden = 'ESMC_Ligand_Path',
                                   receptor_hidden = 'ESMC_Receptor_Path',
                                   )
        
        optimizer.add_param_group({'params': s2s_loss.log_var_rosetta})
        # 初始化自定义数据收集器
        
        data_collator = CustomDataCollator()
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            label_cut=label_cut,
            train_target=train_target,
            accelerator=accelerator,
            optimizer = optimizer
        )

        # 数据加载器
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size,
                                                   shuffle=True, pin_memory=True)
        eval_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=training_args.per_device_eval_batch_size,
                                                  shuffle=False, pin_memory=True)

        # 使用加速器包装模型和数据加载器
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)
        
        # 开始训练
        trainer.train()

        # 保存模型
        # model_save_path = f'/home/users/hcdai/AI-peptide/Seq2Score/Seq2Rosscore/ESMC-MLP/ESMC-attn/checkpoint_of_attn_20250125/attn_baseline/attn_baseline_checkpoint_epoch_{epoch}.pth'
        # torch.save(model.state_dict(), model_save_path)

        # 评估模型
        eval_results = trainer.evaluate()
        print(f"Epoch {epoch + 1} Evaluation results: {eval_results}")

if __name__ == '__main__':
    main()
    
    
    