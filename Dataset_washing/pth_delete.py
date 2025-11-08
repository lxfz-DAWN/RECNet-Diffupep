import os
import glob
from tqdm import tqdm

def delete_with_progress():
    """带进度条的删除操作"""
    
    batch_dir = "/inspire/hdd/project/embodied-multimodality/public/hcd/Moss/Diffupep_trasn/Diffupep/DiffuSeq-diffuseq-v2/datasets/uniref50-all-mask/train_batches"
    
    # 获取文件列表
    batch_files = glob.glob(os.path.join(batch_dir, "batch_*.pth"))
    batch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # 筛选需要删除的文件
    files_to_delete = []
    for file_path in batch_files:
        file_number = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
        if file_number >= 140000:
            files_to_delete.append(file_path)
    
    print(f"📊 总文件数: {len(batch_files)}")
    print(f"🗑️  待删除文件: {len(files_to_delete)}")
    print(f"📁 保留文件: {len(batch_files) - len(files_to_delete)}")
    
    if not files_to_delete:
        print("✅ 没有需要删除的文件")
        return
    
    # 显示文件范围
    first_file = os.path.basename(files_to_delete[0])
    last_file = os.path.basename(files_to_delete[-1])
    print(f"📄 删除范围: {first_file} ~ {last_file}")
    
    # 确认
    confirm = input("🚨 确认删除吗？此操作不可恢复！(输入 'DELETE' 确认): ")
    if confirm != 'DELETE':
        print("❌ 操作取消")
        return
    
    # 执行删除（带进度条）
    deleted_count = 0
    for file_path in tqdm(files_to_delete, desc="删除进度"):
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"\n❌ 删除失败: {os.path.basename(file_path)}")
    
    print(f"✅ 删除完成！删除了 {deleted_count} 个文件")
    
    # 验证
    remaining = glob.glob(os.path.join(batch_dir, "batch_*.pth"))
    max_remaining = max([int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in remaining]) if remaining else -1
    print(f"📊 剩余文件: {len(remaining)} 个")
    print(f"🔢 最大文件编号: {max_remaining}")

# 执行
delete_with_progress()