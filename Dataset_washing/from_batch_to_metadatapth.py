import os
import torch
import glob
from tqdm import tqdm
import time

def create_metadata_with_light_validation():
    """创建元数据文件，带进度条且只验证前10个文件"""
    
    # 配置路径
    batch_dir = "/inspire/hdd/project/embodied-multimodality/public/hcd/Moss/Diffupep_trasn/Diffupep/DiffuSeq-diffuseq-v2/datasets/uniref50-all-mask/valid_batches"
    output_file = "/inspire/hdd/project/embodied-multimodality/public/hcd/Moss/Diffupep_trasn/Diffupep/DiffuSeq-diffuseq-v2/datasets/uniref50-all-mask/valid.pth"
    
    print("🚀 开始创建训练数据元文件...")
    print(f"📁 批次文件夹: {batch_dir}")
    print(f"💾 输出文件: {output_file}")
    print("-" * 60)
    
    # 步骤1: 扫描批次文件（带进度条）
    print("📡 扫描批次文件中...")
    time.sleep(0.5)
    
    # 获取所有batch文件
    batch_files = glob.glob(os.path.join(batch_dir, "batch_*.pth"))
    if not batch_files:
        print("❌ 没有找到batch_*.pth文件")
        return
    
    # 按数字排序
    batch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"✅ 找到 {len(batch_files)} 个批次文件")
    print(f"📄 文件范围: {os.path.basename(batch_files[0])} ~ {os.path.basename(batch_files[-1])}")
    
    # 检查文件数量是否符合预期
    expected_count = 140000
    if len(batch_files) != expected_count:
        print(f"⚠️  警告: 找到 {len(batch_files)} 个文件，期望 {expected_count} 个")
    print("-" * 60)
    
    # 步骤2: 快速验证前10个文件（带进度条）
    print("🔍 快速验证前10个文件可读性...")
    valid_files = batch_files  # 假设所有文件都有效
    corrupted_files = []
    
    # 只验证前10个文件
    files_to_check = batch_files[:10]
    
    for i, file_path in enumerate(tqdm(files_to_check, desc="验证文件", unit="file")):
        try:
            # 快速验证文件可读性
            with open(file_path, 'rb') as f:
                data = torch.load(f, map_location='cpu')
            print(f"   ✅ {os.path.basename(file_path)}: 验证通过")
        except Exception as e:
            corrupted_files.append((os.path.basename(file_path), str(e)))
            print(f"   ❌ {os.path.basename(file_path)}: 验证失败 - {e}")
    
    if corrupted_files:
        print(f"⚠️  发现 {len(corrupted_files)} 个损坏文件，建议检查数据完整性")
    else:
        print("✅ 前10个文件验证全部通过")
    print("-" * 60)
    
    # 步骤3: 创建元数据结构（带进度条）
    print("📊 创建元数据结构...")
    
    # 创建完整的批次路径列表
    all_batch_paths = []
    for file_path in tqdm(batch_files, desc="生成路径列表", unit="file"):
        all_batch_paths.append(file_path)
    
    metadata = {
        "batch_paths": all_batch_paths,
        "total_batches": len(all_batch_paths),
        "source_folder": batch_dir,
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation_info": {
            "files_checked": len(files_to_check),
            "files_passed": len(files_to_check) - len(corrupted_files),
            "files_failed": len(corrupted_files),
            "first_file": os.path.basename(all_batch_paths[0]),
            "last_file": os.path.basename(all_batch_paths[-1])
        }
    }
    
    # 步骤4: 保存元数据文件
    print("💾 保存元数据文件...")
    try:
        torch.save(metadata, output_file)
        print(f"✅ 元数据文件保存成功: {output_file}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return
    
    # 步骤5: 验证生成的文件
    print("🔍 验证生成的文件...")
    try:
        # 重新加载验证
        loaded_metadata = torch.load(output_file)
        print(f"✅ 文件验证成功")
        print(f"📊 包含批次路径: {len(loaded_metadata['batch_paths'])} 个")
        print(f"📄 第一个批次: {os.path.basename(loaded_metadata['batch_paths'][0])}")
        print(f"📄 最后一个批次: {os.path.basename(loaded_metadata['batch_paths'][-1])}")
        
        # 验证数据结构
        if 'batch_paths' in loaded_metadata and isinstance(loaded_metadata['batch_paths'], list):
            print("✅ 数据结构正确")
        else:
            print("❌ 数据结构不正确")
            
    except Exception as e:
        print(f"❌ 文件验证失败: {e}")
        return
    
    print("-" * 60)
    print("🎉 元数据文件创建完成！")
    print(f"📁 输出位置: {output_file}")
    print(f"📊 总批次数量: {len(all_batch_paths)}")
    
    return metadata

# 执行创建
if __name__ == "__main__":
    create_metadata_with_light_validation()