import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from Bio.SeqUtils import ProtParam
from Bio import SeqIO
from transformers import AutoModel, AutoTokenizer
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
import warnings
import re
import random
import json
from tqdm import tqdm
warnings.filterwarnings('ignore')
class ProteinSequenceCleaner:
    """蛋白质序列清洗器"""
    
    @staticmethod
    def clean_sequence(sequence):
        """
        清洗蛋白质序列，去除无效字符和空格
        
        Args:
            sequence (str): 原始蛋白质序列
            
        Returns:
            str: 清洗后的序列
        """
        if not sequence or not isinstance(sequence, str):
            return ""
        
        # 转换为大写
        sequence = sequence.upper()
        
        # 移除所有空格、制表符、换行符等空白字符
        sequence = re.sub(r'\s+', '', sequence)
        
        # 移除数字
        sequence = re.sub(r'\d+', '', sequence)
        
        # 定义标准氨基酸字符（20种标准氨基酸）
        standard_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        
        # 处理非标准氨基酸字符
        cleaned_sequence = []
        for aa in sequence:
            if aa in standard_amino_acids:
                cleaned_sequence.append(aa)
            else:
                # 处理常见非标准氨基酸表示
                if aa == 'B':  # 天冬酰胺或天冬氨酸（不确定）
                    # 保守处理：替换为天冬酰胺(N)
                    cleaned_sequence.append('N')
                elif aa == 'Z':  # 谷氨酰胺或谷氨酸（不确定）
                    # 保守处理：替换为谷氨酰胺(Q)
                    cleaned_sequence.append('Q')
                elif aa == 'J':  # 亮氨酸或异亮氨酸（不确定）
                    # 保守处理：替换为亮氨酸(L)
                    cleaned_sequence.append('L')
                elif aa == 'O':  # 吡咯赖氨酸
                    # 用赖氨酸(K)替代
                    cleaned_sequence.append('K')
                elif aa == 'U':  # 硒代半胱氨酸
                    # 用半胱氨酸(C)替代
                    cleaned_sequence.append('C')
                elif aa == 'X':  # 任意氨基酸
                    # 用丙氨酸(A)替代（中性选择）
                    cleaned_sequence.append('A')
                elif aa in ['*', '.', '-']:  # 终止符或间隔符
                    # 跳过这些字符
                    continue
                else:
                    # 其他未知字符，用丙氨酸(A)替代
                    cleaned_sequence.append('A')
        
        cleaned_sequence = ''.join(cleaned_sequence)
        
        return cleaned_sequence
    
    @staticmethod
    def validate_sequence(sequence, min_length=10, max_length=10000):
        """
        验证序列是否有效
        
        Args:
            sequence (str): 蛋白质序列
            min_length (int): 最小长度
            max_length (int): 最大长度
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not sequence:
            return False, "序列为空"
        
        if len(sequence) < min_length:
            return False, f"序列长度({len(sequence)})小于最小要求({min_length})"
        
        if len(sequence) > max_length:
            return False, f"序列长度({len(sequence)})超过最大限制({max_length})"
        
        # 检查是否包含有效氨基酸
        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        sequence_chars = set(sequence)
        
        if not sequence_chars.issubset(valid_amino_acids):
            invalid_chars = sequence_chars - valid_amino_acids
            return False, f"序列包含无效字符: {invalid_chars}"
        
        return True, "序列有效"

class StandardSequenceGenerator:
    """标准序列生成器"""
    
    @staticmethod
    def truncate_to_standard_lengths(sequence, target_lengths=[55, 56, 57, 58, 59, 60]):
        """
        将长序列截短为多个标准长度的序列
        
        Args:
            sequence (str): 原始氨基酸序列
            target_lengths (list): 目标长度列表
            
        Returns:
            list: 标准序列列表，每个元素为(sequence_id, sequence)
        """
        standard_sequences = []
        seq_len = len(sequence)
        
        # 如果序列太短，直接返回空列表
        if seq_len < min(target_lengths):
            return standard_sequences
        
        # 对每个目标长度，从序列中随机截取一段
        for target_len in target_lengths:
            if seq_len >= target_len:
                # 随机选择起始位置
                start_pos = random.randint(0, seq_len - target_len)
                truncated_seq = sequence[start_pos:start_pos + target_len]
                
                # 生成唯一的序列ID
                seq_id = f"std_seq_{start_pos}_{start_pos+target_len}_len{target_len}"
                
                standard_sequences.append((seq_id, truncated_seq))
        
        return standard_sequences
    
    @staticmethod
    def generate_standard_sequences_batch(sequences_dict, target_lengths=[55, 56, 57, 58, 59, 60]):
        """
        批量生成标准序列
        
        Args:
            sequences_dict (dict): 蛋白质ID到序列的映射
            target_lengths (list): 目标长度列表
            
        Returns:
            dict: 标准序列ID到序列的映射
        """
        standard_sequences = {}
        
        for protein_id, sequence in sequences_dict.items():
            std_seqs = StandardSequenceGenerator.truncate_to_standard_lengths(sequence, target_lengths)
            for seq_id, std_seq in std_seqs:
                standard_sequences[seq_id] = std_seq
        
        return standard_sequences

class ProteinPropertyCalculator:
    """蛋白质理化性质综合计算器"""
    
    def __init__(self, use_deep_learning=True):
        self.use_deep_learning = use_deep_learning
        self.cleaner = ProteinSequenceCleaner()
        self.setup_models()
        
    def setup_models(self):
        """初始化各种预测模型"""
        # ESM模型用于深度学习方法
        if self.use_deep_learning:
            try:
                self.esm_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
                self.esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            except:
                print("ESM模型加载失败，将使用传统方法")
                self.use_deep_learning = False
        
        # 传统机器学习模型
        self.expression_predictor = ExpressionPredictor()
        self.solubility_predictor = SolubilityPredictor()
    
    def preprocess_sequence(self, sequence):
        """
        预处理序列：清洗和验证
        
        Args:
            sequence (str): 原始序列
            
        Returns:
            tuple: (cleaned_sequence, is_valid, error_message)
        """
        # 清洗序列
        cleaned_sequence = self.cleaner.clean_sequence(sequence)
        
        # 验证序列
        is_valid, error_message = self.cleaner.validate_sequence(cleaned_sequence)
        
        return cleaned_sequence, is_valid, error_message
    
    def calculate_basic_properties(self, sequence):
        """计算基础物理化学性质"""
        analyzer = ProtParam.ProteinAnalysis(sequence)
        
        properties = {
            'sequence_length': len(sequence),
            'molecular_weight': analyzer.molecular_weight(),
            'isoelectric_point': analyzer.isoelectric_point(),
            'instability_index': analyzer.instability_index(),
            'aromaticity': analyzer.aromaticity(),
            'gravy': analyzer.gravy(),
            'extinction_coefficient_reduced': analyzer.molar_extinction_coefficient()[0],
            'extinction_coefficient_oxidized': analyzer.molar_extinction_coefficient()[1],
        }
        
        # 二级结构倾向
        sec_struct = analyzer.secondary_structure_fraction()
        properties.update({
            'helix_fraction': sec_struct[0],
            'turn_fraction': sec_struct[1],
            'sheet_fraction': sec_struct[2]
        })
        
        return properties
    
    def calculate_electronegativity_polarity(self, sequence):
        """计算电负性和极性"""
        # 氨基酸电负性值
        aa_electronegativity = {
            'A': 2.3, 'R': 3.0, 'N': 3.0, 'D': 3.5, 'C': 2.5,
            'Q': 3.0, 'E': 3.4, 'G': 2.3, 'H': 3.2, 'I': 2.3,
            'L': 2.3, 'K': 2.8, 'M': 2.5, 'F': 2.7, 'P': 2.3,
            'S': 2.8, 'T': 2.8, 'W': 2.9, 'Y': 2.9, 'V': 2.3
        }
        
        # 极性氨基酸
        polar_amino_acids = {'R', 'N', 'D', 'Q', 'E', 'H', 'K', 'S', 'T', 'Y', 'C'}
        # 带电氨基酸
        charged_amino_acids = {'R', 'H', 'K', 'D', 'E'}
        
        total_electronegativity = 0
        polar_count = 0
        charged_count = 0
        
        for aa in sequence:
            total_electronegativity += aa_electronegativity.get(aa, 2.5)
            if aa in polar_amino_acids:
                polar_count += 1
            if aa in charged_amino_acids:
                charged_count += 1
        
        return {
            'electronegativity': total_electronegativity / len(sequence),
            'polarity_ratio': polar_count / len(sequence),
            'charged_aa_ratio': charged_count / len(sequence)
        }
    
    def predict_stability(self, sequence):
        """预测稳定性相关指标"""
        analyzer = ProtParam.ProteinAnalysis(sequence)
        
        # 热稳定性预测 (简化版)
        instability_index = analyzer.instability_index()
        gravy = analyzer.gravy()
        
        # 基于经验规则的稳定性评分
        stability_score = 0
        if instability_index < 40:
            stability_score += 0.6
        elif instability_index < 50:
            stability_score += 0.3
        else:
            stability_score += 0.1
            
        if gravy < 0:
            stability_score += 0.4
        else:
            stability_score += 0.2
            
        return {
            'thermal_stability_score': min(1.0, stability_score),
            'thermodynamic_stability_score': 1 - (instability_index / 100),
            'kinetic_stability_score': max(0, 1 - (instability_index - 30) / 70)
        }
    
    def predict_activity_function(self, sequence):
        """预测活性/功能相关指标"""
        # 简化版的活性预测
        charged_ratio = sum(1 for aa in sequence if aa in 'DEKRH') / len(sequence)
        cysteine_content = sequence.count('C') / len(sequence)
        
        # 催化效率评分
        catalytic_efficiency_score = 0.3 + 0.4 * charged_ratio + 0.3 * (1 - cysteine_content)
        
        # 结合亲和力评分
        binding_affinity_score = 0.4 + 0.3 * charged_ratio + 0.3 * (len(sequence) / 500)
        
        # 特异性评分
        specificity_score = 0.5 + 0.2 * charged_ratio + 0.3 * (1 - sequence.count('C') / 20)
        
        return {
            'catalytic_efficiency_score': min(1.0, catalytic_efficiency_score),
            'binding_affinity_score': min(1.0, binding_affinity_score),
            'specificity_score': min(1.0, specificity_score)
        }
    
    def predict_expressibility(self, sequence):
        """预测可表达性"""
        # 表达量预测
        expression_score = self.expression_predictor.predict_expression(sequence)
        
        # 可溶性预测
        solubility_score = self.solubility_predictor.predict_solubility(sequence)
        
        return {
            'expression_score': expression_score,
            'solubility_score': solubility_score
        }
    
    def predict_developability(self, sequence):
        """预测可开发性"""
        analyzer = ProtParam.ProteinAnalysis(sequence)
        
        # 聚集倾向
        gravy = analyzer.gravy()
        instability = analyzer.instability_index()
        cysteine_content = sequence.count('C') / len(sequence)
        
        aggregation_score = 0.4 * max(0, gravy) + 0.3 * (instability / 100) + 0.3 * cysteine_content
        
        # 免疫原性 (简化版)
        aromaticity = analyzer.aromaticity()
        charged_ratio = sum(1 for aa in sequence if aa in 'DEKRH') / len(sequence)
        immunogenicity_score = 0.3 * aromaticity + 0.4 * charged_ratio + 0.3 * (len(sequence) / 300)
        
        return {
            'aggregation_propensity': min(1.0, aggregation_score),
            'immunogenicity_score': min(1.0, immunogenicity_score)
        }
    
    def predict_structural_properties(self, sequence):
        """预测结构特性"""
        analyzer = ProtParam.ProteinAnalysis(sequence)
        
        # 结构刚性/柔性
        proline_content = sequence.count('P') / len(sequence)
        glycine_content = sequence.count('G') / len(sequence)
        
        flexibility_score = 0.6 * glycine_content + 0.4 * (1 - proline_content)
        
        return {
            'structural_rigidity_score': 1 - flexibility_score,
            'flexibility_score': flexibility_score
        }
    
    def calculate_sequence_evolution_properties(self, sequence):
        """计算序列与进化特性"""
        # 天然度 (使用氨基酸组成复杂性)
        aa_composition = {}
        for aa in sequence:
            aa_composition[aa] = aa_composition.get(aa, 0) + 1
        
        # 计算序列复杂性 (香农熵)
        entropy = 0
        for count in aa_composition.values():
            p = count / len(sequence)
            entropy -= p * np.log2(p) if p > 0 else 0
        
        max_entropy = np.log2(min(20, len(sequence)))
        naturalness_score = entropy / max_entropy if max_entropy > 0 else 0
        
        # 进化保守性 (简化版)
        conserved_aas = 'GAVLIMFWPSTCYNQDEKRH'
        conservation_score = sum(1 for aa in sequence if aa in conserved_aas) / len(sequence)
        
        return {
            'naturalness_score': naturalness_score,
            'evolutionary_conservation': conservation_score
        }
    
    def predict_programmability(self, sequence):
        """预测可编程性与可控性"""
        # 对诱导物的响应 (简化版)
        charged_ratio = sum(1 for aa in sequence if aa in 'DEKRH') / len(sequence)
        small_aa_ratio = sum(1 for aa in sequence if aa in 'GAS') / len(sequence)
        
        inducibility_score = 0.5 * charged_ratio + 0.5 * small_aa_ratio
        
        # 自组装能力
        hydrophobic_ratio = sum(1 for aa in sequence if aa in 'AVILMFYW') / len(sequence)
        polar_ratio = sum(1 for aa in sequence if aa in 'NQSTY') / len(sequence)
        
        self_assembly_score = 0.6 * hydrophobic_ratio + 0.4 * polar_ratio
        
        return {
            'inducibility_score': min(1.0, inducibility_score),
            'self_assembly_score': min(1.0, self_assembly_score)
        }
    
    def calculate_composite_score(self, properties):
        """计算综合得分"""
        weights = {
            # 稳定性相关
            'thermal_stability_score': 0.08,
            'thermodynamic_stability_score': 0.08,
            'kinetic_stability_score': 0.06,
            
            # 活性/功能相关
            'catalytic_efficiency_score': 0.10,
            'binding_affinity_score': 0.08,
            'specificity_score': 0.06,
            
            # 可表达性
            'expression_score': 0.08,
            'solubility_score': 0.08,
            
            # 可开发性
            'aggregation_propensity': -0.06,  # 负权重
            'immunogenicity_score': -0.05,    # 负权重
            
            # 结构特性
            'structural_rigidity_score': 0.04,
            'flexibility_score': 0.04,
            
            # 序列与进化特性
            'naturalness_score': 0.07,
            'evolutionary_conservation': 0.06,
            
            # 可编程性
            'inducibility_score': 0.06,
            'self_assembly_score': 0.06
        }
        
        composite_score = 0
        for prop, weight in weights.items():
            if prop in properties:
                composite_score += properties[prop] * weight
        
        # 归一化到0-1范围
        composite_score = max(0, min(1, composite_score))
        return composite_score
    
    def analyze_standard_sequence(self, sequence_id, sequence):
        """综合分析单个标准序列"""
        # 预处理序列
        cleaned_sequence, is_valid, error_message = self.preprocess_sequence(sequence)
        
        if not is_valid:
            print(f"警告: 标准序列 {sequence_id} 无效 - {error_message}")
            return None
        
        # 收集所有性质
        properties = {
            'sequence_id': sequence_id, 
            'sequence': cleaned_sequence,
        }
        
        # 基础性质
        properties.update(self.calculate_basic_properties(cleaned_sequence))
        
        # 电负性和极性
        properties.update(self.calculate_electronegativity_polarity(cleaned_sequence))
        
        # 稳定性
        properties.update(self.predict_stability(cleaned_sequence))
        
        # 活性/功能
        properties.update(self.predict_activity_function(cleaned_sequence))
        
        # 可表达性
        properties.update(self.predict_expressibility(cleaned_sequence))
        
        # 可开发性
        properties.update(self.predict_developability(cleaned_sequence))
        
        # 结构特性
        properties.update(self.predict_structural_properties(cleaned_sequence))
        
        # 序列与进化特性
        properties.update(self.calculate_sequence_evolution_properties(cleaned_sequence))
        
        # 可编程性
        properties.update(self.predict_programmability(cleaned_sequence))
        
        # 综合得分
        properties['composite_score'] = self.calculate_composite_score(properties)
        
        return properties

class ExpressionPredictor:
    """表达量预测器"""
    
    def __init__(self):
        self.feature_names = ['length', 'molecular_weight', 'instability_index', 
                             'aromaticity', 'gravy', 'cysteine_content']
    
    def extract_features(self, sequence):
        """提取序列特征"""
        analyzer = ProtParam.ProteinAnalysis(sequence)
        
        features = {
            'length': len(sequence),
            'molecular_weight': analyzer.molecular_weight(),
            'instability_index': analyzer.instability_index(),
            'aromaticity': analyzer.aromaticity(),
            'gravy': analyzer.gravy(),
            'cysteine_content': sequence.count('C') / len(sequence)
        }
        return np.array([features[name] for name in self.feature_names])
    
    def predict_expression(self, sequence):
        """预测表达量"""
        features = self.extract_features(sequence)
        
        # 基于经验规则的简化预测
        score = 0.5
        score += 0.2 if features[2] < 40 else -0.2  # 不稳定指数
        score += 0.1 if features[4] < 0 else -0.1   # GRAVY
        score += 0.2 if features[0] < 400 else -0.1  # 长度
        
        return max(0, min(1, score))

class SolubilityPredictor:
    """可溶性预测器"""
    
    def __init__(self):
        pass
    
    def predict_solubility(self, sequence):
        """预测可溶性"""
        analyzer = ProtParam.ProteinAnalysis(sequence)
        
        gravy = analyzer.gravy()
        instability = analyzer.instability_index()
        charged_ratio = sum(1 for aa in sequence if aa in 'DEKRH') / len(sequence)
        
        # 基于经验规则的溶解度评分
        solubility_score = 0.5
        solubility_score += 0.3 if gravy < 0 else -0.2
        solubility_score += 0.1 if instability < 40 else -0.1
        solubility_score += 0.1 * charged_ratio
        
        return max(0, min(1, solubility_score))
    # 这里保留之前的所有类定义 (ProteinSequenceCleaner, StandardSequenceGenerator, 
# ProteinPropertyCalculator, ExpressionPredictor, SolubilityPredictor)

# 新增突变序列生成器类
class SequenceMutator:
    """序列突变生成器"""
    
    def __init__(self):
        # 20种标准氨基酸
        self.standard_amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    
    def random_mutation(self, sequence, mutation_rate=0.05):
        """
        对序列进行随机点突变
        
        Args:
            sequence (str): 原始序列
            mutation_rate (float): 突变率（0-1之间）
            
        Returns:
            tuple: (mutated_sequence, mutation_positions)
        """
        sequence = list(sequence)
        seq_len = len(sequence)
        num_mutations = max(1, int(seq_len * mutation_rate))  # 至少突变1个位点
        
        # 随机选择突变位点
        mutation_positions = random.sample(range(seq_len), num_mutations)
        mutation_info = []
        
        for pos in mutation_positions:
            original_aa = sequence[pos]
            # 从其他19种氨基酸中随机选择
            possible_mutations = [aa for aa in self.standard_amino_acids if aa != original_aa]
            mutated_aa = random.choice(possible_mutations)
            sequence[pos] = mutated_aa
            mutation_info.append({
                'position': pos,
                'original': original_aa,
                'mutated': mutated_aa
            })
        
        return ''.join(sequence), mutation_info
    
    def generate_mutated_sequence(self, original_sequence, mutation_rate=0.05):
        """
        生成突变序列并返回详细信息
        
        Args:
            original_sequence (str): 原始序列
            mutation_rate (float): 突变率
            
        Returns:
            dict: 包含突变序列和详细信息的字典
        """
        mutated_sequence, mutation_info = self.random_mutation(original_sequence, mutation_rate)
        
        return {
            'original_sequence': original_sequence,
            'mutated_sequence': mutated_sequence,
            'mutation_rate': mutation_rate,
            'num_mutations': len(mutation_info),
            'mutation_details': mutation_info
        }

# 修改主处理函数
def process_csv_with_mutations(csv_file_path, output_folder, mutation_rate=0.05):
    """
    处理CSV文件，对每条序列进行突变，比较突变前后的性质
    
    Args:
        csv_file_path (str): CSV文件路径
        output_folder (str): 输出文件夹
        mutation_rate (float): 突变率
        
    Returns:
        tuple: (comparison_results, json_data)
    """
    # 初始化计算器和突变器
    calculator = ProteinPropertyCalculator(use_deep_learning=False)
    mutator = SequenceMutator()
    
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    comparison_results = []
    json_data = []
    
    print(f"处理文件: {os.path.basename(csv_file_path)}")
    print(f"总序列数: {len(df)}")
    
    # 处理每条序列
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理序列突变"):
        original_sequence = row['sequence']
        original_score = row['composite_score']
        
        # 生成突变序列
        mutation_result = mutator.generate_mutated_sequence(original_sequence, mutation_rate)
        mutated_sequence = mutation_result['mutated_sequence']
        
        # 计算突变序列的性质
        try:
            mutated_properties = calculator.analyze_standard_sequence(
                f"mutated_{row['sequence_id']}", 
                mutated_sequence
            )
            
            if mutated_properties is not None:
                mutated_score = mutated_properties['composite_score']
                
                # 比较得分
                comparison = {
                    'original_sequence': original_sequence,
                    'mutated_sequence': mutated_sequence,
                    'original_score': original_score,
                    'mutated_score': mutated_score,
                    'score_difference': mutated_score - original_score,
                    'num_mutations': mutation_result['num_mutations'],
                    'mutation_details': mutation_result['mutation_details'],
                    'sequence_id': row['sequence_id']
                }
                
                comparison_results.append(comparison)
                
                # 准备JSON数据
                if original_score < mutated_score:
                    json_entry = {
                        "src": original_sequence,
                        "trg": mutated_sequence
                    }
                else:
                    json_entry = {
                        "src": mutated_sequence,
                        "trg": original_sequence
                    }
                json_data.append(json_entry)
                
        except Exception as e:
            print(f"处理序列 {row['sequence_id']} 时出错: {e}")
            continue
    
    # 保存比较结果到CSV
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        output_csv_path = os.path.join(
            output_folder, 
            f"mutation_comparison_{os.path.basename(csv_file_path)}"
        )
        comparison_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"突变比较结果已保存至: {output_csv_path}")
    
    # 保存JSON数据
    if json_data:
        json_filename = os.path.splitext(os.path.basename(csv_file_path))[0] + '_comparison.json'
        json_path = os.path.join(output_folder, json_filename)
        
        # 保存为每行一个JSON对象
        with open(json_path, 'w', encoding='utf-8') as f:
            for entry in json_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"JSON比较数据已保存至: {json_path}")
    
    return comparison_results, json_data

def batch_process_csv_with_mutations(input_folder, output_folder, mutation_rate=0.05):
    """
    批量处理文件夹中的所有CSV文件
    
    Args:
        input_folder (str): 输入CSV文件夹路径
        output_folder (str): 输出文件夹路径
        mutation_rate (float): 突变率
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 收集所有CSV文件
    csv_files = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv') and filename.startswith('standard_sequences_batch_'):
            csv_files.append(os.path.join(input_folder, filename))
    
    print(f"找到 {len(csv_files)} 个CSV文件需要处理")
    
    all_comparison_results = []
    all_json_data = []
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        comparison_results, json_data = process_csv_with_mutations(
            csv_file, output_folder, mutation_rate
        )
        all_comparison_results.extend(comparison_results)
        all_json_data.extend(json_data)
        
        # 打印当前文件的统计信息
        if comparison_results:
            print_mutation_statistics(comparison_results, os.path.basename(csv_file))
    
    # 生成总体统计
    if all_comparison_results:
        generate_mutation_overall_statistics(all_comparison_results, output_folder)
    
    print(f"\n处理完成! 共处理 {len(all_comparison_results)} 对序列比较")
    return all_comparison_results, all_json_data

def print_mutation_statistics(comparison_results, filename):
    """打印单个文件的突变统计信息"""
    df = pd.DataFrame(comparison_results)
    
    print(f"\n=== {filename} 突变分析统计 ===")
    print(f"总序列对: {len(df)}")
    print(f"原始序列平均得分: {df['original_score'].mean():.3f} ± {df['original_score'].std():.3f}")
    print(f"突变序列平均得分: {df['mutated_score'].mean():.3f} ± {df['mutated_score'].std():.3f}")
    
    # 得分变化统计
    improved = len(df[df['score_difference'] > 0])
    worsened = len(df[df['score_difference'] < 0])
    unchanged = len(df[df['score_difference'] == 0])
    
    print(f"得分提升的序列: {improved} ({improved/len(df)*100:.1f}%)")
    print(f"得分降低的序列: {worsened} ({worsened/len(df)*100:.1f}%)")
    print(f"得分不变的序列: {unchanged} ({unchanged/len(df)*100:.1f}%)")
    
    # 平均突变数量
    avg_mutations = df['num_mutations'].mean()
    print(f"平均突变位点数: {avg_mutations:.1f}")

def generate_mutation_overall_statistics(comparison_results, output_folder):
    """生成总体突变统计报告"""
    df = pd.DataFrame(comparison_results)
    
    stats_path = os.path.join(output_folder, "mutation_overall_statistics.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("序列突变分析总体统计\n")
        f.write("=" * 50 + "\n")
        f.write(f"总序列对: {len(df)}\n")
        f.write(f"原始序列平均得分: {df['original_score'].mean():.3f} ± {df['original_score'].std():.3f}\n")
        f.write(f"突变序列平均得分: {df['mutated_score'].mean():.3f} ± {df['mutated_score'].std():.3f}\n")
        
        # 得分变化统计
        improved = len(df[df['score_difference'] > 0])
        worsened = len(df[df['score_difference'] < 0])
        unchanged = len(df[df['score_difference'] == 0])
        
        f.write(f"\n得分变化统计:\n")
        f.write(f"  得分提升的序列: {improved} ({improved/len(df)*100:.1f}%)\n")
        f.write(f"  得分降低的序列: {worsened} ({worsened/len(df)*100:.1f}%)\n")
        f.write(f"  得分不变的序列: {unchanged} ({unchanged/len(df)*100:.1f}%)\n")
        
        # 得分变化分布
        f.write(f"\n得分变化分布:\n")
        diff_bins = [-1, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1]
        diff_labels = ['大幅下降(<-0.5)', '中度下降(-0.5~-0.2)', '轻微下降(-0.2~-0.05)', 
                      '基本不变(-0.05~0.05)', '轻微提升(0.05~0.2)', '中度提升(0.2~0.5)', '大幅提升(>0.5)']
        
        df['diff_category'] = pd.cut(df['score_difference'], bins=diff_bins, labels=diff_labels)
        diff_dist = df['diff_category'].value_counts().sort_index()
        
        for category, count in diff_dist.items():
            percentage = (count / len(df)) * 100
            f.write(f"  {category}: {count} 对序列 ({percentage:.1f}%)\n")
        
        # 突变数量统计
        f.write(f"\n突变数量统计:\n")
        mutation_counts = df['num_mutations'].value_counts().sort_index()
        for count, freq in mutation_counts.items():
            f.write(f"  {count} 个突变: {freq} 条序列 ({freq/len(df)*100:.1f}%)\n")
    
    print(f"总体突变统计已保存至: {stats_path}")

# 主执行函数
if __name__ == "__main__":
    # 处理CSV文件并进行突变分析
    input_csv_folder = r"/inspire/hdd/project/embodied-multimodality/public/hcd/Moss/dataset_washing/55_60seq/TODO"  # 标准序列CSV文件夹
    output_mutation_folder = r"/inspire/hdd/project/embodied-multimodality/public/hcd/Moss/dataset_washing/mutation_compare"  # 突变分析输出文件夹
    
    if os.path.exists(input_csv_folder):
        all_comparisons, all_json = batch_process_csv_with_mutations(
            input_folder=input_csv_folder,
            output_folder=output_mutation_folder,
            mutation_rate=0.05  # 5%的突变率
        )
        print(f"\n突变分析完成! 共生成 {len(all_json)} 个JSON条目")
    else:
        print(f"输入文件夹不存在: {input_csv_folder}")
# Running code:"python /inspire/hdd/project/embodied-multimodality/public/hcd/Moss/dataset_washing/sequence_compare.py"