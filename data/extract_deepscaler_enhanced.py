#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepScaler数据提取工具（增强版）
提取deepscaler.json中前N个{question, gt}对的脚本
支持自定义数量、统计分析和格式选择
python extract_deepscaler_enhanced.py 100 csv
"""

import json
import sys
import os
from collections import Counter

def analyze_data(data):
    """分析数据统计信息"""
    print("\n数据分析:")
    print("-" * 30)
    
    # 基本统计
    print(f"总条目数: {len(data)}")
    
    # 字段统计
    if data:
        fields = list(data[0].keys())
        print(f"字段: {fields}")
        
        # 答案长度统计
        gt_lengths = [len(str(item.get('gt', ''))) for item in data]
        print(f"答案平均长度: {sum(gt_lengths)/len(gt_lengths):.1f} 字符")
        
        # 问题长度统计
        question_lengths = [len(str(item.get('question', ''))) for item in data]
        print(f"问题平均长度: {sum(question_lengths)/len(question_lengths):.1f} 字符")
        
        # 检查空字段
        empty_questions = sum(1 for item in data if not item.get('question', '').strip())
        empty_gts = sum(1 for item in data if not item.get('gt', '').strip())
        
        print(f"空问题数: {empty_questions}")
        print(f"空答案数: {empty_gts}")

def extract_data(input_file, output_file, num_items=100, format_type='json'):
    """
    从deepscaler.json文件中提取前N个{question, gt}对
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        num_items: 要提取的条目数量（默认100）
        format_type: 输出格式 ('json', 'csv', 'txt')
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"错误: 找不到文件 {input_file}")
            return False
            
        # 读取原始JSON文件
        print(f"正在读取文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"原文件总共包含 {len(data)} 个条目")
        
        # 验证数据格式
        if not data or not isinstance(data, list):
            print("错误: 文件格式不正确，应该是包含字典的列表")
            return False
        
        # 验证必需字段
        required_fields = {'problem', 'answer', 'solution'}
        if not all(required_fields.issubset(item.keys()) for item in data[:5]):
            print("警告: 某些条目可能缺少必需字段 (problem, answer, solution)")
        
        # 提取指定数量的条目并转换字段名
        num_items = min(num_items, len(data))
        raw_data = data[:num_items]
        
        # 转换字段名并删除solution字段
        extracted_data = []
        for item in raw_data:
            new_item = {
                'question': item.get('problem', ''),
                'gt': item.get('answer', '')
            }
            extracted_data.append(new_item)
        
        print(f"提取前 {num_items} 个条目")
        
        # 数据分析
        analyze_data(extracted_data)
        
        # 根据格式类型保存文件
        if format_type.lower() == 'json':
            save_as_json(extracted_data, output_file)
        elif format_type.lower() == 'csv':
            save_as_csv(extracted_data, output_file)
        elif format_type.lower() == 'txt':
            save_as_txt(extracted_data, output_file)
        else:
            print(f"不支持的格式: {format_type}")
            return False
        
        print(f"已成功保存到文件: {output_file}")
        
        # 显示示例条目
        show_sample(extracted_data)
        
        return True
        
    except json.JSONDecodeError:
        print(f"错误: {input_file} 不是有效的JSON文件")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def save_as_json(data, output_file):
    """保存为JSON格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_as_csv(data, output_file):
    """保存为CSV格式"""
    import csv
    
    output_file = output_file.replace('.json', '.csv')
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'gt'])
        writer.writeheader()
        writer.writerows(data)

def save_as_txt(data, output_file):
    """保存为TXT格式"""
    output_file = output_file.replace('.json', '.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(data, 1):
            f.write(f"=== 第 {i} 题 ===\n")
            f.write(f"问题: {item.get('question', 'N/A')}\n\n")
            f.write(f"答案: {item.get('gt', 'N/A')}\n\n")
            f.write("-" * 80 + "\n\n")

def show_sample(data):
    """显示示例条目"""
    if not data:
        return
        
    print("\n示例条目:")
    print("=" * 50)
    
    # 显示第一个条目
    first = data[0]
    print("第 1 个条目:")
    print(f"问题: {first.get('question', 'N/A')[:150]}...")
    print(f"答案: {first.get('gt', 'N/A')}")
    
    if len(data) > 1:
        print(f"\n第 {len(data)} 个条目:")
        last = data[-1]
        print(f"问题: {last.get('question', 'N/A')[:150]}...")
        print(f"答案: {last.get('gt', 'N/A')}")

def main():
    print("DeepScaler数据提取工具（增强版）")
    print("=" * 60)
    
    # 默认参数
    input_file = "deepscaler.json"
    output_file = "deepscaler_first_100.json"
    num_items = 100
    format_type = 'json'
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        try:
            num_items = int(sys.argv[1])
            print(f"设置提取数量: {num_items}")
        except ValueError:
            print("警告: 无效的数量参数，使用默认值100")
    
    if len(sys.argv) > 2:
        format_type = sys.argv[2].lower()
        if format_type in ['json', 'csv', 'txt']:
            print(f"设置输出格式: {format_type}")
        else:
            print("警告: 无效的格式参数，使用默认格式json")
            format_type = 'json'
    
    # 根据格式调整输出文件名
    base_name = f"deepscaler_first_{num_items}"
    if format_type == 'csv':
        output_file = f"{base_name}.csv"
    elif format_type == 'txt':
        output_file = f"{base_name}.txt"
    else:
        output_file = f"{base_name}.json"
    
    success = extract_data(input_file, output_file, num_items, format_type)
    
    if success:
        print("\n✅ 提取完成！")
        print(f"输出文件: {output_file}")
    else:
        print("\n❌ 提取失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
