#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主营业务分词与旅游行业分析 - 主程序
Main program for business text analysis and tourism industry identification
"""

import sys
import argparse
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / 'src'))

from business_analyzer import BusinessAnalyzer
from api_analyzer import APIAnalyzer

def run_basic_analysis(data_file, config_dir='config', output_dir='output'):
    """运行基础分析"""
    print("🚀 开始基础分词与旅游行业分析...")
    
    analyzer = BusinessAnalyzer(config_dir=config_dir, output_dir=output_dir)
    analyzer.run_complete_analysis(data_file)
    
    print("✅ 基础分析完成！")
    print(f"📁 结果文件保存在: {output_dir}/")
    print("   - word_frequency.csv: 词频统计")
    print("   - data_with_tourism_flag.csv: 添加旅游标识的数据")
    print("   - analysis_report.md: 分析报告")
    print("   - analysis_charts.png: 可视化图表")

def run_api_analysis(data_file, sample_size=None, config_dir='config', output_dir='output',
                    enable_deduplication=True):
    """运行API智能分析"""
    print("🤖 开始API智能分析...")

    if enable_deduplication:
        print("📊 启用去重优化，相同业务文本只分析一次")
    else:
        print("⚠️  已禁用去重优化，将分析所有重复文本")

    analyzer = APIAnalyzer(config_dir=config_dir, output_dir=output_dir)
    result_file = analyzer.analyze_csv_file(data_file, sample_size=sample_size,
                                          enable_deduplication=enable_deduplication)

    print("✅ API分析完成！")
    print(f"📁 结果文件: {result_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='主营业务分词与旅游行业分析工具')
    parser.add_argument('data_file', help='CSV数据文件路径')
    parser.add_argument('--mode', choices=['basic', 'api', 'both'], default='basic',
                       help='分析模式: basic(基础分析), api(API分析), both(两种都运行)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='API分析的样本大小 (默认: 全部数据)')
    parser.add_argument('--config-dir', default='config',
                       help='配置文件目录 (默认: config)')
    parser.add_argument('--output-dir', default='output',
                       help='输出文件目录 (默认: output)')
    parser.add_argument('--no-deduplication', action='store_true',
                       help='禁用去重优化，处理所有重复文本')
    
    args = parser.parse_args()
    
    # 检查数据文件
    if not Path(args.data_file).exists():
        print(f"❌ 数据文件不存在: {args.data_file}")
        return
    
    # 创建输出目录
    Path(args.output_dir).mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    print("=" * 60)
    print("🏨 主营业务分词与旅游行业分析工具")
    print("=" * 60)
    
    try:
        if args.mode in ['basic', 'both']:
            run_basic_analysis(args.data_file, args.config_dir, args.output_dir)
            print()
        
        if args.mode in ['api', 'both']:
            enable_dedup = not args.no_deduplication
            run_api_analysis(args.data_file, args.sample_size, args.config_dir, args.output_dir,
                           enable_deduplication=enable_dedup)
        
        print("🎉 所有分析任务完成！")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        print("📋 请查看 logs/ 目录下的日志文件获取详细信息")

if __name__ == "__main__":
    main()
