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

def merge_api_results(base_file=None, api_results_file=None, output_dir='output'):
    """合并API分析结果"""
    print("🔗 开始合并API分析结果...")

    analyzer = BusinessAnalyzer(output_dir=output_dir)
    result_file = analyzer.merge_api_results(base_file, api_results_file)

    print("✅ API结果合并完成！")
    print(f"📁 合并文件: {result_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='主营业务分词与旅游行业分析工具')
    parser.add_argument('data_file', nargs='?', help='CSV数据文件路径 (merge模式下可选)')
    parser.add_argument('--mode', choices=['basic', 'api', 'merge'], default='api',
                       help='分析模式: basic(基础词频分析), api(API智能分析+自动合并), merge(仅合并已有结果)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='API分析的样本大小 (默认: 全部数据)')
    parser.add_argument('--config-dir', default='config',
                       help='配置文件目录 (默认: config)')
    parser.add_argument('--output-dir', default='output',
                       help='输出文件目录 (默认: output)')
    parser.add_argument('--no-deduplication', action='store_true',
                       help='禁用去重优化，处理所有重复文本')
    parser.add_argument('--base-file',
                       help='基础分析文件路径 (仅merge模式使用，默认: output/data_with_tourism_flag.csv)')
    parser.add_argument('--api-results-file',
                       help='API分析结果文件路径 (仅merge模式使用，默认: 最新的 api_analysis_results_*.csv)')

    args = parser.parse_args()

    # 创建输出目录
    Path(args.output_dir).mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)

    # 处理合并模式
    if args.mode == 'merge':
        print("=" * 60)
        print("🔗 API分析结果合并工具")
        print("=" * 60)
        try:
            merge_api_results(args.base_file, args.api_results_file, args.output_dir)
            print("🎉 API结果合并完成！")
        except Exception as e:
            print(f"❌ 合并过程中出现错误: {e}")
            print("📋 请查看 logs/ 目录下的日志文件获取详细信息")
        return

    # 检查数据文件（仅在非合并模式下需要）
    if not args.data_file:
        print(f"❌ 非merge模式下必须提供数据文件路径")
        return
    if not Path(args.data_file).exists():
        print(f"❌ 数据文件不存在: {args.data_file}")
        return

    print("=" * 60)
    print("🏨 主营业务分词与旅游行业分析工具")
    print("=" * 60)

    try:
        if args.mode == 'basic':
            run_basic_analysis(args.data_file, args.config_dir, args.output_dir)
        elif args.mode == 'api':
            enable_dedup = not args.no_deduplication
            run_api_analysis(args.data_file, args.sample_size, args.config_dir, args.output_dir,
                           enable_deduplication=enable_dedup)
            print("💡 API分析已自动检测并合并基础分析结果（如果存在）")

        print("🎉 分析任务完成！")

    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        print("📋 请查看 logs/ 目录下的日志文件获取详细信息")

if __name__ == "__main__":
    main()
