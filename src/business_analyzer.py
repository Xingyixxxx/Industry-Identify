#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主营业务分词与旅游行业分析核心模块
Core module for business text analysis and tourism industry identification
"""

import pandas as pd
import jieba
from collections import Counter, defaultdict
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BusinessAnalyzer:
    """主营业务分析器"""
    
    def __init__(self, config_dir='config', output_dir='output'):
        """初始化分析器"""
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 数据存储
        self.df = None
        self.business_texts = []
        self.word_freq = Counter()
        self.tourism_companies = []
        
        # 加载配置
        self.stop_words = self._load_stopwords()
        self.tourism_keywords = self._load_tourism_keywords()
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/analysis.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        logging.info(f"分析器初始化完成 - 停用词: {len(self.stop_words)}, 旅游分类: {len(self.tourism_keywords)}")
    
    def _load_stopwords(self):
        """加载停用词"""
        stopwords_file = self.config_dir / 'stopwords.txt'
        stop_words = set()
        
        if stopwords_file.exists():
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        stop_words.add(line)
        
        return stop_words
    
    def _load_tourism_keywords(self):
        """加载旅游关键词分类"""
        keywords_file = self.config_dir / 'tourism_keywords.txt'
        tourism_keywords = {}
        
        if keywords_file.exists():
            with open(keywords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and ':' in line:
                        category, keywords_str = line.split(':', 1)
                        keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
                        tourism_keywords[category] = keywords
        
        return tourism_keywords
    
    def load_data(self, csv_file):
        """加载CSV数据"""
        logging.info(f"加载数据文件: {csv_file}")
        self.df = pd.read_csv(csv_file, encoding='utf-8')
        self.business_texts = self.df['MAINBUSSINESS'].dropna().tolist()
        logging.info(f"数据加载完成 - 总记录: {len(self.df)}, 有效业务描述: {len(self.business_texts)}")
    
    def segment_words(self):
        """分词处理"""
        logging.info("开始分词处理...")
        all_words = []
        
        for text in self.business_texts:
            # 文本预处理
            clean_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', str(text))
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            # 分词
            words = jieba.lcut(clean_text)
            
            # 过滤停用词和短词
            filtered_words = [
                word for word in words 
                if len(word) >= 2 and word not in self.stop_words
            ]
            all_words.extend(filtered_words)
        
        self.word_freq = Counter(all_words)
        logging.info(f"分词完成 - 总词数: {len(all_words)}, 不重复词数: {len(self.word_freq)}")
        
        return self.word_freq
    
    def export_word_frequency(self, top_n=1000):
        """导出词频统计到CSV"""
        if not self.word_freq:
            logging.warning("词频数据为空，请先执行分词")
            return
        
        # 准备数据
        word_data = []
        total_count = sum(self.word_freq.values())
        
        for rank, (word, freq) in enumerate(self.word_freq.most_common(top_n), 1):
            percentage = (freq / total_count) * 100
            word_data.append({
                '排名': rank,
                '词汇': word,
                '频次': freq,
                '占比(%)': round(percentage, 3)
            })
        
        # 保存到CSV
        df_words = pd.DataFrame(word_data)
        output_file = self.output_dir / 'word_frequency.csv'
        df_words.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 保存统计信息
        stats_file = self.output_dir / 'word_statistics.txt'
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"词频统计报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总词数: {total_count:,}\n")
            f.write(f"不重复词数: {len(self.word_freq):,}\n")
            f.write(f"导出词数: {len(word_data):,}\n")
            f.write(f"覆盖率: {sum(item['频次'] for item in word_data) / total_count * 100:.2f}%\n")
        
        logging.info(f"词频统计已导出: {output_file}")
        logging.info(f"统计信息已保存: {stats_file}")
        
        return output_file
    
    def identify_tourism_companies(self):
        """识别旅游相关公司"""
        logging.info("开始识别旅游相关公司...")
        self.tourism_companies = []
        
        for idx, text in enumerate(self.business_texts):
            company_info = self.df.iloc[idx]
            matched_categories = []
            
            # 检查旅游关键词
            for category, keywords in self.tourism_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        matched_categories.append(f"{category}({keyword})")
            
            if matched_categories:
                self.tourism_companies.append({
                    'Symbol': company_info['Symbol'],
                    'ShortName': company_info['ShortName'],
                    'EndDate': company_info['EndDate'],
                    'Business': text,
                    'Categories': ', '.join(matched_categories)
                })
        
        logging.info(f"识别完成 - 旅游相关公司记录: {len(self.tourism_companies)}")
        return self.tourism_companies
    
    def add_tourism_flags(self):
        """为原始数据添加旅游标识"""
        if not self.tourism_companies:
            self.identify_tourism_companies()
        
        # 创建旅游公司集合
        tourism_symbols = set(item['Symbol'] for item in self.tourism_companies)
        
        # 添加标识列
        self.df['IS_TOURISM'] = self.df['Symbol'].isin(tourism_symbols).astype(int)
        
        # 保存标注后的数据
        output_file = self.output_dir / 'data_with_tourism_flag.csv'
        self.df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 统计信息
        tourism_count = self.df['IS_TOURISM'].sum()
        total_count = len(self.df)
        
        logging.info(f"数据标注完成 - 旅游记录: {tourism_count}/{total_count} ({tourism_count/total_count*100:.1f}%)")
        logging.info(f"标注数据已保存: {output_file}")
        
        return output_file
    
    def generate_analysis_report(self):
        """生成分析报告"""
        if not self.word_freq:
            logging.warning("请先执行分词分析")
            return
        
        report_file = self.output_dir / 'analysis_report.md'
        
        # 统计信息
        total_words = sum(self.word_freq.values())
        unique_words = len(self.word_freq)
        tourism_records = len(self.tourism_companies)
        total_records = len(self.df) if self.df is not None else 0
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 主营业务分词与旅游行业分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## 分词统计\n")
            f.write(f"- 总词数: {total_words:,}\n")
            f.write(f"- 不重复词数: {unique_words:,}\n")
            f.write(f"- 平均词长: {total_words/unique_words:.1f}\n\n")
            
            f.write(f"## 旅游行业识别\n")
            f.write(f"- 总记录数: {total_records:,}\n")
            f.write(f"- 旅游相关记录: {tourism_records:,}\n")
            if total_records > 0:
                f.write(f"- 旅游比例: {tourism_records/total_records*100:.1f}%\n\n")
            
            f.write(f"## 高频词汇 (Top 20)\n")
            for i, (word, freq) in enumerate(self.word_freq.most_common(20), 1):
                f.write(f"{i}. {word}: {freq:,}次\n")
            
            f.write(f"\n## 旅游分类统计\n")
            category_stats = defaultdict(int)
            for company in self.tourism_companies:
                categories = company['Categories'].split(', ')
                for cat in categories:
                    if '(' in cat:
                        main_cat = cat.split('(')[0]
                        category_stats[main_cat] += 1
            
            for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {category}: {count}家公司\n")
        
        logging.info(f"分析报告已生成: {report_file}")
        return report_file
    
    def create_visualization(self):
        """创建可视化图表"""
        if not self.word_freq:
            logging.warning("请先执行分词分析")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 词频柱状图
        top_words = dict(self.word_freq.most_common(20))
        ax1 = axes[0, 0]
        words = list(top_words.keys())
        freqs = list(top_words.values())
        ax1.barh(words, freqs)
        ax1.set_title('高频词汇 (Top 20)')
        ax1.set_xlabel('频次')
        
        # 2. 旅游分类饼图
        ax2 = axes[0, 1]
        category_stats = defaultdict(int)
        for company in self.tourism_companies:
            categories = company['Categories'].split(', ')
            for cat in categories:
                if '(' in cat:
                    main_cat = cat.split('(')[0]
                    category_stats[main_cat] += 1
        
        if category_stats:
            top_categories = dict(sorted(category_stats.items(), key=lambda x: x[1], reverse=True)[:10])
            ax2.pie(top_categories.values(), labels=top_categories.keys(), autopct='%1.1f%%')
            ax2.set_title('旅游分类分布 (Top 10)')
        
        # 3. 词频分布
        ax3 = axes[1, 0]
        freq_counts = Counter(self.word_freq.values())
        frequencies = sorted(freq_counts.keys())[:20]
        counts = [freq_counts[f] for f in frequencies]
        ax3.bar(frequencies, counts)
        ax3.set_title('词频分布')
        ax3.set_xlabel('出现频次')
        ax3.set_ylabel('词汇数量')
        
        # 4. 累积分布
        ax4 = axes[1, 1]
        sorted_freqs = sorted(self.word_freq.values(), reverse=True)
        cumsum = np.cumsum(sorted_freqs)
        ax4.plot(range(1, min(1001, len(cumsum)+1)), cumsum[:1000])
        ax4.set_title('词频累积分布')
        ax4.set_xlabel('词汇排名')
        ax4.set_ylabel('累积频次')
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = self.output_dir / 'analysis_charts.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logging.info(f"可视化图表已保存: {chart_file}")
        return chart_file
    
    def run_complete_analysis(self, csv_file):
        """运行完整分析流程"""
        try:
            # 1. 加载数据
            self.load_data(csv_file)
            
            # 2. 分词分析
            self.segment_words()
            
            # 3. 导出词频统计
            self.export_word_frequency()
            
            # 4. 识别旅游公司
            self.identify_tourism_companies()
            
            # 5. 添加旅游标识
            self.add_tourism_flags()
            
            # 6. 生成报告
            self.generate_analysis_report()
            
            # 7. 创建可视化
            self.create_visualization()
            
            logging.info("完整分析流程执行完成")
            
        except Exception as e:
            logging.error(f"分析过程出错: {e}")
            raise

def main():
    """主函数"""
    analyzer = BusinessAnalyzer()
    analyzer.run_complete_analysis('data/STK_LISTEDCOINFOANL.csv')

if __name__ == "__main__":
    main()
