#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API智能分析模块 - 优化版
Optimized API analysis module with batch processing
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Any
import backoff

class APIAnalyzer:
    """API智能分析器 - 批量处理优化版"""
    
    def __init__(self, config_dir='config', output_dir='output'):
        """初始化API分析器"""
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载配置
        self.config = self._load_config()
        self.request_times = []
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/api_analysis.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        logging.info(f"API分析器初始化完成 - 模型: {self.config.get('model', 'N/A')}")
    
    def _load_config(self):
        """加载API配置"""
        config_file = self.config_dir / 'api_config.json'
        
        # 默认配置
        default_config = {
            "api_key": os.getenv('SILICONFLOW_API_KEY', ''),
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "deepseek-ai/DeepSeek-V3",
            "max_requests_per_minute": 20,
            "batch_size": 8,
            "timeout": 60
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logging.warning(f"配置文件加载失败，使用默认配置: {e}")
        
        if not default_config['api_key']:
            logging.warning("未设置API密钥，请设置环境变量SILICONFLOW_API_KEY")
        
        return default_config
    
    def _rate_limit(self):
        """请求限流"""
        current_time = time.time()
        
        # 清理超过1分钟的请求记录
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # 检查是否超过限制
        max_requests = self.config.get('max_requests_per_minute', 20)
        if len(self.request_times) >= max_requests:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                logging.info(f"达到请求限制，等待 {sleep_time:.1f} 秒")
                time.sleep(sleep_time)
        
        self.request_times.append(current_time)
    
    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
    def _make_api_request(self, business_texts: List[str], company_names: List[str]) -> Dict[str, Any]:
        """发送批量API请求"""
        self._rate_limit()
        
        # 构建批量分析提示
        batch_prompt = self._build_batch_prompt(business_texts, company_names)
        
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config['model'],
            "messages": [
                {
                    "role": "system",
                    "content": "你是专业的商业分析师，擅长分析公司主营业务并判断是否与旅游行业相关。请对每家公司进行分析，给出是否属于旅游相关行业的判断和评分。"
                },
                {
                    "role": "user",
                    "content": batch_prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                f"{self.config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=self.config['timeout']
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"API请求失败: {e}")
            raise
    
    def _build_batch_prompt(self, business_texts: List[str], company_names: List[str]) -> str:
        """构建批量分析提示"""
        prompt = "请分析以下公司的主营业务，判断是否与旅游行业相关：\n\n"
        
        for i, (name, business) in enumerate(zip(company_names, business_texts), 1):
            prompt += f"{i}. 公司：{name}\n"
            prompt += f"   业务：{business}\n\n"
        
        prompt += """
请对每家公司按以下格式回答：
公司X: 是否旅游相关(是/否) | 评分(0-100) | 分类 | 理由

要求：
1. 评分标准：0-30分为无关，31-60分为间接相关，61-100分为直接相关
2. 分类包括：酒店住宿、餐饮服务、旅游服务、交通运输、娱乐休闲、零售商业、会展服务等
3. 理由简洁明了，不超过30字
"""
        return prompt
    
    def _parse_batch_response(self, response_content: str, company_names: List[str]) -> List[Dict[str, Any]]:
        """解析批量响应结果"""
        results = []
        lines = response_content.split('\n')
        
        for i, company_name in enumerate(company_names):
            result = {
                'company_name': company_name,
                'is_tourism_related': False,
                'tourism_score': 0,
                'tourism_category': '',
                'analysis_reason': '',
                'api_response': response_content
            }
            
            # 查找对应公司的分析结果
            company_pattern = f"公司{i+1}"
            for line in lines:
                if company_pattern in line and '|' in line:
                    try:
                        parts = line.split('|')
                        if len(parts) >= 4:
                            # 解析是否旅游相关
                            tourism_part = parts[0].strip()
                            result['is_tourism_related'] = '是' in tourism_part
                            
                            # 解析评分
                            score_part = parts[1].strip()
                            import re
                            score_match = re.search(r'\d+', score_part)
                            if score_match:
                                result['tourism_score'] = int(score_match.group())
                            
                            # 解析分类和理由
                            result['tourism_category'] = parts[2].strip()
                            result['analysis_reason'] = parts[3].strip()
                    except Exception as e:
                        logging.warning(f"解析公司 {company_name} 结果失败: {e}")
                    break
            
            results.append(result)
        
        return results
    
    def analyze_batch(self, business_data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """批量分析业务文本"""
        if not self.config['api_key']:
            logging.error("API密钥未设置，无法进行分析")
            return []
        
        batch_size = self.config.get('batch_size', 8)
        all_results = []
        
        logging.info(f"开始批量分析 {len(business_data)} 条记录，批次大小: {batch_size}")
        
        for i in range(0, len(business_data), batch_size):
            batch = business_data[i:i+batch_size]
            
            try:
                # 提取批次数据
                business_texts = [item.get('business_text', '') for item in batch]
                company_names = [item.get('company_name', '') for item in batch]
                
                logging.info(f"处理批次 {i//batch_size + 1}/{(len(business_data)-1)//batch_size + 1}")
                
                # 发送API请求
                response = self._make_api_request(business_texts, company_names)
                content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # 解析结果
                batch_results = self._parse_batch_response(content, company_names)
                
                # 合并原始数据
                for j, result in enumerate(batch_results):
                    original_data = batch[j]
                    combined_result = {**original_data, **result}
                    combined_result['timestamp'] = datetime.now().isoformat()
                    all_results.append(combined_result)
                
                # 短暂延迟
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"批次 {i//batch_size + 1} 处理失败: {e}")
                # 添加错误记录
                for item in batch:
                    error_result = {
                        **item,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    all_results.append(error_result)
        
        logging.info(f"批量分析完成，共处理 {len(all_results)} 条记录")
        return all_results
    
    def analyze_csv_file(self, csv_file: str, sample_size: int = None) -> str:
        """分析CSV文件"""
        logging.info(f"开始分析CSV文件: {csv_file}")
        
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8')
        
        # 采样
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logging.info(f"随机采样 {sample_size} 条记录")
        
        # 准备分析数据
        business_data = []
        for _, row in df.iterrows():
            business_data.append({
                'company_name': str(row.get('ShortName', '')),
                'business_text': str(row.get('MAINBUSSINESS', '')),
                'symbol': str(row.get('Symbol', '')),
                'end_date': str(row.get('EndDate', ''))
            })
        
        # 执行批量分析
        results = self.analyze_batch(business_data)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为CSV
        results_df = pd.DataFrame(results)
        csv_file = self.output_dir / f"api_analysis_results_{timestamp}.csv"
        results_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # 生成统计报告
        self._generate_api_report(results, timestamp)
        
        logging.info(f"API分析结果已保存: {csv_file}")
        return str(csv_file)
    
    def _generate_api_report(self, results: List[Dict[str, Any]], timestamp: str):
        """生成API分析报告"""
        report_file = self.output_dir / f"api_analysis_report_{timestamp}.md"
        
        # 统计信息
        total_count = len(results)
        tourism_count = sum(1 for r in results if r.get('is_tourism_related', False))
        error_count = sum(1 for r in results if 'error' in r)
        
        # 评分分布
        scores = [r.get('tourism_score', 0) for r in results if 'tourism_score' in r]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# API智能分析报告\n\n")
            f.write(f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**使用模型**: {self.config['model']}\n\n")
            
            f.write(f"## 统计概览\n")
            f.write(f"- 总分析数量: {total_count}\n")
            f.write(f"- 旅游相关企业: {tourism_count} ({tourism_count/total_count*100:.1f}%)\n")
            f.write(f"- 分析失败: {error_count}\n")
            f.write(f"- 平均旅游相关度: {avg_score:.1f}分\n\n")
            
            # 高分企业
            high_score_companies = [
                r for r in results 
                if r.get('tourism_score', 0) >= 80 and r.get('is_tourism_related', False)
            ]
            
            if high_score_companies:
                f.write(f"## 高分旅游企业 (≥80分)\n")
                for company in sorted(high_score_companies, key=lambda x: x.get('tourism_score', 0), reverse=True):
                    f.write(f"- **{company.get('company_name', 'N/A')}** (评分: {company.get('tourism_score', 0)})\n")
                    f.write(f"  - 分类: {company.get('tourism_category', 'N/A')}\n")
                    f.write(f"  - 理由: {company.get('analysis_reason', 'N/A')}\n\n")
        
        logging.info(f"API分析报告已生成: {report_file}")

def main():
    """主函数"""
    analyzer = APIAnalyzer()
    
    # 分析前20条记录作为示例
    result_file = analyzer.analyze_csv_file('data/STK_LISTEDCOINFOANL.csv', sample_size=20)
    print(f"API分析完成，结果保存在: {result_file}")

if __name__ == "__main__":
    main()
