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
            "max_requests_per_minute": 50,
            "batch_size": 20,
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
    
    def _deduplicate_business_texts(self, business_data: List[Dict[str, str]]) -> tuple:
        """对业务文本去重，返回去重后的数据和映射关系

        Returns:
            tuple: (unique_business_data, text_to_records_map, dedup_stats)
        """
        text_to_records = {}
        unique_texts = {}

        # 按业务文本分组
        for i, record in enumerate(business_data):
            business_text = record.get('business_text', '').strip()

            if business_text not in text_to_records:
                text_to_records[business_text] = []
                unique_texts[business_text] = record  # 保存第一个出现的记录作为代表

            text_to_records[business_text].append((i, record))

        # 创建去重后的数据列表
        unique_business_data = []
        text_to_index_map = {}

        for text, representative_record in unique_texts.items():
            text_to_index_map[text] = len(unique_business_data)
            unique_business_data.append(representative_record)

        # 统计信息
        original_count = len(business_data)
        unique_count = len(unique_business_data)
        duplicate_count = original_count - unique_count

        dedup_stats = {
            'original_count': original_count,
            'unique_count': unique_count,
            'duplicate_count': duplicate_count,
            'dedup_ratio': duplicate_count / original_count if original_count > 0 else 0
        }

        logging.info(f"业务文本去重完成: {original_count} -> {unique_count} "
                    f"(去重 {duplicate_count} 条, 比例 {dedup_stats['dedup_ratio']:.1%})")

        return unique_business_data, text_to_records, dedup_stats

    def _apply_results_to_duplicates(self, unique_results: List[Dict[str, Any]],
                                   text_to_records: Dict[str, List],
                                   original_business_data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """将去重分析的结果应用到所有重复记录上"""
        all_results = [None] * len(original_business_data)

        for result in unique_results:
            business_text = result.get('business_text', '').strip()

            if business_text in text_to_records:
                # 将结果应用到所有具有相同业务文本的记录
                for original_index, original_record in text_to_records[business_text]:
                    # 创建新的结果记录，保留原始记录的所有字段
                    combined_result = {**original_record}

                    # 添加分析结果字段
                    analysis_fields = [
                        'is_tourism_related', 'tourism_score', 'tourism_category',
                        'analysis_reason', 'api_response', 'timestamp', 'batch_number'
                    ]

                    for field in analysis_fields:
                        if field in result:
                            combined_result[field] = result[field]

                    # 标记这是从去重结果复制的
                    combined_result['is_deduplicated'] = True

                    all_results[original_index] = combined_result

        # 过滤掉None值（理论上不应该有）
        final_results = [r for r in all_results if r is not None]

        logging.info(f"去重结果应用完成: {len(unique_results)} 个唯一结果 -> {len(final_results)} 条最终记录")

        return final_results

    def analyze_batch_with_incremental_save(self, business_data: List[Dict[str, str]],
                                           progress_file: str = None,
                                           enable_deduplication: bool = True) -> List[Dict[str, Any]]:
        """批量分析业务文本，每批次后立即保存结果，支持去重优化"""
        if not self.config['api_key']:
            logging.error("API密钥未设置，无法进行分析")
            return []

        # 保存原始数据用于最终结果映射
        original_business_data = business_data.copy()

        # 去重处理
        if enable_deduplication and len(business_data) > 1:
            unique_business_data, text_to_records, dedup_stats = self._deduplicate_business_texts(business_data)
            business_data_to_process = unique_business_data

            logging.info(f"启用去重优化: 原始 {dedup_stats['original_count']} 条 -> "
                        f"需处理 {dedup_stats['unique_count']} 条 "
                        f"(节省 {dedup_stats['dedup_ratio']:.1%} API调用)")
        else:
            business_data_to_process = business_data
            text_to_records = None
            dedup_stats = None
            logging.info("未启用去重优化，将处理全部数据")

        batch_size = self.config.get('batch_size', 8)
        all_results = []

        # 设置进度文件路径
        if not progress_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            progress_file = self.output_dir / f"api_analysis_progress_{timestamp}.csv"
        else:
            progress_file = Path(progress_file)

        # 加载已有的进度文件（如果存在）
        if progress_file.exists():
            try:
                existing_df = pd.read_csv(progress_file, encoding='utf-8')
                all_results = existing_df.to_dict('records')
                logging.info(f"从进度文件加载了 {len(all_results)} 条已处理记录")
            except Exception as e:
                logging.warning(f"加载进度文件失败: {e}")

        logging.info(f"开始批量分析 {len(business_data_to_process)} 条去重后记录，批次大小: {batch_size}")
        logging.info(f"进度文件: {progress_file}")

        # 处理去重后的数据
        unique_results = []

        for i in range(0, len(business_data_to_process), batch_size):
            batch = business_data_to_process[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(business_data_to_process)-1)//batch_size + 1

            try:
                # 提取批次数据
                business_texts = [item.get('business_text', '') for item in batch]
                company_names = [item.get('company_name', '') for item in batch]

                logging.info(f"处理去重批次 {batch_num}/{total_batches} (记录 {i+1}-{min(i+batch_size, len(business_data_to_process))})")

                # 发送API请求
                response = self._make_api_request(business_texts, company_names)
                content = response.get('choices', [{}])[0].get('message', {}).get('content', '')

                # 解析结果
                batch_results = self._parse_batch_response(content, company_names)

                # 合并原始数据
                batch_combined_results = []
                for j, result in enumerate(batch_results):
                    original_data = batch[j]
                    combined_result = {**original_data, **result}
                    combined_result['timestamp'] = datetime.now().isoformat()
                    combined_result['batch_number'] = batch_num
                    combined_result['is_deduplicated'] = False  # 标记为原始分析结果
                    batch_combined_results.append(combined_result)

                # 添加到去重结果
                unique_results.extend(batch_combined_results)

                logging.info(f"去重批次 {batch_num} 完成，分析了 {len(batch_combined_results)} 条唯一记录")

                # 短暂延迟
                time.sleep(1)

            except Exception as e:
                logging.error(f"去重批次 {batch_num} 处理失败: {e}")
                # 添加错误记录
                error_results = []
                for item in batch:
                    error_result = {
                        **item,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                        'batch_number': batch_num,
                        'is_deduplicated': False
                    }
                    error_results.append(error_result)

                unique_results.extend(error_results)
                logging.info(f"去重批次 {batch_num} 错误已记录")

        # 如果启用了去重，将结果应用到所有重复记录
        if enable_deduplication and text_to_records:
            final_results = self._apply_results_to_duplicates(unique_results, text_to_records, original_business_data)
            logging.info(f"去重结果已应用到所有 {len(final_results)} 条原始记录")
        else:
            final_results = unique_results

        # 合并已有结果（如果有的话）
        if all_results:
            # 移除重复的已有结果
            existing_ids = set()
            for result in all_results:
                record_id = self._create_record_id(
                    result.get('company_name', ''),
                    result.get('business_text', ''),
                    result.get('symbol', '')
                )
                existing_ids.add(record_id)

            # 只添加新的结果
            new_results = []
            for result in final_results:
                record_id = self._create_record_id(
                    result.get('company_name', ''),
                    result.get('business_text', ''),
                    result.get('symbol', '')
                )
                if record_id not in existing_ids:
                    new_results.append(result)

            final_results = all_results + new_results
            logging.info(f"合并结果: 已有 {len(all_results)} + 新增 {len(new_results)} = 总计 {len(final_results)}")

        # 保存最终进度
        self._save_progress(final_results, progress_file)

        logging.info(f"批量分析完成，共处理 {len(final_results)} 条记录")
        if dedup_stats:
            logging.info(f"去重优化效果: 节省了 {dedup_stats['duplicate_count']} 次API调用 "
                        f"({dedup_stats['dedup_ratio']:.1%})")
        logging.info(f"最终进度文件: {progress_file}")

        return final_results, str(progress_file)

    def _save_progress(self, results: List[Dict[str, Any]], progress_file: Path):
        """保存进度到文件"""
        try:
            results_df = pd.DataFrame(results)
            results_df.to_csv(progress_file, index=False, encoding='utf-8-sig')
            logging.debug(f"进度已保存: {len(results)} 条记录 -> {progress_file}")
        except Exception as e:
            logging.error(f"保存进度文件失败: {e}")

    def analyze_batch(self, business_data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """批量分析业务文本（兼容性方法）"""
        results, _ = self.analyze_batch_with_incremental_save(business_data)
        return results
    
    def _load_processed_records(self, csv_file: str = None) -> set:
        """加载已处理的记录，返回已处理的公司标识符集合

        Args:
            csv_file: 原始CSV文件路径（用于日志记录）
        """
        processed_ids = set()

        # 查找最新的结果文件
        pattern = f"api_analysis_results_*.csv"
        result_files = list(self.output_dir.glob(pattern))

        if not result_files:
            logging.info("未找到已处理的结果文件，将从头开始处理")
            return processed_ids

        # 按时间戳排序，获取最新的文件
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)

        try:
            processed_df = pd.read_csv(latest_file, encoding='utf-8')

            # 使用公司名称+业务描述的组合作为唯一标识符
            for _, row in processed_df.iterrows():
                company_name = str(row.get('company_name', ''))
                business_text = str(row.get('business_text', ''))
                symbol = str(row.get('symbol', ''))

                # 创建唯一标识符
                record_id = f"{company_name}|{business_text}|{symbol}"
                processed_ids.add(record_id)

            source_info = f"来源文件: {csv_file}" if csv_file else ""
            logging.info(f"从 {latest_file.name} 加载了 {len(processed_ids)} 条已处理记录 {source_info}")

        except Exception as e:
            logging.warning(f"加载已处理记录失败: {e}")

        return processed_ids

    def _create_record_id(self, company_name: str, business_text: str, symbol: str) -> str:
        """创建记录的唯一标识符"""
        return f"{company_name}|{business_text}|{symbol}"

    def _filter_unprocessed_data(self, business_data: List[Dict[str, str]], processed_ids: set) -> List[Dict[str, str]]:
        """过滤出未处理的数据"""
        unprocessed_data = []

        for item in business_data:
            record_id = self._create_record_id(
                item.get('company_name', ''),
                item.get('business_text', ''),
                item.get('symbol', '')
            )

            if record_id not in processed_ids:
                unprocessed_data.append(item)

        return unprocessed_data

    def _find_latest_progress_file(self) -> Path:
        """查找最新的进度文件"""
        pattern = f"api_analysis_progress_*.csv"
        progress_files = list(self.output_dir.glob(pattern))

        if progress_files:
            latest_file = max(progress_files, key=lambda x: x.stat().st_mtime)
            return latest_file
        return None

    def _load_processed_records_from_progress(self) -> tuple:
        """从进度文件加载已处理的记录

        Returns:
            tuple: (processed_ids_set, progress_file_path, existing_results_list)
        """
        processed_ids = set()
        existing_results = []
        progress_file = self._find_latest_progress_file()

        if not progress_file or not progress_file.exists():
            logging.info("未找到进度文件，将从头开始处理")
            return processed_ids, None, existing_results

        try:
            progress_df = pd.read_csv(progress_file, encoding='utf-8')
            existing_results = progress_df.to_dict('records')

            # 创建已处理记录的ID集合
            for result in existing_results:
                company_name = str(result.get('company_name', ''))
                business_text = str(result.get('business_text', ''))
                symbol = str(result.get('symbol', ''))
                record_id = self._create_record_id(company_name, business_text, symbol)
                processed_ids.add(record_id)

            logging.info(f"从进度文件 {progress_file.name} 加载了 {len(processed_ids)} 条已处理记录")

        except Exception as e:
            logging.warning(f"加载进度文件失败: {e}")
            return set(), None, []

        return processed_ids, str(progress_file), existing_results

    def analyze_csv_file_incremental(self, csv_file: str, sample_size: int = None,
                                    force_restart: bool = False, enable_deduplication: bool = True) -> str:
        """增量分析CSV文件，支持断点续传"""
        logging.info(f"开始增量分析CSV文件: {csv_file}")

        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8')
        logging.info(f"读取到 {len(df)} 条记录")

        # 采样（如果指定了sample_size且小于总数据量）
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logging.info(f"随机采样 {sample_size} 条记录进行分析")

        # 准备分析数据
        business_data = []
        for _, row in df.iterrows():
            business_data.append({
                'company_name': str(row.get('ShortName', '')),
                'business_text': str(row.get('MAINBUSSINESS', '')),
                'symbol': str(row.get('Symbol', '')),
                'end_date': str(row.get('EndDate', ''))
            })

        # 处理断点续传逻辑
        progress_file = None
        existing_results = []

        if not force_restart:
            # 从进度文件加载已处理的记录
            processed_ids, progress_file, existing_results = self._load_processed_records_from_progress()
            unprocessed_data = self._filter_unprocessed_data(business_data, processed_ids)

            if not unprocessed_data:
                logging.info("所有记录都已处理完成，无需重复分析")
                if progress_file:
                    # 将进度文件转换为最终结果文件
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    final_file = self.output_dir / f"api_analysis_results_{timestamp}.csv"

                    results_df = pd.DataFrame(existing_results)
                    results_df.to_csv(final_file, index=False, encoding='utf-8-sig')

                    # 生成报告
                    self._generate_api_report(existing_results, timestamp)

                    logging.info(f"所有数据已处理完成，最终结果保存为: {final_file}")
                    return str(final_file)
                else:
                    logging.warning("未找到结果文件")
                    return ""

            logging.info(f"发现 {len(unprocessed_data)} 条未处理记录，将继续处理")
            business_data = unprocessed_data
        else:
            logging.info("强制重新开始，将处理全部数据")
            # 删除旧的进度文件
            old_progress_files = list(self.output_dir.glob("api_analysis_progress_*.csv"))
            for old_file in old_progress_files:
                try:
                    old_file.unlink()
                    logging.info(f"删除旧进度文件: {old_file}")
                except Exception as e:
                    logging.warning(f"删除旧进度文件失败: {e}")

        # 执行批量分析（使用增量保存和去重优化）
        new_results, final_progress_file = self.analyze_batch_with_incremental_save(
            business_data, progress_file, enable_deduplication
        )

        # 生成最终结果文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_result_file = self.output_dir / f"api_analysis_results_{timestamp}.csv"

        # 合并所有结果（已有的 + 新的）
        if not force_restart and existing_results:
            # 移除重复的已有结果（因为new_results可能包含了所有结果）
            all_results = existing_results + [r for r in new_results if r not in existing_results]
        else:
            all_results = new_results

        # 保存最终结果
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(final_result_file, index=False, encoding='utf-8-sig')

        # 生成统计报告
        self._generate_api_report(all_results, timestamp)

        # 自动检测并合并基础分析结果
        self._auto_merge_with_basic_analysis(str(final_result_file))

        logging.info(f"增量分析完成！")
        logging.info(f"进度文件: {final_progress_file}")
        logging.info(f"最终结果: {final_result_file}")
        logging.info(f"本次新增分析: {len(business_data)} 条，总计: {len(all_results)} 条")

        return str(final_result_file)

    def get_processing_status(self, csv_file: str, sample_size: int = None) -> Dict[str, Any]:
        """获取处理状态信息"""
        # 读取原始数据
        df = pd.read_csv(csv_file, encoding='utf-8')
        total_records = len(df)

        # 如果指定了采样大小
        if sample_size and sample_size < total_records:
            target_records = sample_size
        else:
            target_records = total_records

        # 优先从进度文件加载已处理的记录
        processed_ids, progress_file, existing_results = self._load_processed_records_from_progress()

        # 如果没有进度文件，尝试从结果文件加载
        if not processed_ids:
            processed_ids = self._load_processed_records(csv_file)

        processed_count = len(processed_ids)

        # 计算进度
        progress_percentage = (processed_count / target_records * 100) if target_records > 0 else 0
        remaining_count = max(0, target_records - processed_count)

        status = {
            'total_records': total_records,
            'target_records': target_records,
            'processed_count': processed_count,
            'remaining_count': remaining_count,
            'progress_percentage': round(progress_percentage, 2),
            'is_complete': remaining_count == 0,
            'progress_file': progress_file,
            'latest_result_file': None,
            'last_batch_number': None
        }

        # 获取最后处理的批次号
        if existing_results:
            batch_numbers = [r.get('batch_number', 0) for r in existing_results if 'batch_number' in r]
            if batch_numbers:
                status['last_batch_number'] = max(batch_numbers)

        # 查找最新结果文件
        pattern = f"api_analysis_results_*.csv"
        result_files = list(self.output_dir.glob(pattern))
        if result_files:
            latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
            status['latest_result_file'] = str(latest_file)

        return status

    def analyze_csv_file(self, csv_file: str, sample_size: int = None,
                        incremental: bool = True, enable_deduplication: bool = True) -> str:
        """分析CSV文件

        Args:
            csv_file: CSV文件路径
            sample_size: 采样大小，None表示全部数据
            incremental: 是否启用增量处理（默认True）
            enable_deduplication: 是否启用去重优化（默认True）
        """
        if incremental:
            return self.analyze_csv_file_incremental(csv_file, sample_size, force_restart=False,
                                                   enable_deduplication=enable_deduplication)
        else:
            # 原有的完整处理逻辑
            logging.info(f"开始完整分析CSV文件: {csv_file}")

            # 读取数据
            df = pd.read_csv(csv_file, encoding='utf-8')
            logging.info(f"读取到 {len(df)} 条记录")

            # 采样（如果指定了sample_size且小于总数据量）
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)
                logging.info(f"随机采样 {sample_size} 条记录进行分析")
            else:
                logging.info(f"将分析全部 {len(df)} 条记录")

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
            csv_file_path = self.output_dir / f"api_analysis_results_{timestamp}.csv"
            results_df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

            # 生成统计报告
            self._generate_api_report(results, timestamp)

            # 自动检测并合并基础分析结果
            self._auto_merge_with_basic_analysis(str(csv_file_path))

            logging.info(f"API分析结果已保存: {csv_file_path}")
            return str(csv_file_path)
    
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

    def _auto_merge_with_basic_analysis(self, api_results_file: str):
        """自动检测并合并基础分析结果"""
        try:
            # 检查是否存在基础分析结果文件
            base_file = self.output_dir / 'data_with_tourism_flag.csv'

            if not base_file.exists():
                logging.info("未找到基础分析结果文件，跳过自动合并")
                return

            logging.info("检测到基础分析结果，开始自动合并...")

            # 导入BusinessAnalyzer类
            from .business_analyzer import BusinessAnalyzer

            # 创建BusinessAnalyzer实例并执行合并
            business_analyzer = BusinessAnalyzer(output_dir=str(self.output_dir))
            merged_file = business_analyzer.merge_api_results(
                base_file=str(base_file),
                api_results_file=api_results_file
            )

            logging.info(f"自动合并完成，合并文件: {merged_file}")
            print(f"🔗 自动合并完成！合并文件: {merged_file}")

        except Exception as e:
            logging.warning(f"自动合并失败: {e}")
            print(f"⚠️  自动合并失败: {e}")
            print("💡 您可以稍后手动运行合并功能")

    def print_processing_status(self, csv_file: str, sample_size: int = None):
        """打印处理状态信息"""
        status = self.get_processing_status(csv_file, sample_size)

        print("=" * 70)
        print("📊 API分析处理状态")
        print("=" * 70)
        print(f"📁 数据文件: {csv_file}")
        print(f"📈 总记录数: {status['total_records']:,}")
        print(f"🎯 目标处理数: {status['target_records']:,}")
        print(f"✅ 已处理数: {status['processed_count']:,}")
        print(f"⏳ 剩余数量: {status['remaining_count']:,}")
        print(f"📊 完成进度: {status['progress_percentage']:.1f}%")
        print(f"🏁 是否完成: {'是' if status['is_complete'] else '否'}")

        # 显示进度文件信息
        if status['progress_file']:
            print(f"💾 进度文件: {status['progress_file']}")
            if status['last_batch_number']:
                print(f"🔢 最后批次: 第 {status['last_batch_number']} 批")
        else:
            print("💾 进度文件: 暂无")

        # 显示最终结果文件
        if status['latest_result_file']:
            print(f"📄 最新结果: {status['latest_result_file']}")
        else:
            print("📄 最新结果: 暂无")

        # 显示进度条
        if status['target_records'] > 0:
            progress_bar_length = 40
            filled_length = int(progress_bar_length * status['processed_count'] / status['target_records'])
            bar = '█' * filled_length + '░' * (progress_bar_length - filled_length)
            print(f"📈 进度条: |{bar}| {status['progress_percentage']:.1f}%")

        print("=" * 70)

        return status

def main():
    """主函数 - 支持增量处理和状态查看"""
    import argparse

    parser = argparse.ArgumentParser(description='API智能分析工具 - 支持增量处理')
    parser.add_argument('csv_file', nargs='?', default='data/STK_LISTEDCOINFOANL.csv',
                       help='CSV数据文件路径')
    parser.add_argument('--action', choices=['analyze', 'status', 'continue'], default='analyze',
                       help='操作类型: analyze(分析), status(查看状态), continue(继续处理)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='采样大小，默认处理全部数据')
    parser.add_argument('--force-restart', action='store_true',
                       help='强制重新开始，忽略已处理的数据')
    parser.add_argument('--no-incremental', action='store_true',
                       help='禁用增量处理，每次都完整处理')
    parser.add_argument('--no-deduplication', action='store_true',
                       help='禁用去重优化，处理所有重复文本')

    args = parser.parse_args()

    analyzer = APIAnalyzer()

    if args.action == 'status':
        # 查看处理状态
        analyzer.print_processing_status(args.csv_file, args.sample_size)

    elif args.action == 'continue':
        # 继续处理未完成的数据
        print("🔄 继续处理未完成的数据...")
        enable_dedup = not args.no_deduplication
        if enable_dedup:
            print("📊 启用去重优化，相同业务文本只分析一次")
        else:
            print("⚠️  已禁用去重优化，将分析所有重复文本")

        result_file = analyzer.analyze_csv_file_incremental(
            args.csv_file,
            args.sample_size,
            force_restart=False,
            enable_deduplication=enable_dedup
        )
        print(f"✅ 处理完成，结果保存在: {result_file}")

    else:  # analyze
        # 执行分析
        enable_dedup = not args.no_deduplication
        if enable_dedup:
            print("📊 启用去重优化，相同业务文本只分析一次")
        else:
            print("⚠️  已禁用去重优化，将分析所有重复文本")

        if args.no_incremental:
            print("🚀 开始完整分析...")
            result_file = analyzer.analyze_csv_file(
                args.csv_file,
                args.sample_size,
                incremental=False,
                enable_deduplication=enable_dedup
            )
        else:
            print("🚀 开始增量分析...")
            result_file = analyzer.analyze_csv_file_incremental(
                args.csv_file,
                args.sample_size,
                force_restart=args.force_restart,
                enable_deduplication=enable_dedup
            )

        print(f"✅ API分析完成，结果保存在: {result_file}")

        # 显示最终状态
        print("\n📊 最终状态:")
        analyzer.print_processing_status(args.csv_file, args.sample_size)

if __name__ == "__main__":
    main()
