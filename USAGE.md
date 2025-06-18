# 使用说明

## 快速开始

### 1. 基础分析
```bash
python main.py data/STK_LISTEDCOINFOANL.csv
```

### 2. API智能分析
```bash
# 设置API密钥
export SILICONFLOW_API_KEY='your_api_key'

# 分析前50条记录
python main.py data/STK_LISTEDCOINFOANL.csv --mode api --sample-size 50
```

### 3. 完整分析
```bash
python main.py data/STK_LISTEDCOINFOANL.csv --mode both
```

## 输出文件说明

### 基础分析输出 (output/)
- **word_frequency.csv**: 词频统计，包含排名、词汇、频次、占比
- **word_statistics.txt**: 分词统计摘要（总词数、不重复词数等）
- **data_with_tourism_flag.csv**: 原始数据+IS_TOURISM标识列
- **analysis_report.md**: 分析报告
- **analysis_charts.png**: 可视化图表

### API分析输出 (output/)
- **api_analysis_results_*.csv**: API智能分析结果
- **api_analysis_report_*.md**: API分析报告

## 配置文件

### config/stopwords.txt
停用词配置，每行一个词：
```
等
及
和
业务
经营
```

### config/tourism_keywords.txt
旅游关键词分类，格式：`分类:关键词1,关键词2`
```
酒店住宿:酒店,宾馆,旅馆,客栈,度假村
餐饮服务:餐饮,餐厅,饭店,食品,饮料
```

### config/api_config.json
API配置（从模板复制）：
```json
{
  "api_key": "your_api_key",
  "model": "deepseek-ai/DeepSeek-V3",
  "batch_size": 8,
  "max_requests_per_minute": 20
}
```

## 分析结果示例

### 词频统计结果
- 总词数: 324,726
- 不重复词数: 6,800
- 高频词汇: 研发(17,410次)、产品(10,623次)、设计(5,444次)

### 旅游行业识别
- 旅游相关记录: 7,468条 (15.8%)
- 主要分类: 交通运输、零售商业、餐饮服务、娱乐休闲等

## 常见问题

**Q: 如何修改旅游关键词？**
A: 编辑 `config/tourism_keywords.txt` 文件，添加或修改关键词分类。

**Q: 分词结果不准确？**
A: 在 `config/stopwords.txt` 中添加更多停用词。

**Q: API调用失败？**
A: 检查API密钥设置和网络连接，查看 `logs/api_analysis.log` 获取详细错误信息。
