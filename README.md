# 主营业务分词与旅游行业分析工具

基于中文分词技术的上市公司主营业务分析工具，自动识别旅游相关企业并进行行业分类。

## 功能特性

- **中文分词分析**: 使用jieba对主营业务文本进行分词，生成词频统计
- **旅游行业识别**: 基于关键词库自动识别旅游相关企业
- **数据标注**: 为原始数据添加旅游行业标识列
- **API智能分析**: 集成硅基流动API进行批量智能分析
- **可视化报告**: 生成分析图表和详细报告

## 项目结构

```
├── main.py                    # 主程序入口
├── src/                       # 源代码目录
│   ├── business_analyzer.py   # 核心分析模块
│   └── api_analyzer.py        # API智能分析模块
├── config/                    # 配置文件目录
│   ├── stopwords.txt          # 停用词配置
│   ├── tourism_keywords.txt   # 旅游关键词配置
│   └── api_config.json.template # API配置模板
├── data/                      # 数据文件目录
├── output/                    # 输出文件目录
├── logs/                      # 日志文件目录
└── requirements.txt           # 依赖包列表
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将CSV数据文件放在 `data/` 目录下，确保包含以下列：
- `Symbol`: 股票代码
- `ShortName`: 公司简称
- `EndDate`: 结束日期
- `MAINBUSSINESS`: 主营业务描述

### 3. 运行分析

```bash
# 基础分析（分词+旅游识别）
python main.py data/your_file.csv

# API智能分析（需要API密钥）
python main.py data/your_file.csv --mode api --sample-size 100

# 运行两种分析
python main.py data/your_file.csv --mode both
```

## 配置说明

### 停用词配置 (config/stopwords.txt)
每行一个停用词，用于过滤无意义词汇：
```
等
及
和
业务
经营
```

### 旅游关键词配置 (config/tourism_keywords.txt)
格式：`分类:关键词1,关键词2,关键词3`
```
酒店住宿:酒店,宾馆,旅馆,客栈,度假村
餐饮服务:餐饮,餐厅,饭店,食品,饮料
旅游服务:旅游,旅行,景区,景点,导游
```

### API配置 (config/api_config.json)
复制模板文件并填入API密钥：
```json
{
  "api_key": "your_siliconflow_api_key",
  "model": "deepseek-ai/DeepSeek-V3",
  "batch_size": 8,
  "max_requests_per_minute": 20
}
```

## 输出文件

### 基础分析输出
- `word_frequency.csv`: 词频统计结果（包含不重复词数）
- `data_with_tourism_flag.csv`: 添加旅游标识的原始数据
- `analysis_report.md`: 分析报告
- `analysis_charts.png`: 可视化图表

### API分析输出
- `api_analysis_results_*.csv`: API分析结果
- `api_analysis_report_*.md`: API分析报告

## 使用示例

### 命令行使用
```bash
# 分析指定文件
python main.py data/STK_LISTEDCOINFOANL.csv

# 指定输出目录
python main.py data/STK_LISTEDCOINFOANL.csv --output-dir results

# API分析前50条记录
python main.py data/STK_LISTEDCOINFOANL.csv --mode api --sample-size 50
```

### 编程使用
```python
from src.business_analyzer import BusinessAnalyzer

# 初始化分析器
analyzer = BusinessAnalyzer()

# 运行完整分析
analyzer.run_complete_analysis('data/your_file.csv')
```

## API功能

使用硅基流动API进行智能分析需要：

1. 设置环境变量：
```bash
export SILICONFLOW_API_KEY='your_api_key'
```

2. 或创建配置文件 `config/api_config.json`

API分析采用批量处理，一次分析多条记录以降低成本和提高效率。

## 常见问题

**Q: 分词结果不准确？**
A: 编辑 `config/stopwords.txt` 添加更多停用词，或调整 `config/tourism_keywords.txt` 中的关键词。

**Q: 如何查看详细错误信息？**
A: 查看 `logs/` 目录下的日志文件。

**Q: API调用失败？**
A: 检查API密钥设置、网络连接和调用限制。

## 依赖包

主要依赖：pandas, jieba, matplotlib, seaborn, requests, wordcloud

详见 `requirements.txt` 文件。
