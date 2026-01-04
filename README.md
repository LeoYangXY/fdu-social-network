# 社交网络分析平台 - 用户手册

## 项目概述

这是一个基于Flask的Web应用，用于分析社交网络数据。系统能够从Facebook等社交网络数据中识别意见领袖、检测社区结构，并提供可视化分析结果。

## 文件结构说明

```
SOCIAL_NETWORK/
├── cache/                          # 缓存目录
│   ├── 8c7ec6ded0f8b475c0c295a657805fd4.json  # 缓存文件示例
├── data/                       # 数据集目录
│       └── facebook_combined.txt   # Facebook数据集文件
├── results/                        # 分析结果目录
├── src/                           # 源代码目录
│   └── backend.py                 # 后端主程序
├── static/                        # 静态资源目录
├── templates/                     # HTML模板目录
│   └── index.html                 # 前端主页面
├── environment.yml                # Conda环境配置文件
├── README.md                      # 说明文档（本文件）
└── requirements.txt               # Python依赖包列表
```

## 安装与配置

### 方式一：使用Conda（推荐）

```bash
# 创建Conda环境
conda env create -f environment.yml

# 安装依赖包
pip install -r requirements.txt

```

## 数据准备

### Facebook数据集获取

1. 访问 [Stanford SNAP数据集网站](http://snap.stanford.edu/data/ego-Facebook.html)
2. 下载 `facebook_combined.txt.gz` 文件
3. 解压后得到 `facebook_combined.txt`
4. 将文件放置在 `/data/` 目录下

### 数据集信息
- **文件名**: facebook_combined.txt
- **节点数**: 4039个
- **边数**: 88234条
- **格式**: 每行包含两个节点ID，表示连接关系

## 启动应用

### 基本启动
```bash
# 激活环境
conda activate social-network

# 启动应用
python src/backend.py
```

### 启动参数说明
启动后控制台会显示以下信息：
- 服务地址：http://localhost:5000
- 项目根目录路径
- 模板目录路径
- 静态资源目录路径

### 验证服务是否正常运行
打开浏览器访问：http://localhost:5000
或检查API状态：http://localhost:5000/api/status

## 使用指南

### 1. Web界面使用
1. 打开浏览器访问 http://localhost:5000
2. 选择要分析的数据集（目前支持Facebook）
3. 设置要分析的意见领袖数量（默认10个）
4. 点击"开始分析"按钮
5. 等待分析完成，查看结果

### 2. 分析功能说明

#### 意见领袖识别
系统会计算4种中心性指标来识别意见领袖：
1. **度中心性**：直接连接的数量
2. **介数中心性**：控制信息流的能力
3. **接近中心性**：到网络中其他节点的平均距离
4. **特征向量中心性**：连接质量的重要性

#### 社区检测
使用Louvain算法自动检测网络中的社区结构：
- 计算模块度（Modularity）评估社区划分质量
- 统计每个社区的节点数量和内部密度
- 使用不同颜色区分不同社区

#### 可视化输出
系统会生成5种可视化图表：
1. 度分布直方图
2. 社区规模分布图
3. 中心性指标对比图
4. 意见领袖与社区综合图
5. 网络结构概览图（小网络适用）

## 配置说明

### 1. 修改分析参数
编辑 `src/backend.py` 文件中的以下部分：

```python
# 数据集路径配置
DATASETS = {
    "facebook": os.path.join(PROJECT_ROOT, "cache", "data", "facebook_combined.txt"),
}

# 缓存配置
CACHE_TTL_HOURS = 24  # 缓存有效期（小时）
MAX_WORKERS = 2       # 最大并发线程数
```

### 2. 添加新数据集
1. 将数据文件放在 `/data/` 目录
2. 在 `src/backend.py` 的 `DATASETS` 字典中添加新条目
3. 重启应用

## 性能优化

### 对于大网络的处理
当节点数超过5000时，系统会自动启用优化策略：
1. 中心性计算采用采样方法
2. 可视化时进行智能节点采样
3. 使用近似算法计算路径长度

### 缓存机制
- 分析结果自动缓存24小时
- 相同参数的分析直接从缓存读取
- 可手动清理缓存：访问 `/api/cache/clear`

## 输出说明

### 1. Web界面输出
- 网络全局指标（节点数、边数、密度等）
- 意见领袖排名表
- 社区统计信息
- 可视化图表

### 2. 缓存文件
- 位置：`cache/` 目录
- 格式：JSON文件
- 命名：MD5哈希值（基于参数生成）

### 3. 日志信息
- 服务启动信息
- 分析进度状态
- 错误和警告信息
- 缓存命中/未命中记录

## 扩展开发

### 添加新的分析指标
1. 在 `SocialNetworkAnalyzer` 类中添加新方法
2. 更新分析流程调用新方法
3. 在前端页面显示新指标

### 支持新数据格式
1. 修改 `load_graph` 方法支持新格式
2. 更新数据预处理逻辑
3. 添加相应的数据验证

## 版本信息

- **当前版本**：v1.0.0
- **Python版本**：3.8+
- **主要依赖包**：
  - Flask 2.0+
  - NetworkX 2.6+
  - Matplotlib 3.4+
  - Pandas 1.3+

---

**开始使用**：
1. 安装依赖包
2. 准备数据文件
3. 启动后端服务
4. 访问Web界面进行分析

**重要提示**：首次使用请确保已完成数据准备步骤！