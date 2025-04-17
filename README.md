# 文本润色系统

这是一个基于RAG（检索增强生成）的文本润色系统，支持文档上传、文本润色和知识库管理功能。系统集成了有道翻译和智谱AI API，能够提供高质量的文本润色服务。

## 项目地址

本项目已上传至GitHub，您可以通过以下方式访问：

1. 直接访问项目地址：
   ```
   https://github.com/[用户名]/text-polishing-system
   ```

2. 使用Git克隆项目：
   ```bash
   git clone https://github.com/[用户名]/text-polishing-system.git
   ```

3. 下载ZIP包：
   - 访问项目页面
   - 点击右上角的"Code"按钮
   - 选择"Download ZIP"

## 项目结构

```
text-polishing-system/
├── app.py                 # 主应用程序
├── file_rag.py           # RAG系统核心代码
├── requirements.txt      # 项目依赖
├── .env.example         # 环境变量模板
├── .gitignore           # Git忽略文件
├── README.md            # 项目说明文档
├── uploads/             # 上传文件目录
├── polished/            # 润色后文件目录
└── knowledge_base/      # 知识库目录
```

## 功能特点

### 1. 文档处理
- 支持多种文件格式：txt、docx、pdf
- 支持批量文档上传和处理
- 自动提取文档内容并进行润色
- 润色结果保存为独立文件，包含时间戳

### 2. 文本润色
- 支持有道翻译和智谱AI双引擎润色
- 提供中英文互译功能
- 支持单引擎或双引擎对比润色
- 自动分析润色结果质量

### 3. 知识库管理
- 支持专业文档上传和管理
- 基于RAG技术检索相关文本
- 利用知识库提升润色质量
- 支持知识库导出和备份

### 4. 系统特性
- 异步处理提高响应速度
- 支持大文件上传（最大16MB）
- 完善的错误处理和日志记录
- 友好的Web界面操作

## 环境要求

- Python 3.8+
- 操作系统：Windows/Linux/MacOS
- 内存：建议4GB以上
- 硬盘空间：建议1GB以上可用空间

## 安装步骤

### 1. 克隆项目
```bash
git clone [项目地址]
cd [项目目录]
```

### 2. 创建虚拟环境（推荐）
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/MacOS
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置环境变量
1. 复制环境变量模板：
```bash
cp .env.example .env
```

2. 编辑.env文件，配置以下参数：
- YOUDAO_APPID：有道翻译API的AppID
- YOUDAO_KEY：有道翻译API的密钥
- ZHIPU_API_KEY：智谱AI API的密钥
- 其他配置参数（可选）

### 5. 创建必要目录
```bash
mkdir uploads polished knowledge_base
```

## 使用方法

### 1. 启动服务器
```bash
python app.py
```
服务器将在 http://localhost:5000 启动

### 2. 文档上传
1. 点击"上传文档"按钮
2. 选择要上传的文件（支持txt、docx、pdf格式）
3. 选择润色模型（有道翻译、智谱AI或两者）
4. 点击"开始润色"按钮

### 3. 文本润色
1. 在文本框中输入或粘贴要润色的文本
2. 选择润色模型
3. 点击"润色"按钮
4. 查看润色结果和分析报告

### 4. 知识库管理
1. 上传专业文档到知识库
2. 系统自动提取文档内容
3. 润色时自动检索相关文本
4. 可随时导出知识库

## 配置说明

### API配置
- YOUDAO_APPID：有道翻译API的AppID
- YOUDAO_KEY：有道翻译API的密钥
- ZHIPU_API_KEY：智谱AI API的密钥

### 文件配置
- UPLOAD_FOLDER：上传文件存储目录
- POLISHED_FOLDER：润色后文件存储目录
- KNOWLEDGE_BASE_FOLDER：知识库文件存储目录
- MAX_CONTENT_LENGTH：最大文件大小（默认16MB）

### 服务器配置
- HOST：服务器主机地址
- PORT：服务器端口
- DEBUG：调试模式开关

## 注意事项

### 1. API使用
- 请确保API密钥配置正确
- 注意API调用频率限制
- 建议使用付费API以获得更好的服务

### 2. 文件处理
- 上传文件大小限制为16MB
- 支持的文件格式：txt、docx、pdf
- 文件名建议使用英文，避免中文

### 3. 知识库管理
- 建议使用专业文档构建知识库
- 定期更新知识库内容
- 注意知识库文件大小

### 4. 系统维护
- 定期清理临时文件
- 备份重要数据
- 检查日志文件

## 常见问题

### 1. 润色效果不理想
- 检查知识库内容是否相关
- 尝试使用不同的润色模型
- 确保原文表达清晰

### 2. 上传文件失败
- 检查文件大小是否超限
- 确认文件格式是否支持
- 检查网络连接

### 3. API调用失败
- 检查API密钥是否正确
- 确认API服务是否正常
- 检查网络连接

## 技术支持

如有问题，请通过以下方式获取支持：
- 在GitHub上提交Issue
- 发送邮件至[邮箱地址]
- 查看项目文档

## 贡献指南

欢迎贡献代码或提出建议：

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个Pull Request

## 更新日志

### v1.0.0 (2024-03-20)
- 初始版本发布
- 支持基本文档润色功能
- 集成有道翻译和智谱AI API
- 实现知识库管理功能

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件 