import os
import json
from docx import Document  # 用于处理Word文档
from PyPDF2 import PdfReader  # 用于处理PDF文件
import numpy as np
from sentence_transformers import SentenceTransformer  # 用于文本嵌入
from sklearn.metrics.pairwise import cosine_similarity  # 用于计算向量相似度
from ollama import Client  # 用于连接本地大模型
import time
from typing import List, Dict, Any  # 类型注解
import re
import requests
import asyncio  # 异步处理
from functools import lru_cache  # 缓存装饰器
import concurrent.futures  # 线程池
import hashlib  # 用于生成签名
import urllib.parse
import random
import aiohttp  # 异步HTTP请求
import uuid


class FileRAGSystem:
    def __init__(self):
        # 初始化API配置 - 这里填入实际的API密钥
        self.youdao_appid = "YOUR_YOUDAO_APPID"  # 网易有道翻译APPID
        self.youdao_key = "YOUR_YOUDAO_KEY"  # 网易有道翻译密钥
        self.zhipu_api_key = "YOUR_ZHIPU_API_KEY"  # 智谱API密钥
        self.zhipu_api_url = "https://open.bigmodel.cn/api/paas/v3/model-api/GLM-4-Flash/invoke"
        
        # 检查API配置并提供警告
        if self.youdao_appid == "YOUR_YOUDAO_APPID" or self.youdao_key == "YOUR_YOUDAO_KEY":
            print("警告：网易有道翻译API未配置，请注册并获取API密钥：")
            print("1. 访问 https://ai.youdao.com/")
            print("2. 注册开发者账号")
            print("3. 创建应用获取APPID和密钥")
            print("4. 将获取的APPID和密钥填入代码中")
        
        if self.zhipu_api_key == "YOUR_ZHIPU_API_KEY":
            print("警告：智谱API未配置，请先配置API密钥")
            print("1. 访问 https://open.bigmodel.cn/")
            print("2. 注册开发者账号")
            print("3. 创建应用获取API密钥")
            print("4. 将获取的API密钥填入代码中")

        # 初始化知识库和嵌入向量
        self.knowledge_base = []  # 存储文档内容
        self.embeddings = None  # 存储文档的向量表示

        # 文件处理配置 - 支持的文件类型及对应的处理函数
        self.supported_extensions = {
            '.txt': self._process_txt,
            '.docx': self._process_docx,
            '.pdf': self._process_pdf,
            '.json': self._process_json
        }

        # 提示词模板 - 用于RAG问答
        self.prompt_template = """基于以下上下文回答问题：

上下文：
{context}

问题：{question}

要求：
1. 如果上下文相关，优先基于上下文回答
2. 保持专业性和准确性
3. 避免编造不知道的信息"""

        # 翻译提示词模板 - 包含从中文到英文和从英文到中文的翻译提示词
        self.translate_templates = {
            'zh2en': """请将以下中文文本翻译成英文，要求：

1. 专业性和准确性：
   - 保持原文的专业术语和概念
   - 确保技术性内容的准确翻译
   - 保持专业领域的表达习惯

2. 语义一致性：
   - 保持原文的核心含义
   - 确保上下文逻辑连贯
   - 避免歧义和误解

3. 语言表达：
   - 使用地道的英文表达
   - 保持原文的语气和风格
   - 确保语法正确，表达流畅

4. 格式和结构：
   - 保持原文的段落结构
   - 保留重要的格式标记
   - 保持标点符号的规范使用

中文文本：
{text}""",

            'en2zh': """请将以下英文文本翻译成中文，要求：

1. 专业性和准确性：
   - 准确翻译专业术语和概念
   - 保持技术性内容的专业性
   - 符合中文专业领域的表达习惯

2. 语义一致性：
   - 保持原文的核心含义
   - 确保上下文逻辑连贯
   - 避免歧义和误解

3. 语言表达：
   - 使用地道的中文表达
   - 保持原文的语气和风格
   - 确保语法正确，表达流畅

4. 格式和结构：
   - 保持原文的段落结构
   - 保留重要的格式标记
   - 保持标点符号的规范使用

英文文本：
{text}"""
        }

        # 文本比较提示词模板 - 用于分析原文与润色后文本的质量
        self.compare_template = """请严格分析以下两段中文文本的质量：

原文：
{original}

润色后：
{translated}

请从以下几个方面进行严格分析：

1. 专业性和准确性：
   - 专业术语的使用是否准确
   - 技术性内容的表达是否专业
   - 是否符合专业领域的表达习惯

2. 语义一致性：
   - 是否保持了原文的核心含义
   - 是否存在语义偏差或错误
   - 是否有重复或冗余的内容

3. 语言表达：
   - 用词是否准确、专业
   - 是否存在语法错误
   - 是否符合中文表达习惯
   - 表达是否流畅自然

4. 逻辑连贯性：
   - 句子之间的逻辑关系是否合理
   - 是否存在逻辑跳跃或矛盾
   - 整体结构是否清晰

5. 改进建议：
   - 如果润色后的文本存在问题，请指出具体问题
   - 如果原文更好，请说明原因
   - 如果润色后的文本更好，请说明具体改进之处

请用中文回答，要求：
1. 分析必须客观、严谨
2. 对每个方面都要给出具体分析
3. 如果发现润色后的文本存在明显问题（如重复、语义错误等），必须明确指出
4. 最终结论必须基于以上分析得出"""

        # 创建线程池 - 用于并行处理任务
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        # 缓存配置 - 用于存储已处理的翻译和分析结果
        self.translation_cache = {}
        self.analysis_cache = {}

        # 模型实例 - 懒加载模式
        self._embedding_model = None  # 文本嵌入模型
        self._ollama_client = None  # 本地大模型客户端
        self._local_model = "llama3:8b"  # 使用的本地模型名称

    @property
    def embedding_model(self):
        """懒加载嵌入模型"""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._embedding_model

    @property
    def ollama_client(self):
        """懒加载Ollama客户端"""
        if self._ollama_client is None:
            self._ollama_client = Client(host='http://localhost:11434')
        return self._ollama_client

    @lru_cache(maxsize=100)
    def _process_txt(self, file_path: str) -> List[Dict[str, str]]:
        """处理txt文件 - 读取内容并返回结构化数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [{"text": content, "source": os.path.basename(file_path)}]

    @lru_cache(maxsize=100)
    def _process_docx(self, file_path: str) -> List[Dict[str, str]]:
        """处理docx文件 - 提取所有段落文本"""
        doc = Document(file_path)
        content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return [{"text": content, "source": os.path.basename(file_path)}]

    @lru_cache(maxsize=100)
    def _process_pdf(self, file_path: str) -> List[Dict[str, str]]:
        """处理pdf文件 - 提取所有页面的文本"""
        reader = PdfReader(file_path)
        content = ""
        for page in reader.pages:
            content += page.extract_text() + "\n"
        return [{"text": content, "source": os.path.basename(file_path)}]

    @lru_cache(maxsize=100)
    def _process_json(self, file_path: str) -> List[Dict[str, str]]:
        """处理json文件 - 支持不同的JSON结构"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data  # 假设列表已经是合适的格式
        elif isinstance(data, dict):
            return [{"text": str(data), "source": os.path.basename(file_path)}]
        else:
            return [{"text": str(data), "source": os.path.basename(file_path)}]

    def upload_file(self, file_path: str) -> bool:
        """上传并处理文件 - 添加到知识库并更新嵌入向量"""
        try:
            # 检查文件扩展名是否支持
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_extensions:
                print(f"❌ 不支持的文件格式: {file_ext}")
                return False

            # 处理文件内容
            documents = self.supported_extensions[file_ext](file_path)

            # 添加到知识库
            self.knowledge_base.extend(documents)

            # 异步更新向量 - 在后台线程中执行
            def update_embeddings():
                texts = [doc["text"] for doc in self.knowledge_base]
                self.embeddings = self.embedding_model.encode(texts)

            # 使用线程池异步处理向量更新
            self.executor.submit(update_embeddings)

            print(f"✅ 成功上传文件: {os.path.basename(file_path)}")
            print(f"📚 当前知识库文档数: {len(self.knowledge_base)}")
            return True

        except Exception as e:
            print(f"❌ 处理文件时出错: {str(e)}")
            return False

    def save_knowledge_base(self, output_path: str = None):
        """保存知识库到JSON文件 - 便于后续加载"""
        try:
            # 如果没有指定文件名，使用默认格式
            if output_path is None:
                output_path = f"knowledge_base_{time.strftime('%Y%m%d_%H%M%S')}.json"
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 优化JSON序列化 - 处理NumPy数组
            def default(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return str(obj)
            
            # 使用更高效的JSON序列化方式
            with open(output_path, 'w', encoding='utf-8') as f:
                # 使用separators参数减少文件大小
                json.dump(
                    self.knowledge_base, 
                    f, 
                    ensure_ascii=False, 
                    indent=None,  # 移除缩进以减小文件大小
                    separators=(',', ':'),  # 使用最小分隔符
                    default=default
                )
            
            print(f"✅ 知识库已保存到: {output_path}")
            return True
        except Exception as e:
            print(f"❌ 保存知识库时出错: {str(e)}")
            return False

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """检索最相关的文档片段 - 基于向量相似度"""
        if not self.knowledge_base:
            return []

        # 将查询转换为向量
        query_embedding = self.embedding_model.encode([query])
        # 计算查询向量与所有文档向量的相似度
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        # 获取相似度最高的top_k个文档索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        # 返回对应的文档文本
        return [self.knowledge_base[i]["text"] for i in top_indices]

    def ask(self, question: str) -> str:
        """基于知识库回答问题 - RAG方法"""
        if not self.knowledge_base:
            return "知识库为空，请先上传文件。"

        try:
            # 检索相关文档
            contexts = self.retrieve(question)
            # 构建包含上下文的提示词
            prompt = self.prompt_template.format(
                context="\n\n".join(contexts),
                question=question
            )

            # 使用本地大模型生成回答
            response = self.ollama_client.generate(
                model=self._local_model,
                prompt=prompt,
                stream=False,
                options={'temperature': 0.3}  # 低温度以获得更确定的回答
            )
            return response['response'].strip()
        except Exception as e:
            return f"回答问题时出错: {str(e)}"

    async def translate_with_youdao(self, text, from_lang='zh-CHS', to_lang='en'):
        """使用有道翻译API进行文本翻译 - 异步方法"""
        try:
            # 检查缓存，避免重复翻译
            cache_key = f"{from_lang}_{to_lang}_{text}"
            if cache_key in self.translation_cache:
                return self.translation_cache[cache_key]
                
            # 检查API配置
            if not self.youdao_key or not self.youdao_appid:
                raise ValueError("有道翻译API配置缺失，请检查环境变量")
                
            # 准备请求参数
            app_key = self.youdao_appid
            app_secret = self.youdao_key
            
            # 当前UTC时间戳
            curtime = str(int(time.time()))
            # 随机数，使用UUID
            salt = str(uuid.uuid1())
            
            # 根据输入文本长度处理签名计算
            if len(text) <= 20:
                input_text = text
            else:
                input_text = text[:10] + str(len(text)) + text[-10:]
                
            # 生成签名: sha256(应用ID+input+salt+curtime+应用密钥)
            sign_str = app_key + input_text + salt + curtime + app_secret
            sign = hashlib.sha256(sign_str.encode('utf-8')).hexdigest()
            
            # 构建请求数据
            data = {
                'q': text,
                'from': from_lang,
                'to': to_lang,
                'appKey': app_key,
                'salt': salt,
                'sign': sign,
                'signType': 'v3',
                'curtime': curtime
            }
            
            # 发送异步请求
            async with aiohttp.ClientSession() as session:
                # 增加超时时间到60秒
                async with session.post(
                    'https://openapi.youdao.com/api', 
                    data=data, 
                    timeout=60
                ) as response:
                    result = await response.json()
                    
            # 解析结果
            if result.get('errorCode') == '0':
                # 获取翻译结果
                translations = result.get('translation', [])
                if translations:
                    translated_text = translations[0]
                    # 存入缓存
                    self.translation_cache[cache_key] = translated_text
                    return translated_text
                else:
                    return f"有道翻译错误: 未返回翻译结果"
            else:
                # 处理错误响应
                error_code = result.get('errorCode', 'unknown')
                error_msg = {
                    '101': '缺少必填参数，请检查是否缺少appKey、salt、sign、curtime等参数',
                    '102': '不支持的语言类型',
                    '103': '翻译文本过长',
                    '104': '不支持的API类型',
                    '105': '不支持的签名类型',
                    '106': '无效的应用ID',
                    '107': '无效的IP地址',
                    '108': '无效的应用密钥',
                    '109': 'batchLog格式不正确',
                    '110': '无相关服务的有效实例',
                    '111': '开发者账号已经欠费',
                    '112': '请求频率受限',
                    '113': '服务器内部错误',
                    '114': '账户校验失败',
                    '201': '解密失败，可能为DES加密等级不够',
                    '202': '签名检验失败，请检查签名生成方法',
                    '203': '访问IP地址不在可访问IP列表',
                    '205': '请求的接口与应用的接口类型不一致',
                    '206': '因为时间戳太旧而被拒绝',
                    '207': '重放请求',
                    '301': '辞典查询失败',
                    '302': '翻译查询失败',
                    '303': '服务端的其它异常',
                    '304': '会话闲置太久超时',
                    '401': '账户已经欠费',
                    '402': 'offlinesdk不可用',
                    '411': '访问频率受限',
                    '412': '长请求过于频繁'
                }.get(error_code, f'未知错误（{error_code}）')
                
                # 根据错误代码提供更详细的说明
                error_details = ""
                if error_code == '202':
                    error_details = "，请检查appKey、appSecret配置是否正确，以及签名生成方法是否正确"
                elif error_code == '108':
                    error_details = "，请检查API密钥是否正确"
                elif error_code == '106':
                    error_details = "，请检查API ID是否正确"
                elif error_code == '112' or error_code == '411':
                    error_details = "，请检查API调用频率或稍后再试"
                elif error_code == '401':
                    error_details = "，请充值账户"
                
                return f"有道翻译错误: {error_code}，{error_msg}{error_details}"
                
        except aiohttp.ClientError as e:
            return f"有道翻译请求错误: {str(e)}"
        except asyncio.TimeoutError:
            return "有道翻译超时，请稍后重试"
        except Exception as e:
            return f"有道翻译未知错误: {str(e)}"

    async def translate_with_zhipu(self, text: str, from_lang: str = 'zh', to_lang: str = 'en', context_prompt: str = '') -> str:
        """使用智谱API进行翻译 - 带上下文的翻译"""
        if not self.zhipu_api_key:
            return "错误：智谱API未配置，请先配置API密钥"
        
        # 检查缓存
        cache_key = f"zhipu_{from_lang}_{to_lang}_{hash(text)}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            # 准备请求头
            headers = {
                "Authorization": f"Bearer {self.zhipu_api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建带上下文的翻译提示词
            if from_lang == 'zh' and to_lang == 'en':
                prompt = f"{context_prompt}请将以下中文文本翻译成英文，保持专业性和准确性：\n\n{text}"
            else:
                prompt = f"{context_prompt}请将以下英文文本翻译成中文，保持专业性和准确性：\n\n{text}"
            
            # 准备请求数据
            data = {
                "prompt": prompt,
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            # 异步发送请求
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: requests.post(self.zhipu_api_url, headers=headers, json=data)
            )
            
            # 检查响应状态
            if response.status_code != 200:
                return f"智谱API请求失败，状态码：{response.status_code}"
            
            # 处理响应数据
            result = response.json()
            if 'data' in result and 'choices' in result['data']:
                translated_text = result['data']['choices'][0]['content'].strip()
                # 保存到缓存
                self.translation_cache[cache_key] = translated_text
                return translated_text
            else:
                error_msg = result.get('msg', '未知错误')
                return f"智谱API错误: {error_msg}"
        except Exception as e:
            return f"智谱API调用错误: {str(e)}"

    async def mirror_polish(self, text: str, model: str = 'both', context: str = '') -> Dict[str, Any]:
        """镜式润色：中->英->中，并比较结果 - 实现中英互译润色"""
        # 初始化结果结构
        result = {
            'original': text,
            'intermediate': {},  # 中间英文翻译结果
            'final': {},         # 最终中文润色结果
            'comparison': {}     # 原文与润色结果比较
        }

        # 构建带有上下文的提示词，用于更专业的翻译
        context_prompt = f"""请参考以下相关文本进行翻译：

相关文本：
{context}

要求：
1. 保持专业术语的一致性
2. 参考相关文本的表达方式
3. 确保翻译的准确性和专业性

"""

        # 并行处理翻译任务
        tasks = []
        if model in ['youdao', 'both']:
            # 使用有道翻译
            tasks.append(self.translate_with_youdao(text, 'zh-CHS', 'en'))
        if model in ['zhipu', 'both']:
            # 使用智谱翻译
            tasks.append(self.translate_with_zhipu(text, 'zh', 'en', context_prompt))

        # 等待所有翻译任务完成
        translations = await asyncio.gather(*tasks)

        # 处理有道翻译结果
        if model in ['youdao', 'both']:
            result['intermediate']['youdao'] = translations[0]
            # 将英文翻译回中文
            result['final']['youdao'] = await self.translate_with_youdao(translations[0], 'en', 'zh-CHS')
            # 分析润色结果
            result['comparison']['youdao'] = await self.analyze_text(text, result['final']['youdao'], context)

        # 处理智谱翻译结果
        if model in ['zhipu', 'both']:
            idx = 1 if model == 'both' else 0
            result['intermediate']['zhipu'] = translations[idx]
            # 将英文翻译回中文
            result['final']['zhipu'] = await self.translate_with_zhipu(translations[idx], 'en', 'zh', context_prompt)
            # 分析润色结果
            result['comparison']['zhipu'] = await self.analyze_text(text, result['final']['zhipu'], context)

        return result

    async def analyze_text(self, original: str, translated: str, context: str = '') -> Dict[str, Any]:
        """使用智谱API分析文本质量 - 比较原文与润色后文本"""
        try:
            # 准备请求头
            headers = {
                "Authorization": f"Bearer {self.zhipu_api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建带有上下文的分析提示词
            context_prompt = f"""请参考以下相关文本进行分析：

相关文本：
{context}

"""
            
            # 构建详细的分析提示词，更明确地要求判断质量
            prompt = f"""{context_prompt}请严格分析以下两段中文文本的质量，并明确判断哪个文本更好：

原文：
{original}

润色后：
{translated}

请从以下几个方面进行严格分析：

1. 专业性和准确性：
   - 专业术语的使用是否准确
   - 技术性内容的表达是否专业
   - 是否符合专业领域的表达习惯
   - 是否与相关文本保持一致

2. 语义一致性：
   - 是否保持了原文的核心含义
   - 是否存在语义偏差或错误
   - 是否有重复或冗余的内容
   - 是否与相关文本的语义一致

3. 语言表达：
   - 用词是否准确、专业
   - 是否存在语法错误
   - 是否符合中文表达习惯
   - 表达是否流畅自然

4. 逻辑连贯性：
   - 句子之间的逻辑关系是否合理
   - 是否存在逻辑跳跃或矛盾
   - 整体结构是否清晰
   - 是否与相关文本的逻辑一致

5. 改进建议：
   - 如果润色后的文本存在问题，请指出具体问题
   - 如果原文更好，请说明原因
   - 如果润色后的文本更好，请说明具体改进之处
   - 参考相关文本，提出更专业的改进建议

最终评价：
1. 请明确给出评价结论：润色后的文本是否优于原文
2. 给出一个1-10的质量评分(10分最高)，分别为原文和润色后的文本评分
3. 具体指出哪一个更好，以及为什么

请用中文回答，要求：
1. 分析必须客观、严谨
2. 对每个方面都要给出具体分析
3. 如果发现润色后的文本存在明显问题（如重复、语义错误等），必须明确指出
4. 最终结论必须基于以上分析得出，并明确指出哪个文本更好"""
            
            # 准备请求数据
            data = {
                "prompt": prompt,
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            # 异步发送请求
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: requests.post(self.zhipu_api_url, headers=headers, json=data)
            )
            
            # 检查响应状态
            if response.status_code != 200:
                return {
                    'analysis': f"分析请求失败，状态码：{response.status_code}",
                    'better_version': 'original',
                    'suggested_text': original
                }
            
            # 处理响应结果
            result = response.json()
            if 'data' in result and 'choices' in result['data']:
                analysis = result['data']['choices'][0]['content'].strip()
                
                # 判断哪个版本更好 - 使用更精确的判断标准
                better_version = 'original'  # 默认为原文更好
                
                # 判断评分或明确陈述来确定更好的版本
                if '润色后的文本评分' in analysis and '原文评分' in analysis:
                    # 尝试从评分内容中提取得分
                    try:
                        # 使用正则表达式匹配分数
                        import re
                        original_score_match = re.search(r'原文评分[：:]\s*(\d+(?:\.\d+)?)', analysis)
                        translated_score_match = re.search(r'润色后的文本评分[：:]\s*(\d+(?:\.\d+)?)', analysis)
                        
                        if original_score_match and translated_score_match:
                            original_score = float(original_score_match.group(1))
                            translated_score = float(translated_score_match.group(1))
                            
                            # 如果润色后的分数更高，则设为更好的版本
                            if translated_score > original_score:
                                better_version = 'translated'
                    except:
                        # 如果解析评分失败，使用关键词判断
                        pass
                
                # 基于关键词判断
                if better_version == 'original':  # 如果评分方式未确定结果，使用关键词
                    if ('润色后的文本更好' in analysis or 
                        '润色后的版本更好' in analysis or 
                        '润色后的文本优于原文' in analysis or
                        '润色版本优于原文' in analysis):
                        better_version = 'translated'
                
                return {
                    'analysis': analysis,
                    'better_version': better_version,
                    'suggested_text': translated if better_version == 'translated' else original,
                    'scores': {
                        'original': original_score_match.group(1) if 'original_score_match' in locals() and original_score_match else 'N/A',
                        'translated': translated_score_match.group(1) if 'translated_score_match' in locals() and translated_score_match else 'N/A'
                    } if 'original_score_match' in locals() or 'translated_score_match' in locals() else {}
                }
            else:
                return {
                    'analysis': f"分析失败：{result.get('msg', '未知错误')}",
                    'better_version': 'original',
                    'suggested_text': original
                }
        except Exception as e:
            return {
                'analysis': f"分析过程出错: {str(e)}",
                'better_version': 'original',
                'suggested_text': original
            }

    async def polish_text(self, text: str, model: str = 'both') -> Dict[str, Any]:
        """润色文本，可选择使用单个模型或两个模型 - 主要润色入口方法"""
        try:
            # 设置超时时间（秒）
            timeout = 60
            
            # 使用RAG检索相关文本，作为翻译的上下文参考
            if self.knowledge_base:
                related_texts = self.retrieve(text, top_k=3)
                context = "\n\n".join(related_texts)
            else:
                context = ""

            # 使用asyncio.wait_for添加超时控制
            result = await asyncio.wait_for(
                self.mirror_polish(text, model, context),
                timeout=timeout
            )
            
            # 格式化最终结果
            final_result = {
                'original': text,
                'suggested': {},
                'analysis': {}
            }
            
            # 添加有道翻译结果
            if 'youdao' in result['final']:
                final_result['suggested']['youdao'] = result['final']['youdao']
                final_result['analysis']['youdao'] = result['comparison']['youdao']['analysis']
            
            # 添加智谱翻译结果
            if 'zhipu' in result['final']:
                final_result['suggested']['zhipu'] = result['final']['zhipu']
                final_result['analysis']['zhipu'] = result['comparison']['zhipu']['analysis']
            
            return final_result
            
        except asyncio.TimeoutError:
            print("❌ 润色操作超时")
            return {
                'original': text,
                'suggested': {},
                'analysis': {},
                'error': '润色操作超时，请稍后重试'
            }
        except Exception as e:
            print(f"❌ 润色过程中出错: {str(e)}")
            return {
                'original': text,
                'suggested': {},
                'analysis': {},
                'error': f'润色过程中出错: {str(e)}'
            }


def main():
    """主函数 - 命令行界面入口"""
    # 初始化RAG系统
    rag_system = FileRAGSystem()

    # 显示欢迎信息
    print("📚 RAG文件问答系统")
    print("=" * 50)
    print("支持的文件格式: .txt, .docx, .pdf, .json")
    print("输入 'exit' 退出")
    print("输入 'save' 保存知识库")
    print("输入 'list' 查看已上传文件")
    print("=" * 50)

    # 命令行交互循环
    while True:
        command = input("\n请输入文件路径或命令：")

        # 退出系统
        if command.lower() in ['exit', 'quit']:
            break

        # 保存知识库
        elif command.lower() == 'save':
            filename = input("请输入保存文件名（直接回车使用默认名称）：")
            rag_system.save_knowledge_base(filename if filename.strip() else None)

        # 列出已上传文件
        elif command.lower() == 'list':
            if not rag_system.knowledge_base:
                print("📝 知识库为空")
            else:
                print("\n📚 已上传文件：")
                for doc in rag_system.knowledge_base:
                    print(f"- {doc['source']}")

        # 上传文件到知识库
        elif os.path.isfile(command):
            rag_system.upload_file(command)

        # 无效命令
        else:
            print("❌ 无效的命令或文件路径")

    # 退出前保存知识库
    if rag_system.knowledge_base:
        save = input("\n是否保存知识库？(y/n): ")
        if save.lower() == 'y':
            rag_system.save_knowledge_base()


# 当直接运行脚本时执行主函数
if __name__ == '__main__':
    main()
