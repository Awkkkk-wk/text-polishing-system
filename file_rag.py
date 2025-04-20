import os
import json
from docx import Document
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ollama import Client
import time
from typing import List, Dict, Any
import re
import requests
import asyncio
from functools import lru_cache
import concurrent.futures
import hashlib
import urllib.parse
import random
import aiohttp


class FileRAGSystem:
    def __init__(self):
        # 初始化API配置
        self.youdao_appid = "1f5bd96a38c3b8b2"  # 网易有道翻译APPID
        self.youdao_key = "EwPd4WD9wnhTsBLWZffR5RPXBtLiXWNy"  # 网易有道翻译密钥
        self.zhipu_api_key = "0e9517beeddfa990fa4535cf5a586d51.vexg1jTHVdU2b4I5"  # 智谱API密钥
        self.zhipu_api_url = "https://open.bigmodel.cn/api/paas/v3/model-api/GLM-4-flash/invoke"
        
        # 检查API配置
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

        # 初始化知识库
        self.knowledge_base = []
        self.embeddings = None

        # 文件处理配置
        self.supported_extensions = {
            '.txt': self._process_txt,
            '.docx': self._process_docx,
            '.pdf': self._process_pdf,
            '.json': self._process_json
        }

        # 提示词模板
        self.prompt_template = """基于以下上下文回答问题：

上下文：
{context}

问题：{question}

要求：
1. 如果上下文相关，优先基于上下文回答
2. 保持专业性和准确性
3. 避免编造不知道的信息"""

        # 翻译提示词模板
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

        # 文本比较提示词模板
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

        # 创建线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        # 缓存配置
        self.translation_cache = {}
        self.analysis_cache = {}

        # 模型实例
        self._embedding_model = None
        self._ollama_client = None
        self._local_model = "llama3:8b"

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._embedding_model

    @property
    def ollama_client(self):
        if self._ollama_client is None:
            self._ollama_client = Client(host='http://localhost:11434')
        return self._ollama_client

    @lru_cache(maxsize=100)
    def _process_txt(self, file_path: str) -> List[Dict[str, str]]:
        """处理txt文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [{"text": content, "source": os.path.basename(file_path)}]

    @lru_cache(maxsize=100)
    def _process_docx(self, file_path: str) -> List[Dict[str, str]]:
        """处理docx文件"""
        doc = Document(file_path)
        content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return [{"text": content, "source": os.path.basename(file_path)}]

    @lru_cache(maxsize=100)
    def _process_pdf(self, file_path: str) -> List[Dict[str, str]]:
        """处理pdf文件"""
        reader = PdfReader(file_path)
        content = ""
        for page in reader.pages:
            content += page.extract_text() + "\n"
        return [{"text": content, "source": os.path.basename(file_path)}]

    @lru_cache(maxsize=100)
    def _process_json(self, file_path: str) -> List[Dict[str, str]]:
        """处理json文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [{"text": str(data), "source": os.path.basename(file_path)}]
        else:
            return [{"text": str(data), "source": os.path.basename(file_path)}]

    def upload_file(self, file_path: str) -> bool:
        """上传并处理文件"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_extensions:
                print(f"❌ 不支持的文件格式: {file_ext}")
                return False

            # 处理文件
            documents = self.supported_extensions[file_ext](file_path)

            # 添加到知识库
            self.knowledge_base.extend(documents)

            # 异步更新向量
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
        """保存知识库到JSON文件"""
        try:
            # 如果没有指定文件名，使用默认格式
            if output_path is None:
                output_path = f"knowledge_base_{time.strftime('%Y%m%d_%H%M%S')}.json"
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 优化JSON序列化
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
        """检索最相关的文档片段"""
        if not self.knowledge_base:
            return []

        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.knowledge_base[i]["text"] for i in top_indices]

    def ask(self, question: str) -> str:
        """基于知识库回答问题"""
        if not self.knowledge_base:
            return "知识库为空，请先上传文件。"

        try:
            contexts = self.retrieve(question)
            prompt = self.prompt_template.format(
                context="\n\n".join(contexts),
                question=question
            )

            response = self.ollama_client.generate(
                model=self._local_model,
                prompt=prompt,
                stream=False,
                options={'temperature': 0.3}
            )
            return response['response'].strip()
        except Exception as e:
            return f"回答问题时出错: {str(e)}"

    async def translate_with_youdao(self, text: str, from_lang: str = 'zh-CHS', to_lang: str = 'en') -> str:
        """使用网易有道翻译API进行翻译"""
        if self.youdao_appid == "YOUR_YOUDAO_APPID" or self.youdao_key == "YOUR_YOUDAO_KEY":
            return "错误：网易有道翻译API未配置，请先配置API密钥"
        
        cache_key = f"youdao_{from_lang}_{to_lang}_{hash(text)}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            salt = str(random.randint(32768, 65536))
            sign = self.youdao_appid + text + salt + self.youdao_key
            sign = hashlib.md5(sign.encode()).hexdigest()
            
            data = {
                'q': text,
                'from': from_lang,
                'to': to_lang,
                'appKey': self.youdao_appid,
                'salt': salt,
                'sign': sign
            }
            
            # 设置超时时间
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post('https://openapi.youdao.com/api', data=data) as response:
                    if response.status != 200:
                        return f"有道翻译API请求失败，状态码：{response.status}"
                    
                    result = await response.json()
                    if 'translation' in result:
                        translated_text = result['translation'][0]
                        self.translation_cache[cache_key] = translated_text
                        return translated_text
                    else:
                        error_msg = result.get('errorCode', '未知错误')
                        return f"有道翻译错误: {error_msg}，请检查API配置是否正确"
                    
        except asyncio.TimeoutError:
            return "有道翻译请求超时，请稍后重试"
        except aiohttp.ClientError as e:
            return f"有道翻译网络错误: {str(e)}"
        except Exception as e:
            return f"有道翻译处理错误: {str(e)}"

    async def translate_with_zhipu(self, text: str, from_lang: str = 'zh', to_lang: str = 'en', context_prompt: str = '') -> str:
        """使用智谱API进行翻译"""
        if not self.zhipu_api_key:
            return "错误：智谱API未配置，请先配置API密钥"
        
        cache_key = f"zhipu_{from_lang}_{to_lang}_{hash(text)}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            headers = {
                "Authorization": f"Bearer {self.zhipu_api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建翻译提示词
            if from_lang == 'zh' and to_lang == 'en':
                prompt = f"{context_prompt}请将以下中文文本翻译成英文，保持专业性和准确性：\n\n{text}"
            else:
                prompt = f"{context_prompt}请将以下英文文本翻译成中文，保持专业性和准确性：\n\n{text}"
            
            data = {
                "prompt": prompt,
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: requests.post(self.zhipu_api_url, headers=headers, json=data)
            )
            
            if response.status_code != 200:
                return f"智谱API请求失败，状态码：{response.status_code}"
            
            result = response.json()
            if 'data' in result and 'choices' in result['data']:
                translated_text = result['data']['choices'][0]['content'].strip()
                self.translation_cache[cache_key] = translated_text
                return translated_text
            else:
                error_msg = result.get('msg', '未知错误')
                return f"智谱API错误: {error_msg}"
        except Exception as e:
            return f"智谱API调用错误: {str(e)}"

    async def mirror_polish(self, text: str, model: str = 'both', context: str = '') -> Dict[str, Any]:
        """镜式润色：中->英->中，并比较结果"""
        result = {
            'original': text,
            'intermediate': {},
            'final': {},
            'comparison': {}
        }

        # 构建带有上下文的提示词
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

        # 处理翻译结果
        if model in ['youdao', 'both']:
            result['intermediate']['youdao'] = translations[0]
            # 将英文翻译回中文
            result['final']['youdao'] = await self.translate_with_youdao(translations[0], 'en', 'zh-CHS')
            # 分析润色结果
            result['comparison']['youdao'] = await self.analyze_text(text, result['final']['youdao'], context)

        if model in ['zhipu', 'both']:
            idx = 1 if model == 'both' else 0
            result['intermediate']['zhipu'] = translations[idx]
            # 将英文翻译回中文
            result['final']['zhipu'] = await self.translate_with_zhipu(translations[idx], 'en', 'zh', context_prompt)
            # 分析润色结果
            result['comparison']['zhipu'] = await self.analyze_text(text, result['final']['zhipu'], context)

        return result

    async def analyze_text(self, original: str, translated: str, context: str = '') -> Dict[str, Any]:
        """使用智谱API分析文本质量"""
        try:
            headers = {
                "Authorization": f"Bearer {self.zhipu_api_key}",
                "Content-Type": "application/json"
            }
            
            context_prompt = f"""请参考以下相关文本进行分析：

相关文本：
{context}

"""
            
            prompt = f"""{context_prompt}请严格分析以下两段中文文本的质量：

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

请用中文回答，要求：
1. 分析必须客观、严谨
2. 对每个方面都要给出具体分析
3. 如果发现润色后的文本存在明显问题（如重复、语义错误等），必须明确指出
4. 最终结论必须基于以上分析得出"""
            
            data = {
                "prompt": prompt,
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: requests.post(self.zhipu_api_url, headers=headers, json=data)
            )
            
            if response.status_code != 200:
                return {
                    'analysis': f"分析请求失败，状态码：{response.status_code}",
                    'better_version': 'original',
                    'suggested_text': original
                }
            
            result = response.json()
            if 'data' in result and 'choices' in result['data']:
                analysis = result['data']['choices'][0]['content'].strip()
                better_version = 'translated' if (
                    '润色后的文本更好' in analysis and 
                    '专业性和准确性' in analysis and 
                    '语义一致性' in analysis and 
                    '语言表达' in analysis and
                    '逻辑连贯性' in analysis and
                    '没有发现明显问题' in analysis
                ) else 'original'
                return {
                    'analysis': analysis,
                    'better_version': better_version,
                    'suggested_text': translated if better_version == 'translated' else original
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
        """润色文本，可选择使用单个模型或两个模型"""
        try:
            # 设置超时时间（秒）
            timeout = 60
            
            # 使用RAG检索相关文本
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
            
            final_result = {
                'original': text,
                'suggested': {},
                'analysis': {}
            }
            
            if 'youdao' in result['final']:
                final_result['suggested']['youdao'] = result['final']['youdao']
                final_result['analysis']['youdao'] = result['comparison']['youdao']['analysis']
            
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
    rag_system = FileRAGSystem()

    print("📚 RAG文件问答系统")
    print("=" * 50)
    print("支持的文件格式: .txt, .docx, .pdf, .json")
    print("输入 'exit' 退出")
    print("输入 'save' 保存知识库")
    print("输入 'list' 查看已上传文件")
    print("=" * 50)

    while True:
        command = input("\n请输入文件路径或命令：")

        if command.lower() in ['exit', 'quit']:
            break

        elif command.lower() == 'save':
            filename = input("请输入保存文件名（直接回车使用默认名称）：")
            rag_system.save_knowledge_base(filename if filename.strip() else None)

        elif command.lower() == 'list':
            if not rag_system.knowledge_base:
                print("📝 知识库为空")
            else:
                print("\n📚 已上传文件：")
                for doc in rag_system.knowledge_base:
                    print(f"- {doc['source']}")

        elif os.path.isfile(command):
            rag_system.upload_file(command)

        else:
            print("❌ 无效的命令或文件路径")

    # 退出前保存知识库
    if rag_system.knowledge_base:
        save = input("\n是否保存知识库？(y/n): ")
        if save.lower() == 'y':
            rag_system.save_knowledge_base()


if __name__ == '__main__':
    main()
