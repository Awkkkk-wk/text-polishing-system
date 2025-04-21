from flask import Flask, render_template, request, jsonify, send_file
from file_rag import FileRAGSystem  # 导入自定义的RAG系统
import os
import asyncio
from quart import Quart, render_template, request, jsonify, send_file  # Quart框架支持异步
from datetime import datetime
import time
import requests

# 创建Quart应用实例，而非Flask，以支持异步处理
app = Quart(__name__)

# 配置文件上传和处理的相关目录
app.config['UPLOAD_FOLDER'] = 'uploads'  # 上传文件临时存储目录
app.config['POLISHED_FOLDER'] = 'polished'  # 润色后文件存储目录
app.config['KNOWLEDGE_BASE_FOLDER'] = 'knowledge_base'  # 知识库文件存储目录
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 确保所需目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['POLISHED_FOLDER'], exist_ok=True)
os.makedirs(app.config['KNOWLEDGE_BASE_FOLDER'], exist_ok=True)

# 使用单例模式初始化RAG系统，确保全应用共享同一实例
_rag_system = None

def get_rag_system():
    """获取RAG系统的单例实例"""
    global _rag_system
    if _rag_system is None:
        _rag_system = FileRAGSystem()
    return _rag_system

@app.route('/')
async def index():
    """主页路由 - 渲染前端界面"""
    return await render_template('index.html')

@app.route('/upload_kb', methods=['POST'])
async def upload_knowledge_base():
    """处理知识库文件上传的路由"""
    # 检查是否有文件被上传
    if 'file' not in (await request.files):
        return jsonify({'error': '没有文件被上传'})
    
    file = (await request.files)['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    if file:
        try:
            # 保存上传的文件到临时目录
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            await file.save(file_path)
            
            # 获取RAG系统实例
            rag_system = get_rag_system()
            
            # 异步处理文件上传到知识库
            def process_upload():
                return rag_system.upload_file(file_path)
            
            # 使用线程池处理文件上传，避免阻塞主线程
            success = await asyncio.get_event_loop().run_in_executor(
                None, process_upload
            )
            
            if success:
                return jsonify({'message': f'知识库文件 {file.filename} 上传成功'})
            else:
                return jsonify({'error': f'处理知识库文件 {file.filename} 时出错'})
                
        except Exception as e:
            return jsonify({'error': f'上传文件时出错: {str(e)}'})

@app.route('/save_kb', methods=['POST'])
async def save_knowledge_base():
    """保存知识库到JSON文件的路由"""
    form = await request.form
    filename = form.get('filename', '')
    rag_system = get_rag_system()
    
    # 处理文件名和路径
    if filename:
        # 如果用户没有指定.json扩展名，自动添加
        if not filename.lower().endswith('.json'):
            filename += '.json'
        output_path = os.path.join(app.config['KNOWLEDGE_BASE_FOLDER'], filename)
    else:
        # 使用默认文件名
        output_path = os.path.join(app.config['KNOWLEDGE_BASE_FOLDER'], f"knowledge_base_{time.strftime('%Y%m%d_%H%M%S')}.json")
    
    try:
        # 异步处理文件保存
        def process_save():
            return rag_system.save_knowledge_base(output_path)
        
        # 使用线程池处理文件保存，避免阻塞主线程
        success = await asyncio.get_event_loop().run_in_executor(
            None, process_save
        )
        
        if success:
            return jsonify({'message': f'知识库已保存为 {os.path.basename(output_path)}'})
        else:
            return jsonify({'error': '保存知识库时出错'})
            
    except Exception as e:
        return jsonify({'error': f'保存文件时出错: {str(e)}'})

@app.route('/polish', methods=['POST'])
async def polish_text():
    """文本润色处理路由 - 处理单段文本的润色请求"""
    form = await request.form
    text = form.get('text', '')  # 获取要润色的文本
    model = form.get('model', 'both')  # 获取使用的模型，默认为both（同时使用有道和智谱）
    
    # 检查文本是否为空
    if not text:
        return jsonify({'error': '文本不能为空'})
    
    try:
        # 增加超时时间到120秒
        timeout = 120
        
        # 获取RAG系统
        rag_system = get_rag_system()
        
        # 调用异步润色方法，并设置超时
        result = await asyncio.wait_for(
            rag_system.polish_text(text, model),
            timeout=timeout
        )
        
        # 检查是否有错误
        if 'error' in result:
            return jsonify({'error': result['error']})
            
        # 检查有道翻译结果，是否有API错误
        if 'youdao' in result.get('suggested', {}) and isinstance(result['suggested']['youdao'], str) and result['suggested']['youdao'].startswith('有道翻译错误'):
            return jsonify({
                'error': result['suggested']['youdao'],
                'original': str(result.get('original', '')),
                'suggested': {
                    'youdao': '',
                    'zhipu': str(result.get('suggested', {}).get('zhipu', ''))
                },
                'analysis': {
                    'youdao': '',
                    'zhipu': str(result.get('analysis', {}).get('zhipu', ''))
                }
            })
            
        # 确保所有值都是字符串类型，防止JSON序列化错误
        processed_result = {
            'original': str(result.get('original', '')),
            'suggested': {
                'youdao': str(result.get('suggested', {}).get('youdao', '')),
                'zhipu': str(result.get('suggested', {}).get('zhipu', ''))
            },
            'analysis': {
                'youdao': str(result.get('analysis', {}).get('youdao', '')),
                'zhipu': str(result.get('analysis', {}).get('zhipu', ''))
            }
        }
        return jsonify(processed_result)
        
    except asyncio.TimeoutError:
        return jsonify({'error': '润色操作超时，请稍后重试或尝试更短的文本'})
    except Exception as e:
        return jsonify({'error': f'润色过程中出错: {str(e)}'})

@app.route('/polish_doc', methods=['POST'])
async def polish_document():
    """文档润色处理路由 - 处理整个文档文件的润色"""
    # 检查是否有文件被上传
    if 'file' not in (await request.files):
        return jsonify({'error': '没有文件被上传'})
    
    file = (await request.files)['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    form = await request.form
    model = form.get('model', 'both')  # 获取使用的模型，默认为both
    
    try:
        # 保存上传的文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        await file.save(file_path)
        
        # 获取RAG系统
        rag_system = get_rag_system()
        
        # 处理文档 - 根据文件类型调用对应的处理方法
        documents = rag_system.supported_extensions[os.path.splitext(file.filename)[1].lower()](file_path)
        
        # 润色文档内容
        polished_content = []
        for doc in documents:
            # 对每个文档片段进行润色
            result = await rag_system.polish_text(doc['text'], model)
            
            # 添加原文
            polished_content.append(f"原文：\n{doc['text']}\n")
            
            # 当同时使用两种模型时，进行综合对比分析
            if model == 'both':
                # 构建增强的对比分析提示词
                comparison_prompt = f"""请详细对比分析以下两个润色版本，判断哪个更好：

原文：
{doc['text']}

有道翻译润色版本：
{result['suggested']['youdao']}

智谱API润色版本：
{result['suggested']['zhipu']}

请从以下几个方面进行详细对比分析：
1. 语义准确性：哪个版本更准确地保留了原文的核心含义和细节
2. 专业性：哪个版本对专业术语和概念的处理更准确
3. 语言流畅度：哪个版本的表达更自然流畅，更符合中文表达习惯
4. 结构与逻辑：哪个版本在结构和逻辑上更清晰连贯
5. 语法与措辞：哪个版本的语法更准确，词语选择更恰当

综合对比：
1. 明确指出哪个版本总体更优秀，给出具体理由
2. 分析两个版本各自的优缺点
3. 建议如何结合两个版本的优点得到最佳润色文本

请提供一个最佳润色版本，可以直接采用更好的那个版本，或结合两者的优点创建一个优化版本。

输出格式：
1. 分点对比分析（按上述5个维度）
2. 综合结论（明确指出哪个版本更好，或各有什么优点）
3. 最佳润色推荐
"""
                
                # 调用智谱API进行对比分析
                headers = {
                    "Authorization": f"Bearer {rag_system.zhipu_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "prompt": comparison_prompt,
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
                
                response = await asyncio.get_event_loop().run_in_executor(
                    rag_system.executor,
                    lambda: requests.post(rag_system.zhipu_api_url, headers=headers, json=data)
                )
                
                if response.status_code == 200:
                    analysis_result = response.json()
                    if 'data' in analysis_result and 'choices' in analysis_result['data']:
                        comparison_analysis = analysis_result['data']['choices'][0]['content'].strip()
                        
                        # 添加各自的润色结果
                        polished_content.append(f"\n有道翻译润色版本：\n{result['suggested']['youdao']}")
                        polished_content.append(f"\n智谱API润色版本：\n{result['suggested']['zhipu']}")
                        
                        # 添加详细的对比分析
                        polished_content.append(f"\n【润色版本对比分析】：\n{comparison_analysis}")
            else:
                # 单模型模式下，只显示该模型的结果
                if model == 'youdao':
                    polished_content.append(f"\n有道翻译润色：\n{result['suggested']['youdao']}")
                    polished_content.append(f"\n有道翻译分析：\n{result['analysis']['youdao']}")
                elif model == 'zhipu':
                    polished_content.append(f"\n智谱API润色：\n{result['suggested']['zhipu']}")
                    polished_content.append(f"\n智谱API分析：\n{result['analysis']['zhipu']}")
            
            # 添加分隔符
            polished_content.append("\n" + "="*50 + "\n")
        
        # 生成输出文件名，包含时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{os.path.splitext(file.filename)[0]}_polished_{timestamp}.txt"
        output_path = os.path.join(app.config['POLISHED_FOLDER'], output_filename)
        
        # 保存润色后的内容到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(polished_content))
        
        # 返回成功信息和文件名
        return jsonify({
            'message': '文档润色完成',
            'filename': output_filename
        })
        
    except Exception as e:
        return jsonify({'error': f'文档润色失败: {str(e)}'})

@app.route('/download_polished/<filename>')
async def download_polished(filename):
    """下载润色后文件的路由"""
    try:
        # 提供润色后的文件下载
        return await send_file(
            os.path.join(app.config['POLISHED_FOLDER'], filename),
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': f'下载文件失败: {str(e)}'})

# 应用入口
if __name__ == '__main__':
    app.run(debug=True)  # 以调试模式运行应用 