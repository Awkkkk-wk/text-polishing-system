from flask import Flask, render_template, request, jsonify, send_file
from file_rag import FileRAGSystem
import os
import asyncio
from quart import Quart, render_template, request, jsonify, send_file
from datetime import datetime
import time
import requests

app = Quart(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['POLISHED_FOLDER'] = 'polished'
app.config['KNOWLEDGE_BASE_FOLDER'] = 'knowledge_base'  # 添加知识库文件夹配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['POLISHED_FOLDER'], exist_ok=True)
os.makedirs(app.config['KNOWLEDGE_BASE_FOLDER'], exist_ok=True)  # 创建知识库文件夹

# 使用单例模式初始化RAG系统
_rag_system = None

def get_rag_system():
    global _rag_system
    if _rag_system is None:
        _rag_system = FileRAGSystem()
    return _rag_system

@app.route('/')
async def index():
    return await render_template('index.html')

@app.route('/upload_kb', methods=['POST'])
async def upload_knowledge_base():
    if 'file' not in (await request.files):
        return jsonify({'error': '没有文件被上传'})
    
    file = (await request.files)['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    if file:
        try:
            # 保存文件到uploads目录
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            await file.save(file_path)
            
            # 处理知识库文件
            rag_system = get_rag_system()
            
            # 异步处理文件上传
            def process_upload():
                return rag_system.upload_file(file_path)
            
            # 使用线程池处理文件上传
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
    form = await request.form
    filename = form.get('filename', '')
    rag_system = get_rag_system()
    
    # 修改保存路径到知识库文件夹，确保文件扩展名为.json
    if filename:
        # 如果用户没有指定.json扩展名，自动添加
        if not filename.lower().endswith('.json'):
            filename += '.json'
        output_path = os.path.join(app.config['KNOWLEDGE_BASE_FOLDER'], filename)
    else:
        output_path = os.path.join(app.config['KNOWLEDGE_BASE_FOLDER'], f"knowledge_base_{time.strftime('%Y%m%d_%H%M%S')}.json")
    
    try:
        # 异步处理文件保存
        def process_save():
            return rag_system.save_knowledge_base(output_path)
        
        # 使用线程池处理文件保存
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
    form = await request.form
    text = form.get('text', '')
    model = form.get('model', 'both')
    
    if not text:
        return jsonify({'error': '文本不能为空'})
    
    try:
        # 增加超时时间到120秒
        timeout = 120
        
        # 获取RAG系统
        rag_system = get_rag_system()
        
        # 直接await异步方法，确保协程被正确等待
        result = await asyncio.wait_for(
            rag_system.polish_text(text, model),
            timeout=timeout
        )
        
        # 检查是否有错误
        if 'error' in result:
            return jsonify({'error': result['error']})
            
        # 检查有道翻译结果
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
            
        # 确保所有值都是字符串类型
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
    if 'file' not in (await request.files):
        return jsonify({'error': '没有文件被上传'})
    
    file = (await request.files)['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    form = await request.form
    model = form.get('model', 'both')
    
    try:
        # 保存上传的文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        await file.save(file_path)
        
        # 处理文档
        rag_system = get_rag_system()
        documents = rag_system.supported_extensions[os.path.splitext(file.filename)[1].lower()](file_path)
        
        # 润色文档内容
        polished_content = []
        for doc in documents:
            result = await rag_system.polish_text(doc['text'], model)
            
            # 添加原文
            polished_content.append(f"原文：\n{doc['text']}\n")
            
            # 添加有道翻译润色结果
            if model in ['youdao', 'both']:
                polished_content.append(f"\n有道翻译润色：\n{result['suggested']['youdao']}")
                # 添加有道翻译的分析结果
                polished_content.append(f"\n有道翻译分析：\n{result['analysis']['youdao']}")
            
            # 添加智谱API润色结果
            if model in ['zhipu', 'both']:
                polished_content.append(f"\n智谱API润色：\n{result['suggested']['zhipu']}")
                # 添加智谱API的分析结果
                polished_content.append(f"\n智谱API分析：\n{result['analysis']['zhipu']}")
            
            # 使用智谱API对润色结果进行综合分析
            if model == 'both':
                analysis_prompt = f"""请综合分析以下润色结果：

原文：
{doc['text']}

有道翻译润色：
{result['suggested']['youdao']}

智谱API润色：
{result['suggested']['zhipu']}

请从以下几个方面进行分析：
1. 两个润色版本的优缺点比较
2. 哪个版本更符合原文的语义
3. 哪个版本的专业性更强
4. 哪个版本的语言表达更流畅
5. 给出最终建议：应该选择哪个版本，或者如何结合两个版本的优点

请用中文回答，要求分析客观、专业、详细。"""
                
                headers = {
                    "Authorization": f"Bearer {rag_system.zhipu_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "prompt": analysis_prompt,
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
                        comprehensive_analysis = analysis_result['data']['choices'][0]['content'].strip()
                        polished_content.append(f"\n综合分析：\n{comprehensive_analysis}")
            
            polished_content.append("\n" + "="*50 + "\n")
        
        # 生成输出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{os.path.splitext(file.filename)[0]}_polished_{timestamp}.txt"
        output_path = os.path.join(app.config['POLISHED_FOLDER'], output_filename)
        
        # 保存润色后的内容
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(polished_content))
        
        return jsonify({
            'message': '文档润色完成',
            'filename': output_filename
        })
        
    except Exception as e:
        return jsonify({'error': f'文档润色失败: {str(e)}'})

@app.route('/download_polished/<filename>')
async def download_polished(filename):
    try:
        return await send_file(
            os.path.join(app.config['POLISHED_FOLDER'], filename),
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': f'下载文件失败: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True) 