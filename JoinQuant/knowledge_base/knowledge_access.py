# -*- coding: utf-8 -*-
# 知识库访问接口
import os
import json

class KnowledgeBase:
    def __init__(self, base_path = "knowledge_base"):
        """
        初始化知识库访问接口
        
        Args:
            base_path: 知识库根目录路径
        """
        self.base_path = base_path
        self.modules = {
            'stock_data': '股票数据',
            'industry_data': '行业数据', 
            'index_data': '指数数据',
            'technical_indicators': '技术分析指标',
            'jq_factor_library': '聚宽因子库'
        }
    
    def get_module_list(self):
        """
        获取知识库模块列表
        
        Returns:
            dict: 模块名称和中文描述的映射
        """
        return self.modules
    
    def get_module_content(self, module_name):
        """
        获取指定模块的文档内容
        
        Args:
            module_name: 模块名称
            
        Returns:
            str: 模块文档内容，如果模块不存在返回None
        """
        if module_name not in self.modules:
            return None
            
        doc_path = os.path.join(self.base_path, module_name, 'api_documentation.md')
        if os.path.exists(doc_path):
            import codecs
            with codecs.open(doc_path, 'r', 'utf-8') as f:
                return f.read()
        return None
    
    def search_in_module(self, module_name, keyword):
        """
        在指定模块中搜索关键词
        
        Args:
            module_name: 模块名称
            keyword: 搜索关键词
            
        Returns:
            list: 包含搜索结果的列表，每个元素包含标题和内容片段
        """
        content = self.get_module_content(module_name)
        if not content:
            return []
        
        results = []
        lines = content.split('\n')
        current_section = ""
        
        for i, line in enumerate(lines):
            if line.startswith('#'):
                current_section = line.strip('# ').strip()
            elif keyword.lower() in line.lower():
                # 获取上下文
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                context = '\n'.join(lines[start:end])
                
                results.append({
                    'section': current_section,
                    'content': context,
                    'line_number': i + 1
                })
        
        return results
    
    def search_across_modules(self, keyword):
        """
        在所有模块中搜索关键词
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            dict: 按模块分组的搜索结果
        """
        results = {}
        for module_name in self.modules:
            module_results = self.search_in_module(module_name, keyword)
            if module_results:
                results[module_name] = module_results
        return results
    
    def get_api_by_name(self, module_name, api_name):
        """
        根据API名称获取API详细信息
        
        Args:
            module_name: 模块名称
            api_name: API名称
            
        Returns:
            dict: API详细信息，包括名称、描述、参数、返回值、示例等
        """
        content = self.get_module_content(module_name)
        if not content:
            return None
        
        lines = content.split('\n')
        api_info = None
        collecting = False
        
        for line in lines:
            if line.startswith('###') and api_name in line:
                collecting = True
                api_info = {
                    'name': line.strip('# ').strip(),
                    'description': '',
                    'parameters': '',
                    'returns': '',
                    'examples': ''
                }
            elif collecting:
                if line.startswith('###'):
                    break
                elif '调用方法' in line:
                    api_info['method'] = ''
                elif '参数' in line:
                    api_info['parameters'] = ''
                elif '返回值' in line:
                    api_info['returns'] = ''
                elif '示例' in line:
                    api_info['examples'] = ''
                elif api_info:
                    # 简化处理，实际应用中可能需要更复杂的解析
                    pass
        
        return api_info

# 全局知识库实例
kb = KnowledgeBase()

def get_knowledge_base():
    """
    获取知识库实例
    
    Returns:
        KnowledgeBase: 知识库实例
    """
    return kb

def list_modules():
    """
    列出所有知识库模块
    
    Returns:
        dict: 模块名称和中文描述的映射
    """
    return kb.get_module_list()

def get_module_documentation(module_name):
    """
    获取模块文档内容
    
    Args:
        module_name: 模块名称
        
    Returns:
        str: 模块文档内容
    """
    return kb.get_module_content(module_name)

def search_knowledge(keyword, module_name = None):
    """
    搜索知识库内容
    
    Args:
        keyword: 搜索关键词
        module_name: 指定模块名称，如果为None则搜索所有模块
        
    Returns:
        dict: 搜索结果
    """
    if module_name:
        results = {module_name: kb.search_in_module(module_name, keyword)}
        return {k: v for k, v in results.items() if v}
    else:
        return kb.search_across_modules(keyword)

def get_api_info(module_name, api_name):
    """
    获取API详细信息
    
    Args:
        module_name: 模块名称
        api_name: API名称
        
    Returns:
        dict: API详细信息
    """
    return kb.get_api_by_name(module_name, api_name)

# 使用示例
if __name__ == "__main__":
    # 创建知识库实例
    knowledge_base = get_knowledge_base()
    
    # 列出所有模块
    print("知识库模块列表:")
    modules = list_modules()
    for name, description in modules.items():
        print("  {}: {}".format(name, description))
    
    # 获取股票数据模块文档
    stock_doc = get_module_documentation('stock_data')
    if stock_doc:
        print("\n股票数据模块文档长度: {}".format(len(stock_doc)))
    
    # 搜索关键词
    print("\n搜索'get_price':")
    search_results = search_knowledge('get_price')
    for module, results in search_results.items():
        print("  {}模块找到{}个结果".format(module, len(results)))
    
    # 在特定模块中搜索
    print("\n在技术指标模块中搜索'MACD':")
    tech_results = search_knowledge('MACD', 'technical_indicators')
    print("  找到{}个结果".format(len(tech_results.get('technical_indicators', []))))
