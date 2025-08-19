# -*- coding: utf-8 -*-
# 知识库使用示例

from knowledge_base.knowledge_access import (
    get_knowledge_base,
    list_modules,
    get_module_documentation,
    search_knowledge,
    get_api_info
)

def demo_knowledge_base():
    """演示知识库的基本使用方法"""
    
    print("=== 量化研究知识库使用演示 ===\n")
    
    # 1. 列出所有模块
    print("1. 知识库模块列表:")
    modules = list_modules()
    for name, description in modules.items():
        print("   - {}: {}".format(name, description))
    print()
    
    # 2. 获取特定模块文档
    print("2. 获取股票数据模块文档:")
    stock_doc = get_module_documentation('stock_data')
    if stock_doc:
        # 只显示前500个字符作为示例
        print("   文档长度: {} 字符".format(len(stock_doc)))
        try:
            print("   文档预览: {}...".format(stock_doc[:500].encode('utf-8')))
        except UnicodeEncodeError:
            print("   文档预览: (包含中文字符)")
    else:
        print("   未找到股票数据模块文档")
    print()
    
    # 3. 搜索功能演示
    print("3. 搜索功能演示:")
    
    # 在所有模块中搜索
    print("   搜索 'get_price' (所有模块):")
    search_results = search_knowledge('get_price')
    for module, results in search_results.items():
        print("     {} 模块找到 {} 个结果".format(module, len(results)))
        for result in results[:2]:  # 只显示前2个结果
            try:
                print("       - {}: {}...".format(result['section'].encode('utf-8'), result['content'][:100].encode('utf-8')))
            except UnicodeEncodeError:
                print("       - Section: {}...".format(result['line_number']))
    print()
    
    # 在特定模块中搜索
    print("   搜索 'MACD' (技术指标模块):")
    tech_results = search_knowledge('MACD', 'technical_indicators')
    tech_results_list = tech_results.get('technical_indicators', [])
    print("     找到 {} 个结果".format(len(tech_results_list)))
    for result in tech_results_list[:2]:  # 只显示前2个结果
        try:
            print("       - {}: {}...".format(result['section'].encode('utf-8'), result['content'][:100].encode('utf-8')))
        except UnicodeEncodeError:
            print("       - Section: {}...".format(result['line_number']))
    print()
    
    # 4. API信息获取演示
    print("4. API信息获取演示:")
    # 注意：这个功能需要进一步完善解析逻辑
    print("   API信息获取功能已实现框架，具体解析逻辑可根据需要进一步完善")
    print()
    
    # 5. 实际使用场景示例
    print("5. 实际使用场景示例:")
    print("   场景：编写策略时需要查找股票数据获取方法")
    print("   代码示例:")
    print("   ```python")
    print("   from jqdata import *")
    print("   # 获取股票基本信息")
    print("   info = get_security_info('000001.XSHE')")
    print("   print(info.display_name)  # 输出: 平安银行")
    print("   ```")
    print()
    
    print("=== 演示结束 ===")

def find_stock_data_apis():
    """查找股票数据相关API"""
    print("=== 股票数据API查找 ===")
    
    # 搜索股票数据模块中的关键API
    key_apis = ['get_security_info', 'get_all_securities', 'get_price', 'get_fundamentals']
    
    for api_name in key_apis:
        print("\n查找API: {}".format(api_name))
        results = search_knowledge(api_name, 'stock_data')
        stock_results = results.get('stock_data', [])
        for result in stock_results:
            try:
                print("  位置: {}".format(result['section'].encode('utf-8')))
                print("  内容: {}...".format(result['content'][:200].encode('utf-8')))
            except UnicodeEncodeError:
                print("  位置: Section {}...".format(result['line_number']))

def find_technical_indicators():
    """查找技术指标相关API"""
    print("\n=== 技术指标API查找 ===")
    
    # 搜索技术指标模块中的关键指标
    key_indicators = ['MACD', 'RSI', 'BOLL', 'KDJ']
    
    for indicator in key_indicators:
        print("\n查找指标: {}".format(indicator))
        results = search_knowledge(indicator, 'technical_indicators')
        tech_results = results.get('technical_indicators', [])
        for result in tech_results:
            try:
                print("  位置: {}".format(result['section'].encode('utf-8')))
                print("  内容: {}...".format(result['content'][:300].encode('utf-8')))
            except UnicodeEncodeError:
                print("  位置: Section {}...".format(result['line_number']))

if __name__ == "__main__":
    # 运行演示
    demo_knowledge_base()
    
    # 查找特定API
    find_stock_data_apis()
    find_technical_indicators()
    
    print("\n=== 使用建议 ===")
    print("1. 在编写量化策略时，可以使用 search_knowledge() 快速查找相关API")
    print("2. 使用 list_modules() 了解知识库包含的内容")
    print("3. 使用 get_module_documentation() 获取完整模块文档")
    print("4. 建议将此知识库集成到您的开发环境中，提高开发效率")
