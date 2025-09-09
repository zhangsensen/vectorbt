#!/usr/bin/env bash
# 一键启动港股分析脚本

# 创建日志目录
mkdir -p logs

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 启动分析
echo "🚀 启动53只港股全时间框架全因子分析..."
echo "📅 开始时间: $(date)"
echo "📝 主日志文件: logs/master_${TIMESTAMP}.log"
echo "📊 检查点将保存在: logs/sequential_analysis_summary_${TIMESTAMP}/"

# 使用screen在后台运行
screen -S hk53_analysis -dm bash -c \
  "python3 run_53_stocks_sequential.py --start 0 --end 53 2>&1 | tee logs/master_${TIMESTAMP}.log"

echo "✅ 分析已在后台启动!"
echo "📋 使用以下命令查看进度:"
echo "   screen -r hk53_analysis"
echo "🔍 使用以下命令查看日志:"
echo "   tail -f logs/master_${TIMESTAMP}.log"
echo "⏹️  使用以下命令停止分析:"
echo "   screen -S hk53_analysis -X quit"
echo ""
echo "🎯 分析完成后，结果将保存在以下目录:"
echo "   logs/individual_analysis/         # 各股票详细分析结果"
echo "   logs/sequential_analysis_summary_${TIMESTAMP}/  # 总体分析报告和检查点"