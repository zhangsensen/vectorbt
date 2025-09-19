#!/bin/bash

# 创建符号链接脚本
cd /Users/zhangshenshen/longport/vectorbt_workspace/data

# 处理 1min 数据
for file in /Users/zhangshenshen/longport/data/HK/*1min*.parquet; do
    symbol=$(basename "$file" | cut -d'_' -f1 | sed 's/HK$/.HK/')
    ln -sf "$file" "1m/$symbol.parquet"
done

# 处理 2min 数据
for file in /Users/zhangshenshen/longport/data/HK/*2min*.parquet; do
    symbol=$(basename "$file" | cut -d'_' -f1 | sed 's/HK$/.HK/')
    ln -sf "$file" "2m/$symbol.parquet"
done

# 处理 3min 数据
for file in /Users/zhangshenshen/longport/data/HK/*3min*.parquet; do
    symbol=$(basename "$file" | cut -d'_' -f1 | sed 's/HK$/.HK/')
    ln -sf "$file" "3m/$symbol.parquet"
done

# 处理 5min 数据
for file in /Users/zhangshenshen/longport/data/HK/*5min*.parquet; do
    symbol=$(basename "$file" | cut -d'_' -f1 | sed 's/HK$/.HK/')
    ln -sf "$file" "5m/$symbol.parquet"
done

# 处理 1day 数据
for file in /Users/zhangshenshen/longport/data/HK/*1day*.parquet; do
    symbol=$(basename "$file" | cut -d'_' -f1 | sed 's/HK$/.HK/')
    ln -sf "$file" "1d/$symbol.parquet"
done

echo "符号链接创建完成！"