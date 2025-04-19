#!/bin/bash
# 自动化多模型时序预测实验脚本
# 主要功能：遍历不同模型、学习率、序列长度和预测长度，在多个数据集上运行实验

# 安全检查：如果命令失败则退出
set -euo pipefail

# 第一部分：创建日志目录
# -----------------------------------------------------------
log_dir="./logs/LongForecasting"
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"  # 自动创建多级日志目录
    echo "Created log directory: $log_dir"
fi

# 第二部分：参数配置
# -----------------------------------------------------------
datasets=("electricity" "ETTh1" "traffic" "weather")  # 数据集数组
declare -A enc_in_map=(  # 数据集与输入维度的映射
    ["electricity"]=321
    ["ETTh1"]=7
    ["traffic"]=862
    ["weather"]=21
)
declare -A batch_size_map=(  # 数据集与batch size的映射
    ["electricity"]=16
    ["ETTh1"]=8
    ["traffic"]=16
    ["weather"]=16
)

# 第三部分：嵌套循环执行实验
# -----------------------------------------------------------
for model_name in NLinear DLinear RLinear GLinear; do  # 模型列表
for lr in 0.001; do                   # 学习率（可扩展添加更多值）
for seq_len in 336; do                # 输入序列长度
for pred_len in 12 24 48 96 192 336 720; do  # 预测长度
for dataset in "${datasets[@]}"; do   # 遍历所有数据集

    # 参数获取
    data_path="$dataset.csv"          # 自动生成数据路径
    enc_in="${enc_in_map[$dataset]}"  # 从映射表获取维度
    batch_size="${batch_size_map[$dataset]}"  # 获取batch size
    
    # 特殊处理ETTh1数据集
    if [ "$dataset" == "ETTh1" ]; then
        data_type="ETTh1"             # 特殊数据格式处理
    else
        data_type="custom"
    fi

    # 构造日志文件名（更安全的变量引用）
    log_file="${model_name}_${dataset}_${seq_len}_${pred_len}_lr_${lr}.log"
    
    # 执行训练命令
    python -u run_longExp.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path "$data_path" \
        --model_id "${dataset}_${seq_len}_${pred_len}" \
        --model "$model_name" \
        --data "$data_type" \
        --features M \
        --seq_len "$seq_len" \
        --pred_len "$pred_len" \
        --enc_in "$enc_in" \
        --des 'Exp' \
        --itr 1 \
        --batch_size "$batch_size" \
        --learning_rate "$lr" \
        > "${log_dir}/${log_file}" 2>&1  # 重定向标准错误和输出

done # 结束数据集循环
done # 结束pred_len循环
done # 结束seq_len循环
done # 结束lr循环
done # 结束model_name循环