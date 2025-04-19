#!/bin/bash
# 这是一个用于批量运行时间序列预测实验的自动化脚本
# 主要功能：创建日志目录，遍历不同模型和参数组合运行训练，并记录日志

# 第一部分：创建日志目录结构
# -----------------------------------------------------------
# 检查当前目录下是否存在logs目录，如果不存在则创建
if [ ! -d "./logs" ]; then
    mkdir ./logs  # 创建顶层日志目录
fi

# 检查是否存在LookBackWindow子目录，不存在则创建
if [ ! -d "./logs/LookBackWindow" ]; then
    mkdir ./logs/LookBackWindow  # 创建特定实验的日志子目录
fi

# 第二部分：参数遍历与实验执行
# -----------------------------------------------------------
# 三层嵌套循环遍历不同参数组合
for model_name in Autoformer  # 模型名称循环（当前只有Autoformer一个模型）
do 
for seq_len in 336  # 输入序列长度循环（注意：实际脚本中后续使用的是固定96，此处可能未生效）
do
for pred_len in 12 24 48 96 192 336 720  # 预测长度循环（多个预测跨度）
do

# 执行electricity数据集实验
# -----------------------------------------------------------
python -u run_longExp.py \  # -u参数强制不缓冲输出
    --is_training 1 \  # 训练模式
    --root_path ./dataset/ \  # 数据集根目录
    --data_path electricity.csv \  # 具体数据文件
    --model_id electricity_96_$pred_len \  # 模型标识（包含预测长度）
    --model $model_name \  # 使用的模型名称
    --data custom \  # 数据类型为自定义格式
    --features M \  # 多变量预测模式（M表示多元）
    --seq_len 96 \  # 输入序列长度（与循环变量不一致，可能是个问题）
    --label_len 48 \  # 解码器初始输入长度
    --pred_len $pred_len \  # 预测长度（使用循环变量）
    --e_layers 2 \  # 编码器层数
    --d_layers 1 \  # 解码器层数
    --factor 3 \  # ProbSparse自注意力因子
    --enc_in 321 \  # 编码器输入特征维度（对应数据集特征数）
    --dec_in 321 \  # 解码器输入特征维度
    --c_out 321 \  # 输出维度
    --des 'Exp' \  # 实验描述
    --itr 1 \  # 实验迭代次数
    > logs/LookBackWindow/$model_name'_electricity'_$seq_len'_'$pred_len.log  # 日志输出重定向

# 类似地执行traffic数据集实验（参数差异说明）
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id traffic_96_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \  # 不同数据集的输入维度不同
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 3 \  # 显示指定训练轮次（其他实验使用默认值）
    > logs/LookBackWindow/$model_name'_traffic'_$seq_len'_'$pred_len.log

# weather数据集实验
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id weather_96_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \  # 天气数据的特征维度
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 2 \  # 指定更少的训练轮次
    > logs/LookBackWindow/$model_name'_weather'_$seq_len'_'$pred_len.log

# ETTh1数据集实验
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \  # 电力变压器温度数据集
    --model_id ETTh1_96_$pred_len \
    --model $model_name \
    --data ETTh1 \  # 使用专门的数据处理方式
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \  # ETTh1的7个特征
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    > logs/LookBackWindow/$model_name'_Etth1'_$seq_len'_'$pred_len.log

done
done
done