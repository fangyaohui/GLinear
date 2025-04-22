import argparse
import os
import datetime
import torch
from sympy.printing.numpy import const

from exp.exp_main import Exp_Main
import random
import numpy as np

# -*- coding: utf-8 -*-
"""
时间序列预测主程序
功能：支持多种Transformer类模型训练/测试，包含完整实验流水线
核心组件：
1. 可复现性设置
2. 参数配置系统
3. GPU资源管理
4. 实验流水线控制
"""

# ==================== 可复现性设置 ====================
fix_seed = 2021  # 固定随机种子保证实验可复现
random.seed(fix_seed)
torch.manual_seed(fix_seed)  # 设置PyTorch的CPU和CUDA随机种子
np.random.seed(fix_seed)  # 设置NumPy随机种子

# ==================== 参数解析器配置 ====================
parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model', type=str, required=False, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')
parser.add_argument('--log_dir', type=str, default='logs/', help='location of model logs')
parser.add_argument('--log_file', type=str, default='train.log', help='location of train logs')
parser.add_argument('--train_data_dir', type=str, default='./train_result/', help='location of train_data_dir')
parser.add_argument('--csv_dir', type=str, default='result_csv/', help='location of csv_dir')
parser.add_argument('--csv_file', type=str, default='train.csv', help='location of csv_file')
parser.add_argument('--train_time', type=str, default='', help='location of csv_file')

# data loader
parser.add_argument('--data', type=str, required=False, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')


def train_ETTh1_GLinear_336_24():
    argparse = parser.parse_args()

    argparse.is_training = 1
    argparse.root_path = "./dataset/"
    argparse.data_path = "ETTh1.csv"
    argparse.model_id = 'ETTh1'
    argparse.model = "GLinear"
    argparse.data = "ETTh1"
    argparse.features = "M"
    argparse.seq_len = 336
    argparse.pred_len = 24
    argparse.enc_in = 7
    argparse.des = "Exp"
    argparse.itr = 1
    argparse.batch_size = 16
    argparse.learning_rate = 0.001
    argparse.train_epochs = 800
    argparse.use_gpu = True
    # argparse.do_predict = True
    return argparse


def train_ETTh1_MyModels_336_24():
    argparse = parser.parse_args()

    argparse.is_training = 1
    argparse.root_path = "./dataset/"
    argparse.data_path = "ETTh1.csv"
    argparse.model_id = 'ETTh1'
    argparse.model = "MyLinearModels"
    argparse.data = "ETTh1"
    argparse.features = "M"
    argparse.seq_len = 336
    argparse.pred_len = 24
    argparse.enc_in = 7
    argparse.des = "Exp"
    argparse.itr = 1
    argparse.batch_size = 16
    argparse.learning_rate = 0.001
    argparse.train_epochs = 800
    argparse.use_gpu = True
    # argparse.do_predict = True



    return argparse

def main():

    # args = train_ETTh1_GLinear_336_24()
    args = train_ETTh1_MyModels_336_24()



    args.train_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.log_file = args.train_time + "_" + args.model + "_" + args.data + "_" + str(args.seq_len) + "_" + str(args.pred_len) + "_" + str(args.learning_rate) + ".log"
    args.csv_file = args.train_time + "_" + args.model + "_" + args.data + "_" + str(args.seq_len) + "_" + str(args.pred_len) + "_" + str(args.learning_rate) + ".csv"
    # GPU自动检测与配置
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    if args.use_gpu:
        # 多GPU配置
        if args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]  # 主GPU设置
            # 注意：实际多GPU训练需要在模型中使用DataParallel/DistributedDataParallel

    # ==================== 实验流水线 ====================
    if args.is_training:
        # 训练模式
        for ii in range(args.itr):
            # 生成实验唯一标识（包含所有关键参数）
            setting = '{model_id}_{model}_{data}_ft{features}_sl{seq_len}ll{label_len}pl{pred_len}_dm{d_model}nh{n_heads}el{e_layers}dl{d_layers}_df{d_ff}_fc{factor}eb{embed}dt{distil}_{des}_{ii}'.format(
                **vars(args), ii=ii)

            # 初始化实验
            exp = Exp_Main(args,setting)
            print(f'>>>>>>> 开始训练 [{setting}] >>>>>>>>>')

            # 训练阶段
            exp.train(setting)

            # 测试阶段（除非设置train_only）
            if not args.train_only:
                print(f'>>>>>>> 测试 [{setting}] <<<<<<<<<')
                exp.test(setting)

            # 未来预测
            if args.do_predict:
                print(f'>>>>>>> 预测 [{setting}] <<<<<<<<<')
                exp.predict(setting, True)

            # 释放GPU缓存
            torch.cuda.empty_cache()
    else:
        # 测试/预测模式
        ii = 0
        setting = ...  # 同训练模式生成逻辑

        exp = Exp_Main(args)

        if args.do_predict:
            print(f'>>>>>>> 预测 [{setting}] <<<<<<<<<')
            exp.predict(setting, True)
        else:
            print(f'>>>>>>> 测试 [{setting}] <<<<<<<<<')
            exp.test(setting, test=1)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

# ==================== 设备配置 ====================

# ==================== 关键组件说明 ====================
"""
Exp_Main 类核心方法：
1. __init__()      : 初始化数据加载器、模型、优化器
2. _get_data()     : 根据参数获取数据集
3. _select_model() : 根据model参数选择模型架构
4. train()         : 包含完整训练循环、验证、早停逻辑
5. test()          : 在测试集评估模型性能
6. predict()       : 进行未来预测

模型架构：
- Autoformer : 结合自相关机制和分解架构的Transformer变体
- Informer   : 使用Prob稀疏注意力的高效Transformer
- DLinear    : 简单高效的线性基准模型
"""

# ==================== 使用示例 ====================
"""
训练命令：
python main.py --is_training 1 --model Informer --data ETTm1 --seq_len 96 --pred_len 24 --batch_size 64

测试命令：
python main.py --is_training 0 --model Autoformer --data custom --do_predict

多GPU训练：
python main.py --use_multi_gpu --devices 0,1,2,3 --batch_size 128
"""

# ==================== 注意事项 ====================
"""
1. 数据准备：确保数据文件位于正确的root_path下
2. GPU内存：大seq_len或大batch_size可能导致OOM，需调整batch_size
3. 混合精度：--use_amp可减少显存占用但可能影响精度
4. 多机训练：需要配合torch.distributed模块使用
5. 结果复现：尽管设置随机种子，不同硬件/库版本仍可能导致微小差异
"""