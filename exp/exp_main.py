import csv
import datetime
from time import sleep

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Autoformer, DLinear, Linear, NLinear, DNGLinear, GLinear, RLinear
from utils.ProjectLogger import ProjectLogger
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')  # 忽略警告信息


class Exp_Main(Exp_Basic):
    def __init__(self, args,setting):
        """ 实验主类初始化
        Args:
            args: 包含所有配置参数的命名空间
        功能：
            继承基础实验类并初始化模型、数据和训练相关组件
        """
        self.setting = setting
        super(Exp_Main, self).__init__(args)
        # =============== 初始化日志系统 ===============
        self._init_logger()

        self._build_result_csv()

    def _init_logger(self):
        """安全初始化日志目录"""
        try:
            log_dir = os.path.join(self.args.train_data_dir, self.setting, self.args.log_dir)
            log_file = os.path.join(log_dir, self.args.log_file)

            # 确保日志目录存在
            os.makedirs(log_dir, exist_ok=True)

            # 配置日志系统
            ProjectLogger.configure_logger(log_file=log_file)
            self.logger = ProjectLogger.get_logger(__name__)
            self.logger.info("Logger initialized successfully")

        except PermissionError as e:
            raise RuntimeError(f"无权限创建日志目录: {log_dir}") from e
        except Exception as e:
            raise RuntimeError(f"日志初始化失败: {str(e)}") from e

    def _build_result_csv(self):
        # =============== 初始化CSV日志 ===============
        csv_dir = os.path.join(
            self.args.train_data_dir,
            self.setting,
            self.args.csv_dir
        )
        self.csv_path = os.path.join(csv_dir, self.args.csv_file)

        # =============== 创建目录结构 ===============
        os.makedirs(csv_dir, exist_ok=True)
        self.logger.info(f"确保结果目录存在: {csv_dir}")
        csv_headers = ['epoch', 'model', 'seq_len', 'pred_len', 'data', 'batch_size', 'train_loss', 'vali_loss', 'test_loss', 'time_cost', 'learning_rate', "date"]

        # 创建CSV文件并写入表头（如果首次运行）
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_headers)

    def _write_result_csv(self, log_data):

        # =============== 写入CSV文件 ===============
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            # 按表头顺序写入数据
            writer.writerow([
                log_data['epoch'],
                log_data['model'],
                log_data['seq_len'],
                log_data['pred_len'],
                log_data['data'],
                log_data['batch_size'],
                log_data['train_loss'],
                log_data['vali_loss'],
                log_data['test_loss'],
                log_data['time_cost'],
                log_data['learning_rate'],
                log_data['date']
            ])


    def _build_model(self):
        """ 构建预测模型
        返回：
            nn.Module: 初始化后的模型实例
        流程：
            1. 根据参数选择模型类型
            2. 初始化模型结构
            3. 多GPU并行处理（如果启用）
        """
        # 模型类型映射字典
        model_dict = {
            'Autoformer': Autoformer,  # 自相关机制+分解架构
            'DLinear': DLinear,  # 分解线性模型
            'NLinear': NLinear,  # 标准化线性模型
            'Linear': Linear,  # 基础线性模型
            'DNGLinear': DNGLinear,  # 深度分解线性模型
            'GLinear': GLinear,  # 图结构线性模型
            'RLinear': RLinear,  # 递归线性模型
        }
        # 初始化选定模型
        model = model_dict[self.args.model].Model(self.args).float()

        # 多GPU并行处理
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """ 数据加载方法
        Args:
            flag: 数据模式标识（train/val/test/pred）
        返回：
            tuple: (数据集对象, 数据加载器)
        功能：
            调用数据提供工厂方法获取指定模式的数据
        """
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        """ 优化器选择
        返回：
            optim.Adam: Adam优化器实例
        """
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        """ 损失函数选择
        返回：
            nn.MSELoss: 均方误差损失函数
        """
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        """ 验证集评估
        Args:
            vali_data: 验证数据集对象
            vali_loader: 验证数据加载器
            criterion: 损失函数
        返回：
            float: 平均验证损失
        流程：
            1. 设置评估模式
            2. 遍历验证数据集
            3. 构造解码器输入
            4. 模型推理计算损失
        """
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # 数据准备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 解码器输入构造（历史数据+零填充）
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 模型推理
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        """完整的模型训练流程控制方法
        Args:
            setting: 字符串类型，表示当前实验的配置标识（通常包含模型参数和超参数组合）
        流程说明：
            1. 数据加载 - 获取训练/验证/测试数据集和数据加载器
            2. 创建检查点目录 - 用于保存训练过程和最佳模型
            3. 初始化训练组件 - 包括优化器、早停策略、损失函数
            4. 混合精度设置 - 使用AMP加速训练（如果启用）
            5. 训练循环 - 包含前向传播、损失计算、反向传播
            6. 验证评估 - 在验证集和测试集上评估模型性能
            7. 模型保存 - 根据早停策略保存最佳模型
            8. 学习率调整 - 动态调整优化器的学习率
        """
       

        # =============== 数据加载阶段 ===============
        # 获取训练数据集和数据加载器
        train_data, train_loader = self._get_data(flag='train')
        # 如果不是仅训练模式，获取验证集和测试集
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        # =============== 创建检查点目录 ===============
        # 拼接检查点保存路径：基础路径 + 实验配置标识
        path = os.path.join(self.args.train_data_dir, setting, self.args.checkpoints)
        # 如果目录不存在则递归创建
        if not os.path.exists(path):
            os.makedirs(path)

        # 记录当前时间用于计算训练速度
        time_now = time.time()

        # =============== 初始化训练组件 ===============
        # 计算每个epoch的训练步数（总batch数）
        train_steps = len(train_loader)
        # 初始化早停策略对象，设置耐心值和是否显示提示信息
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        # 选择优化器（Adam/SGD等）
        model_optim = self._select_optimizer()
        # 选择损失函数（MSE/MAE等）
        criterion = self._select_criterion()

        # =============== 混合精度设置 ===============
        # 如果启用自动混合精度训练
        if self.args.use_amp:
            # 创建梯度缩放器，用于防止float16下溢
            scaler = torch.cuda.amp.GradScaler()

        # =============== 训练主循环 ===============
        # 遍历所有训练轮次
        for epoch in range(self.args.train_epochs):
            # 迭代计数器重置
            iter_count = 0
            # 当前epoch的损失记录列表
            train_loss = []

            # 设置模型为训练模式（启用dropout等）
            self.model.train()
            # 记录单个epoch的开始时间
            epoch_time = time.time()
            # 遍历训练数据加载器
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # 迭代计数器累加
                iter_count += 1
                # 清空优化器梯度
                model_optim.zero_grad()

                # 将数据转移到指定设备（GPU/CPU）
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 构建decoder输入（时间序列预测任务常用方法）
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # =============== 前向传播 ===============
                # 混合精度训练分支
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():  # 自动转换精度上下文
                        # 线性模型分支
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            # 带注意力机制的模型分支
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        # 特征维度处理（多变量预测时取最后一维）
                        f_dim = -1 if self.args.features == 'MS' else 0
                        # 截取预测部分的输出
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        # 对齐标签数据的形状
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # 计算损失值
                        loss = criterion(outputs, batch_y)
                        # 记录当前损失
                        train_loss.append(loss.item())
                # 普通精度训练分支
                else:
                    # 结构同上，不使用自动精度转换
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # 特征维度处理
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # =============== 训练日志打印 ===============
                # 每100个iteration打印一次进度
                if (i + 1) % 100 == 0:
                    # 打印当前迭代次数、epoch数和损失值
                    self.logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    # 计算平均每个iteration耗时
                    speed = (time.time() - time_now) / iter_count
                    # 计算剩余训练时间
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # 打印速度和时间预估
                    self.logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    # 重置计数器和计时器
                    iter_count = 0
                    time_now = time.time()

                # =============== 反向传播 ===============
                if self.args.use_amp:
                    # 混合精度版反向传播
                    scaler.scale(loss).backward()  # 缩放后的损失反向传播
                    scaler.step(model_optim)  # 优化器参数更新
                    scaler.update()  # 调整缩放系数
                else:
                    # 普通精度版反向传播
                    loss.backward()  # 损失反向传播
                    model_optim.step()  # 优化器参数更新

            # =============== 训练后处理 ===============
            # 打印当前epoch耗时
            self.logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # 计算当前epoch的平均训练损失
            train_loss = np.average(train_loss)

            # 非仅训练模式时进行验证
            if not self.args.train_only:
                # 在验证集和测试集上评估
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)
                # 打印详细损失信息
                self.logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                # 执行早停策略（以验证损失为判断依据）
                early_stopping(vali_loss, self.model, path)
            else:
                # 仅训练模式下的打印信息
                self.logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                # 以训练损失作为早停判断依据（不推荐但可用）
                early_stopping(train_loss, self.model, path)

            # 检测是否触发早停
            if early_stopping.early_stop:
                self.logger.info("Early stopping")  # 提前终止提示
                break  # 跳出训练循环

            # 初始化日志数据结构（在epoch循环内）
            log_data = {
                # 实验基本信息
                'epoch': epoch + 1,  # 当前epoch数
                'model': self.args.model,  # 模型名称（从参数获取）
                'seq_len': self.args.seq_len,  # 输入序列长度
                'pred_len': self.args.pred_len,  # 预测序列长度
                'data': self.args.data,  # 数据集名称
                'batch_size': self.args.batch_size,  # 批大小

                # 动态训练指标
                'train_loss': round(train_loss, 5),  # 训练损失（保留5位小数）
                'vali_loss': round(vali_loss, 5) if not self.args.train_only else 'N/A',
                'test_loss': round(test_loss, 5) if not self.args.train_only else 'N/A',

                # 资源与时间
                'time_cost': round(time.time() - epoch_time, 2),  # 耗时（秒）
                'learning_rate': round(model_optim.param_groups[0]['lr'], 6),  # 当前学习率

                # 实验元信息
                'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 精确到秒的时间戳
            }
            self._write_result_csv(log_data)

            # =============== 学习率调整 ===============
            adjust_learning_rate(model_optim, epoch + 1, self.args)  # 调整学习率

        # =============== 训练结束处理 ===============
        # 构建最佳模型路径

        best_model_path = path + '/' + 'checkpoint.pth'
        # 加载最优模型参数
        self.model.load_state_dict(torch.load(best_model_path))

        # 返回训练好的模型
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            self.logger.info('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.train_data_dir, setting, self.args.checkpoints, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        #folder_path = './test_results/' + setting + '/'
        folder_path = self.args.train_data_dir + setting + '/test/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # self.logger.info(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                    pdf_path = os.path.join(folder_path, self.args.train_time)
                    os.makedirs(pdf_path, exist_ok=True)
                    pdf_path = pdf_path + "/" + str(i) + '.pdf'
                    visual(gt, pd, pdf_path)

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        #folder_path = './results/' + setting + '/'
        folder_path = self.args.train_data_dir + setting + '/test_pred_npy/' + self.args.train_time + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        self.logger.info('mse:{}, mae:{}'.format(mse, mae))
        f = open(folder_path + "/" + "result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path  + "_" + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.train_data_dir, setting, self.args.checkpoints, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)
        
        # result save
        #folder_path = './results/' + setting + '/'
        folder_path = self.args.train_data_dir + setting + '/prediction/'
        # folder_path = '/spo/NewbornTime/tahir098/SVF/data/Extra/results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return
