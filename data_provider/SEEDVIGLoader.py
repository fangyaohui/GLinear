import os
import numpy as np
from scipy.io import loadmat
from typing import Dict, Tuple

from torch.utils.data import Dataset


class SEEDVIGLoader(Dataset):
    def __init__(self, data_dir: str, target_feature: str = 'psd_movingAve'):
        """
        初始化加载器
        :param data_dir: MAT文件目录路径
        :param target_feature: 指定加载的特征名称（psd_movingAve, psd_LDS, de_movingAve, de_LDS）
        """
        self.data_dir = data_dir
        self.target_feature = target_feature
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
        self.file_list.sort()  # 按文件名排序以保持被试顺序

    def _load_single_file(self, file_path: str) -> np.ndarray:
        """
        加载单个MAT文件并提取目标特征
        """
        mat_data = loadmat(file_path)
        feature_data = mat_data[self.target_feature]  # 提取指定特征
        return feature_data.astype(np.float32)       # 转换为32位浮点

    def load_all(self) -> Dict[str, np.ndarray]:
        """
        加载所有被试数据，返回字典格式 {被试ID: 特征矩阵}
        """
        data_dict = {}
        for file_name in self.file_list:
            subj_id = os.path.splitext(file_name)[0]  # 假设文件名为被试ID
            file_path = os.path.join(self.data_dir, file_name)
            data_dict[subj_id] = self._load_single_file(file_path)
        return data_dict

    def load_merged(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        合并所有被试数据，返回 (X, y) 格式
        假设标签文件为独立文件 'perclos_labels.mat'
        """
        # 加载特征
        all_features = []
        for file_name in self.file_list:
            file_path = os.path.join(self.data_dir, file_name)
            all_features.append(self._load_single_file(file_path))
        X = np.concatenate(all_features, axis=1)  # 合并样本维度 (17, 21 * 885, 25)

        # 加载标签（假设标签文件在同一目录）
        label_data = loadmat(os.path.join(self.data_dir, '1_20151124_noon_2.mat'))
        y = label_data['perclos'].flatten()  # 转换为1D数组 (21 * 885,)
        return X.transpose(1, 0, 2), y  # 维度重排为 (样本数, 导联, 频带)


if __name__ == '__main__':

    # 初始化加载器
    loader = SEEDVIGLoader(data_dir='../dataset/SEED_VIG/Forehead_EEG/EEG_Feature_2Hz',
                           target_feature='psd_LDS')

    # 加载所有被试数据（字典格式）
    subject_data = loader.load_all()
    print(f"被试数量: {len(subject_data)}, 单个被试数据形状: {subject_data['1_20151124_noon_2'].shape}")

    # 加载合并后的训练数据
    X, y = loader.load_merged()
    print(f"特征矩阵形状: {X.shape}, 标签形状: {y.shape}")