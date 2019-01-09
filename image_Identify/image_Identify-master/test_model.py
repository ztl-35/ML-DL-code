import torch
from torch.autograd.variable import Variable
import Net_class
import torch.nn as nn
import torch.optim as optimizer
import numpy as np
import data_transfer
from torch.utils.data import DataLoader
import Mydataset
import matplotlib.pyplot as plt
import image_cut
import os


# 设置pytorch GPU显卡号码
torch.cuda.set_device(1)
# 将图片的格式进行定义
torch.set_default_tensor_type('torch.cuda.DoubleTensor')

if __name__ == '__main__':
    # file_root_path = input('请输入测试文件夹根目录的绝对路径:\n')
    # 整理数据到测试文件夹下
    # print('将测试数据整理输出到当前目录的 testdata 文件下:')
    # image_cut.cut_test_image(file_root_path)

    # 查看testdata文件夹下面的文件夹数目（代表了有多少个测试数据）
    file_absolute_path = '/home/gpu-105/PycharmProjects/test/test_data/'
    test_data_list_path = []
    for lists in os.listdir(file_absolute_path):
        sub_path = os.path.join(file_absolute_path, lists)
        if os.path.isfile(sub_path):
            pass
        elif os.path.isdir(sub_path):
            test_data_list_path.append(sub_path)
    test_data_matrix, test_data_label_list = data_transfer.process_test_data2Matrix(test_data_list_path)
    test_data_matrix = test_data_matrix[:, np.newaxis, :, :]

    # # 测试数据集成
    test_loader = DataLoader(Mydataset.MyDataset(test_data_matrix, test_data_label_list), batch_size=3, shuffle=False)
    # model = torch.load('model.pkl')
    model = Net_class.Net()
    model.load_state_dict(torch.load('params.pkl'))

    for data in test_loader:
        test_batch_data, test_data_label = data
        if torch.cuda.is_available():
            test_batch_data = Variable(test_batch_data.cuda())
        predict = model(test_batch_data)
        mask = predict.ge(0.5).double()
        predict_result_format01 = mask.data.cpu().numpy()
        flag = False
        for i in range(3):
            if predict_result_format01[i][0] == 1:
                flag = True
            else:
                flag = False
                break
        if flag:
            with open('result.txt', 'a+') as f:
                f.write('合格' + '\n')
        else:
            with open('result.txt', 'a+') as f:
                f.write('不合格' + ' ')
            for i in range(3):
                if predict_result_format01[i][0] == 0:
                    with open('result.txt', 'a+') as f:
                        f.write(str(i + 1) + ' ')
            with open('result.txt', 'a+') as f:
                f.write('\n')
