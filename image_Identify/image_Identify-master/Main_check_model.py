import torch
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.optim as optimizer
import numpy as np
import data_transfer
from torch.utils.data import DataLoader
import Mydataset
import matplotlib.pyplot as plt
import Net_class
# 设置pytorch GPU显卡号码
torch.cuda.set_device(1)
# 将图片的格式进行定义
torch.set_default_tensor_type('torch.cuda.DoubleTensor')
# # 8：1：1 = 训练集 开发集 测试集
train_qualified_image, train_unqualified_image = data_transfer.process_train_data2Matrix()
dev_qualified_image, dev_unqualified_image = data_transfer.process_dev_data2Matrix()

# 分别将合格和不合格的产品对应的图片拼接成一个大的训练数组 与cnn输入网络保持一致
train_data = np.concatenate([train_qualified_image, train_unqualified_image])
train_data = train_data[:, np.newaxis, :, :]
dev_data = np.concatenate([dev_qualified_image, dev_unqualified_image])
dev_data = dev_data[:, np.newaxis, :, :]

# 训练数据集成
train_label = np.array(list(np.ones((1, 800), dtype=int))+(list(np.zeros((1, 800), dtype=int)))).reshape((1600, 1))
train_label_tensor = torch.tensor(train_label)
train_loader = DataLoader(Mydataset.MyDataset(train_data, train_label_tensor), batch_size=64, shuffle=True)

# 开发数据集成
dev_label = np.array(list(np.ones((1, 200), dtype=int))+(list(np.zeros((1, 200), dtype=int)))).reshape((400, 1))
dev_label_tensor = torch.tensor(dev_label)
dev_loader = DataLoader(Mydataset.MyDataset(dev_data, dev_label_tensor), batch_size=64, shuffle=False)

# *****************************************************************************

model = Net_class.Net()
if torch.cuda.is_available():
    model = model.cuda()
loss_fucntion = nn.BCELoss()
optim = optimizer.Adam(model.parameters(), weight_decay=0.1)


epoch_loss_list = []
# 根据F1来选择模型
F1 = 0
print('*'*30)
print('模型开始训练...')
for epoch in range(100):
    correct_count = 0
    loss_sum = 0.0
    for data in train_loader:
        train_batch_data, train_batch_label_tensor = data
        if torch.cuda.is_available():
            train_batch_data = Variable(train_batch_data.cuda())
            train_batch_label_tensor = Variable(train_batch_label_tensor.cuda())
        predict = model(train_batch_data)
        # 保持两者的数据类型一致
        train_batch_label_tensor = train_batch_label_tensor.type_as(predict)
        loss = loss_fucntion(predict, train_batch_label_tensor)

        mask = predict.ge(0.5).double()
        correct_count += (mask.data.cpu().numpy().reshape((train_batch_data.shape[0], 1)) == train_batch_label_tensor.data.cpu().numpy()).sum()
        loss_sum += loss.data.cpu().numpy()

        optim.zero_grad()
        loss.backward()
        optim.step()

    acc = correct_count / train_data.shape[0]
    epoch_loss_list.append(loss_sum)
    print('epoch: ', epoch, ' loss: ', loss_sum, ' acc: ', acc)

    # dev开发集验证  每10代进行一次验证
    if epoch % 10 == 0 or epoch == 99:
        print('*'*30)
        model.eval()
        Tp = 0
        Fn = 0
        Fp = 0
        Tn = 0
        # 记录每次标签比较的位置
        result_index = 0
        for data in dev_loader:
            dev_data, dev_data_label = data
            if torch.cuda.is_available():
                dev_data = Variable(dev_data.cuda())
            predict = model(dev_data)
            mask = predict.ge(0.5).double()
            predict_result_format01 = mask.data.cpu().numpy()
            for i in range(predict_result_format01.shape[0]):
                if dev_label[result_index][0] == 1 and predict_result_format01[i][0] == 1:
                    # 修改混淆矩阵
                    # Tp += 1
                    Tn += 1
                    result_index += 1
                elif dev_label[result_index][0] == 1 and predict_result_format01[i][0] == 0:
                    # Fn += 1
                    Fp += 1
                    result_index += 1
                elif dev_label[result_index][0] == 0 and predict_result_format01[i][0] == 1:
                    # Fp += 1
                    Fn += 1
                    result_index += 1
                else:
                    # Tn += 1
                    Tp += 1
                    result_index += 1
        TpFp = Tp + Fp
        TpFn = Tp + Fn
        if TpFp == 0:
            print('TpFp == 0')
            TpFp = 1
        if TpFn == 0:
            print('TpFn == 0')
            TpFn = 1
        Precision = Tp / TpFp
        Recall = Tp / TpFn
        P_R = Precision + Recall
        if P_R == 0:
            print('P_R == 0')
            P_R = np.inf
        temp_F1 = 2 * Precision * Recall / P_R
        print('epoch: ', epoch, ' precision: ', Precision, ' recall: ', Recall, ' F1: ', temp_F1)
        print('*'*30)
        if temp_F1 > F1:
            F1 = temp_F1
            torch.save(model.state_dict(), 'params.pkl')
plt.figure()
plt.plot(np.linspace(0, len(epoch_loss_list), len(epoch_loss_list)), epoch_loss_list, 'b-o')
plt.show()
