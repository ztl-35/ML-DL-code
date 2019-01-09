# 此文件用来将生成的图片和数值型数组数据的转换
from PIL import Image
import numpy as np
import time

def ImageToMatrix(file_path):
    # 读取图片
    im = Image.open(file_path)
    width, height = im.size
    # 灰度图像打开模式设置
    im = im.convert("L")
    data = im.getdata()
    # 这里不能除 255 否则图片全黑
    data = np.matrix(data, dtype='float')
    new_data = np.reshape(data, (height, width))
    return new_data

def MatrixToImage(data):
    # 传入的是二维数组 np.array
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


# 上面都是处理单一的数据为矩阵，此函数把所有的数据全部加载进来，形成图片矩阵
# 下面三个函数分别对应原始数据 8：1：1 进行切分（train, dev, test）
# 此为训练集
def process_train_data2Matrix():
    start_time = time.time()
    print('*'*30)
    print('将所有图片转换成矩阵处理...')
    # 图片的高度和宽度是由切割图片的范围决定的。  现在的比例是:922*514
    train_qualified_matrix = np.zeros((800, 922, 514))
    train_unqualified_matrix = np.zeros((800, 922, 514))

    # 处理合格矩阵
    for i in range(800):
        data1 = ImageToMatrix('data/train_data/qualified_data/'+str(i+1)+'.png')
        train_qualified_matrix[i, :, :] = data1
    # 处理不合格矩阵
    for i in range(800):
        data1 = ImageToMatrix('data/train_data/unqualified_data/' + str(i + 1) + '.png')
        train_unqualified_matrix[i, :, :] = data1
    print('所有图片转换成矩阵处理完毕...')
    end_time = time.time()
    use_time = end_time - start_time
    print('用时 ', use_time, ' 秒')
    print('*'*30)
    return train_qualified_matrix, train_unqualified_matrix

def process_dev_data2Matrix():
    start_time = time.time()
    print('*' * 30)
    print('将所有开发集图片转换成矩阵处理...')
    # 图片的高度和宽度是由切割图片的范围决定的。  现在的比例是:922*514
    dev_qualified_matrix = np.zeros((200, 922, 514))
    dev_unqualified_matrix = np.zeros((200, 922, 514))

    # 处理合格矩阵
    for i in range(200):
        data1 = ImageToMatrix('data/train_data/qualified_data/' + str(800 + i + 1) + '.png')
        dev_qualified_matrix[i, :, :] = data1

    # 处理不合格矩阵
    for i in range(200):
        data1 = ImageToMatrix('data/train_data/unqualified_data/' + str(800 + i + 1) + '.png')
        dev_unqualified_matrix[i, :, :] = data1
    print('所有图片转换成矩阵处理完毕...')
    end_time = time.time()
    use_time = end_time - start_time
    print('用时 ', use_time, ' 秒')
    print('*' * 30)
    return dev_qualified_matrix, dev_unqualified_matrix

def process_test_data2Matrix(test_data_path_list):
    test_data_size = len(test_data_path_list)
    start_time = time.time()
    print('*' * 30)
    print('将所有测试图片转换成矩阵处理...')
    # 图片的高度和宽度是由切割图片的范围决定的。  现在的比例是:922*514
    test_data_matrix = np.zeros((test_data_size*3, 922, 514))
    test_data_label_list = np.zeros((test_data_size*3, 1))
    # 处理合格矩阵
    for path in test_data_path_list:
        print(path)
        doc_index = int(path[45:])
        if doc_index > 50:
            temp_doc_index = (doc_index - 1) * 3
            test_data_label_list[temp_doc_index][0] = 0
            test_data_label_list[(temp_doc_index + 1)][0] = 0
            test_data_label_list[(temp_doc_index + 2)][0] = 0

        else:
            temp_doc_index = (doc_index - 1) * 3
            test_data_label_list[temp_doc_index][0] = 1
            test_data_label_list[(temp_doc_index + 1)][0] = 1
            test_data_label_list[(temp_doc_index + 2)][0] = 1

        start_index = (doc_index - 1)*3
        for i in range(3):
            temp_index = start_index + i
            temp_path = path + '/cut' + str(i+1) + '.png'
            data = ImageToMatrix(temp_path)
            test_data_matrix[temp_index, :, :] = data

    print('所有图片转换成矩阵处理完毕...')
    end_time = time.time()
    use_time = end_time - start_time
    print('用时 ', use_time, ' 秒')
    print('*' * 30)
    return test_data_matrix, test_data_label_list

# 每张图片的大小： 922 * 514
