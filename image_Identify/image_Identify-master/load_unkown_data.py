# encoding: utf-8

import os
# # 加载数据集不定文件 遍历指定目录，显示目录下的所有文件名

# 因为是递归进行处理，这个返回列表需放在函数外面

def eachFile(filepath):
    # 如果这个文件夹下还有文件夹，那么这个循环输出文件的列表是有bug的
    picture_list = []
    path_dir = os.listdir(filepath)
    for allDir in path_dir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        picture_list.append(child)
    return picture_list
