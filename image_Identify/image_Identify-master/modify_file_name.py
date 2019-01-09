import os
def modify_name(file_path):
    movie_name = os.listdir(file_path)
    count = 51
    for temp in movie_name:
        new_name = str(count) + '.jpg'

        os.rename(file_path + temp, file_path + new_name)
        count += 1
modify_name('/home/gpu-105/PycharmProjects/test/测试文件夹1/不合格/')