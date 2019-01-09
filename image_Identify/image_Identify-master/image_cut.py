import load_unkown_data
from PIL import Image
import os
import time

def cut_train_image():
    train_unqualified_image_dir = 'C:\\Users\\DELL\\Desktop\\不合格照片\\'
    train_qualified_image_dir = 'C:\\Users\\DELL\\Desktop\\合格照片\\'
    picture_unqualified_List = load_unkown_data.eachFile(train_unqualified_image_dir)
    picture_qualified_List = load_unkown_data.eachFile(train_qualified_image_dir)

    if not os.path.exists('./data/'):
        os.makedirs('./data/')
    else:
        pass

    start_time = time.time()
    print('*' * 30)
    print('正在切割训练集图片...')

    # doc_index 是给每张切割成3张图片的图片 单独建立一个文件夹表示
    doc_unqualified_index = 1
    picture_num = 1
    for image in picture_unqualified_List:
        image_file = Image.open(image)

        size = image_file.size

        # 每张图片中子图片的切割范围设定(可调参数)
        region1 = (size[0]*0.15, size[1]*0.3, size[0]*0.36, size[1]*0.75)
        region2 = (size[0]*0.4, size[1]*0.3, size[0]*0.61, size[1]*0.75)
        region3 = (size[0]*0.64, size[1]*0.3, size[0]*0.85, size[1]*0.75)

        # 裁切图片
        cropImg1 = image_file.crop(region1)
        cropImg2 = image_file.crop(region2)
        cropImg3 = image_file.crop(region3)


        # 保存裁切后的子图片
        if not os.path.exists('./data/train_data/unqualified_data/' + str(doc_unqualified_index)):
            os.makedirs('./data/train_data/unqualified_data/' + str(doc_unqualified_index))
        else:
            pass
        cropImg1.save('./data/train_data/unqualified_data/' + str(doc_unqualified_index) + '/'+str(picture_num)+'.png')
        picture_num += 1
        cropImg2.save('./data/train_data/unqualified_data/' + str(doc_unqualified_index) + '/'+str(picture_num)+'.png')
        picture_num += 1
        cropImg3.save('./data/train_data/unqualified_data/' + str(doc_unqualified_index) + '/'+str(picture_num)+'.png')
        picture_num += 1

        doc_unqualified_index += 1
    # *****************************************************

    doc_qualified_index = 1
    for image in picture_qualified_List:
        image_file = Image.open(image)

        size = image_file.size

        # 每张图片中子图片的切割范围设定(可调参数)
        region1 = (size[0]*0.15, size[1]*0.3, size[0]*0.36, size[1]*0.75)
        region2 = (size[0]*0.4, size[1]*0.3, size[0]*0.61, size[1]*0.75)
        region3 = (size[0]*0.64, size[1]*0.3, size[0]*0.85, size[1]*0.75)

        # 裁切图片
        cropImg1 = image_file.crop(region1)
        cropImg2 = image_file.crop(region2)
        cropImg3 = image_file.crop(region3)


        # 保存裁切后的子图片
        cropImg1.save('./data/train_data/qualified_data/' + str(doc_qualified_index) + '.png')
        doc_qualified_index += 1
        cropImg2.save('./data/train_data/qualified_data/' + str(doc_qualified_index) + '.png')
        doc_qualified_index += 1
        cropImg3.save('./data/train_data/qualified_data/' + str(doc_qualified_index) + '.png')
        doc_qualified_index += 1

        # if not os.path.exists('./data/train_data/qualified_data/' + str(doc_qualified_index)):
        #     os.makedirs('./data/train_data/qualified_data/' + str(doc_qualified_index))
        # else:
        #     pass
        # cropImg1.save('./data/train_data/qualified_data/' + str(doc_qualified_index) + '/cut1.png')
        # cropImg2.save('./data/train_data/qualified_data/' + str(doc_qualified_index) + '/cut2.png')
        # cropImg3.save('./data/train_data/qualified_data/' + str(doc_qualified_index) + '/cut3.png')


    end_time = time.time()
    use_time = end_time - start_time
    print('切割训练集图片完成...')
    print('用时 ' + str(use_time) + ' 秒')
    print('*'*30)

def cut_test_image(file_path):
    # 训练集在线下，可以不用传入文件路径参数， 但是测试集，没有指定路径，必须由用户传入
    picture_test_List = load_unkown_data.eachFile(file_path)

    start_time = time.time()
    print('*' * 30)
    print('正在切割测试集图片...')

    # doc_index 是给每张切割成3张图片的图片 单独建立一个文件夹表示
    doc_test_index = 1

    for image in picture_test_List:
        # drop out
        iamge_num = image[41:-4]


        image_file = Image.open(image)

        size = image_file.size

        # 每张图片中子图片的切割范围设定(可调参数)
        # x,y左右两边之差，必须要一样，否则，模型参数无法进行定义
        region1 = (size[0] * 0.15, size[1] * 0.3, size[0] * 0.36, size[1]*0.75)
        region2 = (size[0] * 0.4, size[1] * 0.3, size[0] * 0.61, size[1]*0.75)
        region3 = (size[0] * 0.64, size[1] * 0.3, size[0] * 0.85, size[1]*0.75)

        # 裁切图片
        cropImg1 = image_file.crop(region1)
        cropImg2 = image_file.crop(region2)
        cropImg3 = image_file.crop(region3)

        # 保存裁切后的子图片
        if not os.path.exists('./test_data/' + str(iamge_num)):
            os.makedirs('./test_data/' + str(iamge_num))
        else:
            pass
        cropImg1.save('./test_data/' + str(iamge_num) + '/cut1.png')
        cropImg2.save('./test_data/' + str(iamge_num) + '/cut2.png')
        cropImg3.save('./test_data/' + str(iamge_num) + '/cut3.png')

        doc_test_index += 1

    end_time = time.time()
    use_time = end_time - start_time
    print('切割测试集图片完成...')
    print('用时 ' + str(use_time) + ' 秒')
    print('*'*30)

