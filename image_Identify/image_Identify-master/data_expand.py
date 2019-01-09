import data_reforcement
import data_transfer
import load_unkown_data

# 扩大正样例数据集
def expand_data(file_initial_path):
    movie_name = load_unkown_data.eachFile(file_initial_path)
    name_new_image = 201
    # 旋转
    for temp in movie_name:
        print(name_new_image)
        image_array = data_reforcement.rotate(data_transfer.ImageToMatrix(temp))
        data_transfer.MatrixToImage(image_array).save('data/train_data/unqualified_data/' + str(name_new_image) + '.png')
        name_new_image += 1

    # 增亮
    for temp in movie_name:
        print(name_new_image)

        image_array = data_reforcement.darker(data_transfer.ImageToMatrix(temp))
        data_transfer.MatrixToImage(image_array).save('data/train_data/unqualified_data/' + str(name_new_image) + '.png')
        name_new_image += 1

    for temp in movie_name:
        print(name_new_image)

        image_array = data_reforcement.addGaussianNoise(data_transfer.ImageToMatrix(temp), 0.8)
        data_transfer.MatrixToImage(image_array).save('data/train_data/unqualified_data/' + str(name_new_image) + '.png')
        name_new_image += 1

    for temp in movie_name:
        print(name_new_image)

        image_array = data_reforcement.SaltAndPepper(data_transfer.ImageToMatrix(temp), 0.8)
        data_transfer.MatrixToImage(image_array).save('data/train_data/unqualified_data/' + str(name_new_image) + '.png')
        name_new_image += 1

# expand_data('data/train_data/unqualified_data')