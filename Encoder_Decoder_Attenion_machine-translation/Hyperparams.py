import argparse
def parse_arguments():
    # 设置一个添加参数的解析器
    parse = argparse.ArgumentParser(description='Hyperparams')

    # 往解析器中增加参数
    parse.add_argument('-epochs', type=int, default=100,
                       help='number of epochs for train')
    parse.add_argument('-batch_size', type=int, default=32,
                       help='number of epochs for train')
    parse.add_argument('-lr', type=float, default=0.0001,
                       help='initial learning rate')
    parse.add_argument('-grad_clip', type=float, default=10.0,
                       help='in case of gradient explosion')
    # 返回解析参数的对象
    return parse.parse_args()
