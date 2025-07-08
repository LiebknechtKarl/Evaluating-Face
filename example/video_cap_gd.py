

import numpy as np




def bs2gd(random_array,   td = 0.5 ) :


    # 输出随机数组
    print(random_array,random_array.shape)


    # 沿第一个维度计算每一行之间的差值
    row_diff = np.abs(np.diff(random_array, axis=0))

    print(row_diff,row_diff.shape)

    # 第二次求差值
    row_diff_ = np.abs(np.diff(row_diff, axis=0))


    # 将大于 k 0.5 的值变为 1，其余变为 0
    binary_array = np.where(row_diff_ > td, 1, 0)


    print(binary_array,binary_array.shape)

    mean_ser = np.mean(random_array, axis=1)

    print(mean_ser,mean_ser.shape)

    gd = 1 - np.mean(mean_ser, axis=0)

    print(gd)
    return gd


def main():
    # 生成 100*52 维随机数组
    # random_array = np.random.rand(100, 52)
    # random_array = np.load("Gd/bs.npy")
    random_array = np.load("model/Gd/bs.npy")

    

    print(bs2gd(random_array))
    


if __name__ == "__main__":
    main()     # 0.9278496526362477



