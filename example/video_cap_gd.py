import numpy as np

def bs2gd(random_array,   td = 0.5 ) :
    print(random_array,random_array.shape)                  # 输出随机数组

    row_diff = np.abs(np.diff(random_array, axis=0))        # 沿第一个维度计算每一行之间的差值
    print(row_diff,row_diff.shape)
    
    row_diff_ = np.abs(np.diff(row_diff, axis=0))           # 第二次求差值
    binary_array = np.where(row_diff_ > td, 1, 0)           # 将大于 k 0.5 的值变为 1，其余变为 0
    print(binary_array,binary_array.shape)

    mean_ser = np.mean(random_array, axis=1)
    print(mean_ser,mean_ser.shape)

    gd = 1 - np.mean(mean_ser, axis=0)
    print(gd)
    return gd

if __name__ == "__main__":
    random_array = np.load("model/Gd/bs.npy")               # 生成 100*52 维随机数组
    print(bs2gd(random_array))



