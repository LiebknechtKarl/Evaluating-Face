

from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np



from PIL import Image

# # 加载两张PNG图片
# # image1 = Image.open("image1.png")
# # image2 = Image.open("image2.png")
# # image1 = Image.open("pic/human_01_mainpoint.png")
# # image1 = Image.open("pic/human_01.png")
# # image2 = Image.open("pic/robot_01.png")


# image1 = Image.open("pic_cu/1_h.jpg")
# image2 = Image.open("pic_cu/1_r.jpg")


# # 获取两张图片的尺寸
# size1 = image1.size
# size2 = image2.size

# # 确定统一缩放的尺寸，可以选择较小的图片尺寸或固定尺寸
# new_size = (min(size1[0], size2[0]), min(size1[1], size2[1]))

# # 将两张图片都缩放到相同的大小
# # image1_resized = image1.resize(new_size, Image.ANTIALIAS)
# # image2_resized = image2.resize(new_size, Image.ANTIALIAS)
# image1_resized = image1.resize(new_size, Image.LANCZOS)
# image2_resized = image2.resize(new_size, Image.LANCZOS)

# # 保存缩放后的图片（可选）
# image1_resized.save("image1_resized.png")
# image2_resized.save("image2_resized.png")

# # 打印结果以确认
# print(f"图片1的新尺寸: {image1_resized.size}")
# print(f"图片2的新尺寸: {image2_resized.size}")


# img1 = np.array(image1_resized)
# img2 = np.array(image2_resized)        # SSIM: 0.7189714552172249


# # 指定较小的窗口大小
# win_size = 3  # 确保窗口大小小于或等于图像的最小尺寸

# # 计算 SSIM
# similarity_index = ssim(img1, img2, win_size=win_size, channel_axis=-1)

# # 输出 SSIM
# print(f"SSIM: {similarity_index}")
# print(f"SSIM: {similarity_index/(new_size[0]*new_size[1])}")


def dif_ssim(pic_path1,pic_path2) :

    image1 = Image.open(pic_path1)
    image2 = Image.open(pic_path2)


    # 获取两张图片的尺寸
    size1 = image1.size
    size2 = image2.size

    # 确定统一缩放的尺寸，可以选择较小的图片尺寸或固定尺寸
    # new_size = (min(size1[0], size2[0]), min(size1[1], size2[1]))
    # new_size = (int(max(size1[0], size2[0],size1[1], size2[1])*0.5), int(max(size1[1], size2[1],size1[1], size2[1])*0.5))
    new_size = (int(max(size1[0], size2[0],size1[1], size2[1])*0.9), int(max(size1[1], size2[1],size1[1], size2[1])*0.9))



    # 将两张图片都缩放到相同的大小
    # image1_resized = image1.resize(new_size, Image.ANTIALIAS)
    # image2_resized = image2.resize(new_size, Image.ANTIALIAS)
    image1_resized = image1.resize(new_size, Image.LANCZOS)
    image2_resized = image2.resize(new_size, Image.LANCZOS)

    # 保存缩放后的图片（可选）
    image1_resized.save("image1_resized.png")
    image2_resized.save("image2_resized.png")

    # 打印结果以确认
    print(f"图片1的新尺寸: {image1_resized.size}")
    print(f"图片2的新尺寸: {image2_resized.size}")


    img1 = np.array(image1_resized)
    img2 = np.array(image2_resized)        # SSIM: 0.7189714552172249


    # 指定较小的窗口大小
    win_size = 3  # 确保窗口大小小于或等于图像的最小尺寸

    # 计算 SSIM
    similarity_index = ssim(img1, img2, win_size=win_size, channel_axis=-1)

    # 输出 SSIM
    print(f"SSIM: {similarity_index}")
    print(f"SSIM: {similarity_index/(new_size[0]*new_size[1])}")

    return similarity_index,similarity_index/(new_size[0]*new_size[1])



def main() :
    # # os.path.dirname(os.path.dirname(__file__)) + '/data/' +
    # dif_ssim('pic_cu/1_h.jpg','pic_cu/1_r.jpg')
    # dif_ssim('pic_cu/2_h.jpg','pic_cu/2_r.jpg')
    # dif_ssim('pic_cu/3_h.jpg','pic_cu/3_r.jpg')
    # dif_ssim('pic_cu/4_h.jpg','pic_cu/4_r.jpg')
    # dif_ssim('pic_cu/5_h.jpg','pic_cu/5_r.jpg')    # 2.9631705930429124e-05

    dif_ssim('data/pic_cu/1_h.jpg','data/pic_cu/1_r.jpg')
    dif_ssim('data/pic_cu/2_h.jpg','data/pic_cu/2_r.jpg')
    dif_ssim('data/pic_cu/3_h.jpg','data/pic_cu/3_r.jpg')
    dif_ssim('data/pic_cu/4_h.jpg','data/pic_cu/4_r.jpg')
    dif_ssim('data/pic_cu/5_h.jpg','data/pic_cu/5_r.jpg')    # 2.9631705930429124e-05
    # pass


# img1 = np.array(Image.open('ab.png'))
# img2 = np.array(Image.open('b.png'))        # SSIM: 


if __name__ == "__main__":
    main()
#     # 指定较小的窗口大小
#     win_size = 3  # 确保窗口大小小于或等于图像的最小尺寸
    
#     # 计算 SSIM
#     similarity_index = ssim(img1, img2, win_size=win_size, channel_axis=-1)
    
#     # 输出 SSIM
#     print(f"SSIM: {similarity_index}")
