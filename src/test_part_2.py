import numpy as np
import blur
from PIL import Image
from matplotlib import pyplot as plt

def tester_1():
    im = Image.open('test_images/512_512_car_02.png')
    image = np.array(im)
    image = image.astype(np.float64) / 255

    plt.figure(1)
    plt.gray()
    plt.axis("off")
    plt.imshow(image)
    
    sz = image.shape[0]
    Al = blur.construct_kernel(image)
    print("finish left kernel construction...")
    Al_inv = blur.trunc_SVD(Al, 511)
    # Al_inv_2 = blur.trunc_SVD(Al, 500)
    print("finish left kernel truncated SVD...")
    Ar = blur.construct_kernel(image)
    print("finish right kernel construction...")
    Ar_inv = blur.trunc_SVD(Ar, 511)
    # Ar_inv_2 = blur.trunc_SVD(Ar, 500)
    print("finish right kernel truncated SVD...")

    blur_image = blur.blur_image(image, Al, Ar)
    deblur_image_1 = blur.deblur_image(blur_image, Al_inv, Ar_inv)
    # deblur_image_2 = blur.deblur_image(blur_image, Al_inv_2, Ar_inv_2)

    print("PSNR:", blur.PSNR(deblur_image_1, image, sz))

    plt.figure(2)
    plt.gray()
    plt.axis("off")
    plt.imshow(blur_image)

    plt.figure(3)
    plt.gray()
    plt.axis("off")
    plt.imshow(deblur_image_1)

    # plt.figure(4)
    # plt.gray()
    # plt.axis('off')
    # plt.imshow(deblur_image_2)

    plt.show()

def tester_2():
    im = Image.open('test_images/512_512_pens.png')
    image = np.array(im)
    image = image.astype(np.float64) / 255

    plt.figure(1)
    plt.gray()
    plt.axis("off")
    plt.imshow(image)

    # sz = image.shape[0]
    A = blur.construct_kernel_2(image, 40)
    print("finish kernel construction...")
    A_inv = blur.trunc_SVD(A, 203)
    print("finish kernel truncated SVD...")

    blur_image = blur.blur_image_2(image, A)
    print("finish 1")
    deblur_image_1 = blur.deblur_image(blur_image, A_inv, A_inv)
    print("finish 2")

    sz = image.shape[0]
    psnr = blur.PSNR(deblur_image_1, image, sz)
    print("PSNR: ", psnr)

    plt.figure(2)
    plt.gray()
    plt.axis("off")
    plt.imshow(blur_image)

    plt.figure(3)
    plt.gray()
    plt.axis("off")
    plt.imshow(deblur_image_1)

    plt.show()

if __name__ == '__main__':
    tester_1()