import cv2
from matplotlib import pyplot as plt
import random
import numpy as np
from skimage import data
import skimage.metrics
from PIL import Image
import PIL


# Test 1
def add_gaussian_noise(X_imgs, output_image="gaussian_noise_attack.jpg"):
    input_image = cv2.imread(X_imgs)
    output = cv2.GaussianBlur(input_image, (5, 5), 0)
    cv2.imwrite('gaussian_noise_attack.jpg', output)


# Test 2
def add_salt_pepper_noise(X_imgs, prob=0.5):
    input_image = cv2.imread(X_imgs)
    output = np.zeros(input_image.shape, np.uint8)
    thres = 1 - prob
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = input_image[i][j]

    cv2.imwrite("salt_pepper_noise_attack.jpg", output)


# Test 3
def add_rotation_test(angle, X_img):
    input_image = cv2.imread(X_img, 0)
    input_image = Image.open(X_img)
    output = input_image.rotate(angle)
    input_image = cv2.imread(X_img)
    output.save("rotation_attack.jpg")
    output = cv2.imread("rotation_attack.jpg")


# Test 4
def rescale_test(X_img):
    input_image = Image.open(X_img)
    # Size of the image in pixels (size of orginal image)
    width, height = input_image.size
    # Setting the points for cropped image
    left = 4
    top = 50
    right = 400
    bottom = 450

    # Cropped image of above dimension
    input_image = input_image.crop((left, top, right, bottom))
    newsize = (512, 512)
    input_image = input_image.resize(newsize)
    input_image.save("cropped_attack.jpg")

    input_image = cv2.imread(X_img)
    output = cv2.imread("cropped_attack.jpg")


# Test 5
def add_poisson_noise(X_img):
    input_image = cv2.imread(X_img, 0)
    noise = np.random.poisson(50, input_image.shape)
    plt.hist(noise.ravel(), 256, [-256, 256]);
    plt.show()
    output = input_image + noise
    output = cv2.imwrite("poisson_noise_attack.jpg", output)


def rotate(X_img):
    for i in range(-90, 90, 10):
        add_rotation_test(i, X_img)
        a = calculate_psnr_nc(X_img, "rotation_attack.jpg")
        print("Degree: ", i, " ---PSNR: ", a[0], "---NC: ", a[1])


def plot_image(X_img, attack_img, title):
    input_image = cv2.imread(X_img)
    output = cv2.imread(attack_img)
    plt.subplot(121), plt.imshow(input_image), plt.title('Original')
    plt.subplot(122), plt.imshow(output), plt.title(title)
    plt.show()



def calculate_psnr_nc(img1="lena.jpg", img2="watermarked_lena.jpg"):
    im1 = cv2.imread(img1, 0)
    im2 = cv2.imread(img2, 0)

    # Compute PSNR over tf.uint8 Tensors.
    psnr = skimage.metrics.peak_signal_noise_ratio(im1, im2)

    # Compute Normalized Correlation
    nc = calculate_normalized_correlation(im1, im2)

    return psnr, nc


def calculate_normalized_correlation(image1, image2):
    numerator = image1.dot(image2)
    numerator = numerator.sum()
    fac = 100000
    denominator1 = pow(image1.dot(image1), 0.5)
    denominator1 = denominator1.sum()
    denominator2 = pow(image2.dot(image2), 0.5)
    denominator2 = denominator2.sum()
    denominator = denominator1 * denominator2
    if denominator == 0:
        return 0
    else:
        return (numerator / denominator) * fac




def print_data(attack, img, X_img):
    print("For ", attack)
    a = calculate_psnr_nc(X_img, img)
    print("PSNR: ", a[0], "NC: ", a[1])


def testing(X_img):
    print_data("Gaussian Noise Attack", "gaussian_noise_attack.jpg", X_img)
    print_data("Salt and Pepper Noise Attack", "salt_pepper_noise_attack.jpg", X_img)
    print_data("Rotation Attack", "rotation_attack.jpg", X_img)
    print_data("Crop Attack", "cropped_attack.jpg", X_img)
    print_data("Poisson Noise Attack", "poisson_noise_attack.jpg", X_img)