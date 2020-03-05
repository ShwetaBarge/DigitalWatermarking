# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:34:09 2020

@author: HP
"""
import cv2
from matplotlib import pyplot as plt
import random
import numpy as np
from skimage import data
from skimage.transform import rotate
from skimage.transform import rescale
import skimage.metrics
from PIL import Image 
import PIL 


#Test 1
def add_gaussian_noise(X_imgs):
    input_image = cv2.imread(X_imgs)
    output = cv2.GaussianBlur(input_image, (5,5), 0)
    cv2.imwrite('gaussian_noise_attack.jpg', output)

#Test 2
def add_salt_pepper_noise(X_imgs, prob = 0.05):
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
    
#Test 3
def add_rotation_test(angle, X_img):
    input_image = cv2.imread(X_img,0)
    input_image = Image.open(X_img) 
    output = input_image.rotate(angle) 
    input_image = cv2.imread(X_img)
    output.save("rotation_attack.jpg")
    output = cv2.imread("rotation_attack.jpg")
    
#Test 4  
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

#Test 5
def add_poisson_noise(X_img):
    input_image = cv2.imread(X_img, 0)
    noise = np.random.poisson(50, input_image.shape)
    plt.hist(noise.ravel(),256,[-256,256]);plt.show()
    output = input_image + noise
    output = cv2.imwrite("poisson_noise_attack.jpg", output)

def rotate(X_img):
    for i in range(-90,90,10):
        add_rotation_test(i, X_img)
        a = psnr_cal(X_img, "rotation_attack.jpg")
        print("Degree: ", i, " ---PSNR: ", a[0], "---NC: ", a[1])
       
def plot_image(X_img, attack_img, title):
    input_image = cv2.imread(X_img)
    output = cv2.imread(attack_img)
    plt.subplot(121),plt.imshow(input_image),plt.title('Original')
    plt.subplot(122),plt.imshow(output), plt.title(title)
    plt.show()    
    
def psnr_cal(img1="lena.jpg" , img2="watermarked_lena.jpg"):
    im1 = cv2.imread(img1,0)
    im2 = cv2.imread(img2,0)
    # Compute PSNR over tf.uint8 Tensors.
    psnr = skimage.metrics.peak_signal_noise_ratio(im1, im2)
    numerator = im1.dot(im2)
    numerator = numerator.sum()
    fac = 100000
    deno1 = pow(im1.dot(im1), 0.5)
    deno1 = deno1.sum()
    deno2 = pow(im2.dot(im2), 0.5)
    deno2 = deno2.sum()
    deno = deno1 * deno2
    if deno == 0:
        answer = 0
    else:
        answer = (numerator/ deno)*fac
    return psnr, answer

def print_data(attack, img, X_img):
    print("For ", attack)
    a = psnr_cal(X_img, img)
    print("PSNR: ", a[0], "NC: ", a[1])
    
def testing(X_img):
    print_data("Gaussian Noise Attack", "gaussian_noise_attack.jpg", X_img)
    print_data("Salt and Pepper Noise Attack", "salt_pepper_noise_attack.jpg", X_img)
    print_data("Rotation Attack", "rotation_attack.jpg", X_img)
    print_data("Crop Attack", "cropped_attack.jpg", X_img)
    print_data("Poisson Noise Attack", "poisson_noise_attack.jpg", X_img)