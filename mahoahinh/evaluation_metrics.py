import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
from skimage.feature import graycomatrix, graycoprops
import hashlib
from skimage import io
import cv2
import numpy as np

def correlation_coefficient(input_image_path, output_image_path):
    img1 = cv2.imread(input_image_path)
    img2 = cv2.imread(output_image_path)
    # Chuyển đổi hình ảnh thành dạng grayscale
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Tính toán hệ số tương quan
    mean_img1 = np.mean(gray_img1)
    mean_img2 = np.mean(gray_img2)
    correlation = np.mean((gray_img1 - mean_img1) * (gray_img2 - mean_img2)) / (np.std(gray_img1) * np.std(gray_img2))

    return correlation

def vertical_correlation(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray_img.shape
    # Tính giá trị trung bình của pixel
    mean_img = np.mean(gray_img)
    # Tính hệ số tương quan dọc giữa các pixel liền kề
    sum_numerator = 0
    sum_denominator1 = 0
    sum_denominator2 = 0
    for i in range(H - 1):
        for j in range(W):
            numerator = (gray_img[i, j] - mean_img) * (gray_img[i + 1, j] - mean_img)
            denominator1 = (gray_img[i, j] - mean_img) ** 2
            denominator2 = (gray_img[i + 1, j] - mean_img) ** 2
            sum_numerator += numerator
            sum_denominator1 += denominator1
            sum_denominator2 += denominator2
    correlation = sum_numerator / (np.sqrt(sum_denominator1) * np.sqrt(sum_denominator2))
    return correlation

def horizontal_correlation(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray_img.shape
    # Tính giá trị trung bình của pixel
    mean_img = np.mean(gray_img)
    # Tính hệ số tương quan dọc giữa các pixel liền kề
    sum_numerator = 0
    sum_denominator1 = 0
    sum_denominator2 = 0
    for i in range(H ):
        for j in range(W - 1):
            numerator = (gray_img[i, j] - mean_img) * (gray_img[i , j + 1] - mean_img)
            denominator1 = (gray_img[i, j] - mean_img) ** 2
            denominator2 = (gray_img[i, j + 1] - mean_img) ** 2
            sum_numerator += numerator
            sum_denominator1 += denominator1
            sum_denominator2 += denominator2
    correlation = sum_numerator / (np.sqrt(sum_denominator1) * np.sqrt(sum_denominator2))
    return correlation

def diagonal_correlation(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray_img.shape
    # Tính giá trị trung bình của pixel
    mean_img = np.mean(gray_img)
    # Tính hệ số tương quan dọc giữa các pixel liền kề
    sum_numerator = 0
    sum_denominator1 = 0
    sum_denominator2 = 0
    for i in range(H - 1):
        for j in range(W - 1):
            numerator = (gray_img[i, j] - mean_img) * (gray_img[i + 1, j + 1] - mean_img)
            denominator1 = (gray_img[i, j] - mean_img) ** 2
            denominator2 = (gray_img[i + 1, j + 1] - mean_img) ** 2
            sum_numerator += numerator
            sum_denominator1 += denominator1
            sum_denominator2 += denominator2
    correlation = sum_numerator / (np.sqrt(sum_denominator1) * np.sqrt(sum_denominator2))
    return correlation


def plot_pixel_value_histogram(image, title):
    # Chuyển đổi hình ảnh thành mảng numpy
    image_array = np.array(image)

    # Chuyển đổi mảng 2D thành mảng 1D
    pixel_values = image_array.flatten()

    # Vẽ histogram
    plt.figure(figsize=(10, 5))
    plt.hist(pixel_values, bins=256, range=(0, 255), color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Giá trị pixel')
    plt.ylabel('Số lượng pixel')
    plt.show()

def image_entropy(image_path):
    # Đọc ảnh và chuyển đổi sang ảnh xám
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Tính histogram của ảnh
    hist = cv2.calcHist([img], [0], None, [256], [0,256])

    # Chuẩn hóa histogram để tính xác suất
    hist /= np.sum(hist)

    # Tính entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Thêm một số nhỏ để tránh trường hợp log(0)

    return entropy

def mse(image1_path, image2_path):
    # Load images
    image1 = np.array(cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE), dtype=np.int32).flatten()
    image2 = np.array(cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE), dtype=np.int32).flatten()
    print(image1)
    print(image2)
    # Ensure the images have the same dimensions
    assert image1.shape == image2.shape, "Images must have the same dimensions"

    # Calculate MSE
    mse = np.mean((image1 - image2) ** 2)
    print((image1 - image2)** 2)
    return mse

def calculate_psnr(max_value, mse):
    return 10 * np.log10((max_value ** 2) / mse)


def calculate_NPCR(image1_path, image2_path):
    # Chuyển đổi hình ảnh thành mảng numpy
    image1_array = np.array(cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)).flatten()
    image2_array = np.array(cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)).flatten()

    different_pixels = np.sum(image1_array != image2_array)

    total_pixels = image1_array.size

    NPCR = (different_pixels / total_pixels) * 100

    return NPCR