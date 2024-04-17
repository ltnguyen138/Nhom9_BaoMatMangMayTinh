import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
from skimage.feature import graycomatrix, graycoprops
import hashlib
from skimage import io
import evaluation_metrics as evaluation
import cv2
import numpy as np

current_dir = os.path.dirname(__file__)
img_dir = os.path.join(current_dir, 'img')

input_image_path_a = os.path.join(img_dir, 'hinha.png')
input_image_path_b = os.path.join(img_dir, 'hinhb.jpg')

input_image = Image.open(input_image_path_a)
input_image = Image.open(input_image_path_b)
pw = "ax"
# Chia ảnh thành các khối có kích thước cố định
def split_into_blocks(image):
    width, height = image.size
    phan_chieu_dai = width // 32
    phan_chieu_rong = height // 32
    blocks = []
    for i in range(32):
        row = []  
        for j in range(32):
            left = i * phan_chieu_dai
            top = j * phan_chieu_rong
            right = left + phan_chieu_dai
            bottom = top + phan_chieu_rong        
            phan_anh = image.crop((left, top, right, bottom))
            row.append(phan_anh)
        
        blocks.append(row)
    return blocks

def lower_upper_triangular(blocks, img):
     
    for i in range(31):
        for j in range(i + 1, 32):
            blocks[i][j], blocks[j][i] = blocks[j][i], blocks[i][j]
    width, height = img.size
    width_block, height_block =  blocks[1][1].size        
    sorted_img = Image.new("RGB", (width, height))
    for i in range(32):
        for j in range(32):
            sorted_img.paste(blocks[i][j], (i * width_block, j * height_block))
    return sorted_img

def decimal_to_binary(decimal):
    return bin(decimal)[2:].zfill(8)

def binary_to_decimal(binary):
    return int(binary, 2)

def left_rotate(binary_str, shift):
    return binary_str[shift:] + binary_str[:shift]


def bit_rotation_encrypt(image, password):
    
    height, width = image.size
    image_array = np.array(image, dtype=np.uint8)
    print(height)
    print(width)
    print(image_array.shape)
    encrypted_array = np.zeros((width,height , 3), dtype=np.uint8)
    L = len(password)
    LR = L % 8
    if LR == 7:
        LR = 6

    LR1 = (LR + 1) % 8 

    for i in range(width):
        for j in range(height):
            
            red, green, blue = image_array[i, j]
            
            red_binary = decimal_to_binary(red)
            green_binary = decimal_to_binary(green)
            blue_binary = decimal_to_binary(blue)
            
            rotated_red = ""
            rotated_green = ""
            rotated_blue = ""
            if (i+j)%2 == 0 :
                rotated_red = left_rotate(red_binary, LR1)
                rotated_green = left_rotate(green_binary, LR1)
                rotated_blue = left_rotate(blue_binary, LR1)
            
            else:
                rotated_red = left_rotate(red_binary, LR)
                rotated_green = left_rotate(green_binary, LR)
                rotated_blue = left_rotate(blue_binary, LR)
            encrypted_array[i, j] = [
                binary_to_decimal(rotated_red),
                binary_to_decimal(rotated_green),
                binary_to_decimal(rotated_blue)
            ]

    encrypted_img = Image.fromarray(encrypted_array, 'RGB')
          
    return encrypted_img

def bit_rotation_encrypt_revere(image, password):
    
    height, width = image.size
    image_array = np.array(image, dtype=np.uint8)
    
    encrypted_array = np.zeros((width,height , 3), dtype=np.uint8)

    L = len(password)
    LR = L % 8
    if LR == 7:
        LR = 6

    LR1 = (LR + 1) % 8 
   
    for i in range(width):
        for j in range(height):

            red, green, blue = image_array[i, j]
            
            red_binary = decimal_to_binary(red)
            green_binary = decimal_to_binary(green)
            blue_binary = decimal_to_binary(blue)
            rotated_red = ""
            rotated_green = ""
            rotated_blue = ""
            if (i+j)%2 == 0 :
                rotated_red = left_rotate(red_binary, -LR1)
                rotated_green = left_rotate(green_binary, -LR1)
                rotated_blue = left_rotate(blue_binary, -LR1)
                                       
            else:
                rotated_red = left_rotate(red_binary, -LR)
                rotated_green = left_rotate(green_binary, -LR)
                rotated_blue = left_rotate(blue_binary, -LR)
                
            encrypted_array[i, j] = [
                binary_to_decimal(rotated_red),
                binary_to_decimal(rotated_green),
                binary_to_decimal(rotated_blue)              
            ]
            
    decrypted_img = Image.fromarray(encrypted_array, 'RGB')
    return decrypted_img

def encrypt_image(image_path, password):

    input_image = Image.open(image_path)
    encrypt_blocks = split_into_blocks(input_image)
    encrypt_image = lower_upper_triangular(encrypt_blocks, input_image)
    encrypt_img = bit_rotation_encrypt(encrypt_image, password)
    return encrypt_img

encrypt_image_a = encrypt_image(input_image_path_a, pw)
encrypt_image_b = encrypt_image(input_image_path_b, pw)

encrypt_image_path_a = os.path.join(img_dir, 'encrypt_image_a.png')
encrypt_image_a.save(encrypt_image_path_a)
encrypt_image_path_b = os.path.join(img_dir, 'encrypt_image_b.png')
encrypt_image_b.save(encrypt_image_path_b)


def decryption_image(image_path,  password):
    
    encrypt_image = Image.open(image_path)
    decryption_img = bit_rotation_encrypt_revere(encrypt_image, password)
    decryption_blocks = split_into_blocks(decryption_img)
    decryption_image = lower_upper_triangular(decryption_blocks, decryption_img)

    return decryption_image





# Lưu hình ảnh đã mã hóa

print("--------")


decryption_image_a = decryption_image(encrypt_image_path_a, pw)
decryption_image_b = decryption_image(encrypt_image_path_b, pw)

decryption_image_path_a = os.path.join(img_dir, 'decryption_image_a.png')
decryption_image_a.save(decryption_image_path_a)

decryption_image_path_b = os.path.join(img_dir, 'decryption_image_b.png')
decryption_image_b.save(decryption_image_path_b)

# Hệ số tương quan
corr_coeff_a = evaluation.correlation_coefficient(input_image_path_a, encrypt_image_path_a)
print("Hệ số tương quan giữa hình gốc và hình mã hóa a:", corr_coeff_a)
corr_coeff_b = evaluation.correlation_coefficient(input_image_path_b, encrypt_image_path_b)
print("Hệ số tương quan giữa hình gốc và hình mã hóa b:", corr_coeff_b)


# Tính hệ số tương quan dọc giữa các pixel liền kề trong hình ảnh
corr_coeff_e_a = evaluation.vertical_correlation(input_image_path_a)
corr_coeff_e_b = evaluation.vertical_correlation(input_image_path_b)
corr_coeff_a = evaluation.vertical_correlation(encrypt_image_path_a)
corr_coeff_b = evaluation.vertical_correlation(encrypt_image_path_b)
print("Hệ số tương quan dọc trong ảnh gốc a :", corr_coeff_e_a)
print("Hệ số tương quan dọc trong ảnh gốc b :", corr_coeff_e_b)
print("Hệ số tương quan dọc trong ảnh mã hóa a :", corr_coeff_a)
print("Hệ số tương quan dọc trong ảnh mã hóa b :", corr_coeff_b)

horizontal_correlation_e_a = evaluation.horizontal_correlation(input_image_path_a)
horizontal_correlation_e_b = evaluation.horizontal_correlation(input_image_path_b)
horizontal_correlation_a = evaluation.horizontal_correlation(encrypt_image_path_a)
horizontal_correlation_b = evaluation.horizontal_correlation(encrypt_image_path_b)
print("Hệ số tương quan ngang trong ảnh gốc a :", horizontal_correlation_e_a)
print("Hệ số tương quan ngang trong ảnh gốc b :", horizontal_correlation_e_b)
print("Hệ số tương quan ngang trong ảnh mã hóa a :", horizontal_correlation_a)
print("Hệ số tương quan ngang trong ảnh mã hóa b :", horizontal_correlation_b)

# Đường dẫn đến hình ảnh
# Gọi hàm để vẽ histogram
# plot_pixel_value_histogram(input_image,"s")
# plot_pixel_value_histogram(encoded_image_m,"s")

# Tính entropy của hình ảnh
entropy_value_a = evaluation.image_entropy(encrypt_image_path_a)
print("Entropy của hình ảnh mã hóa a:", entropy_value_a)
entropy_value_b = evaluation.image_entropy(encrypt_image_path_b)
print("Entropy của hình ảnh mã hóa b:", entropy_value_b)


# Example usage
# mse_value = mse(input_image_path, output_image_path)
# print("Mean Square Error (MSE):", mse_value)


NPCR_a = evaluation.calculate_NPCR(input_image_path_a, encrypt_image_path_a)
print("Tỉ lệ NPCR giữa ảnh gốc và ảnh mã hóa a:", NPCR_a)

NPCR_b = evaluation.calculate_NPCR(input_image_path_b, encrypt_image_path_b)
print("Tỉ lệ NPCR giữa ảnh gốc và ảnh mã hóa b:", NPCR_b)