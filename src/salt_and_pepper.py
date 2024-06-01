import numpy as np
import cv2

def add_salt_and_pepper_noise(image, salt_ratio=0.45, pepper_ratio=0.45, num_regions_row=16, num_regions_col=16):
    row, col = image.shape
    noisy_image = np.copy(image)

    region_size_row = row // num_regions_row
    region_size_col = col // num_regions_col

    num_salt_per_region = int((region_size_row * region_size_col) * salt_ratio)
    num_pepper_per_region = int((region_size_row * region_size_col) * pepper_ratio)

    # Divide image to regions to distribute noise equally on every part of the image
    for i in range(num_regions_row):
        for j in range(num_regions_col):
            start_row = i * region_size_row
            end_row = start_row + region_size_row
            start_col = j * region_size_col
            end_col = start_col + region_size_col

            region = noisy_image[start_row:end_row, start_col:end_col]
            region_num_pixels = region.size

            all_coords = np.arange(region_num_pixels)
            np.random.shuffle(all_coords)
            salt_coords = all_coords[:num_salt_per_region]
            salt_coords = np.unravel_index(salt_coords, (region_size_row, region_size_col))
            region[salt_coords] = 255

            pepper_coords = all_coords[num_salt_per_region:num_salt_per_region + num_pepper_per_region]
            pepper_coords = np.unravel_index(pepper_coords, (region_size_row, region_size_col))
            region[pepper_coords] = 0

            noisy_image[start_row:end_row, start_col:end_col] = region

    return noisy_image


image = cv2.imread('lena_original.png', cv2.IMREAD_GRAYSCALE)
noise_list = [10,20,30,40,50,60,70,80,90]
for i in noise_list:
    noisy_image = add_salt_and_pepper_noise(image, i/200, i/200)

    output_path = f'Noisy/lena_noisy_{i}.png'
    # For debug
    #x,y = noisy_image.shape
    #count = 0
    # print(x * y)
    # for i in range(x):
    #     for j in range(y):
    #         if noisy_image[i,j] != 0 and noisy_image[i,j] != 255:
    #             count+=1
    # #print(count)
    cv2.imwrite(output_path,noisy_image)