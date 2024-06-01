import numpy as np
import cv2

def add_salt_and_pepper_noise(image, salt_ratio=0.45, pepper_ratio=0.45):
    row, col = image.shape
    noisy_image = np.copy(image)

    num_pixels = row * col

    num_salt = int(num_pixels * salt_ratio)
    num_pepper = int(num_pixels * pepper_ratio)

    all_coords = np.random.permutation(num_pixels)

    salt_coords = all_coords[:num_salt]
    salt_coords = np.unravel_index(salt_coords, (row, col))
    noisy_image[salt_coords] = 255

    # Pepper coordinates
    pepper_coords = all_coords[num_salt:num_salt + num_pepper]
    pepper_coords = np.unravel_index(pepper_coords, (row, col))
    noisy_image[pepper_coords] = 0

    return noisy_image


image = cv2.imread('lena_original.png', cv2.IMREAD_GRAYSCALE)
noisy_image = add_salt_and_pepper_noise(image)

output_path = 'Noisy/lena_noisy_91.png'
cv2.imwrite(output_path,noisy_image)