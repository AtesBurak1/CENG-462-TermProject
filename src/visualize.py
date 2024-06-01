import matplotlib.pyplot as plt

def displayPlots(images_list):
    plt.figure(figsize=(10, 10))
    
    titles = ['Original Image', 'Noisy Image', 'Proposed Median Filter', 'Normal Median Filter']
    
    for i in range(len(images_list) - 1):
            plt.subplot(2, 2, i + 1)
            plt.imshow(images_list[i], cmap='gray', vmin=0, vmax=255)
            plt.title(f"{titles[i]}\nNoise Level: {images_list[-1]}")
            plt.axis('off')

    plt.show()


