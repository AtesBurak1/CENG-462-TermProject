import cv2
import numpy as np
from visualize import displayPlots

class EnhancedMedian():
    image_list : list = []
    image_original = None
    image_noisy = None
    image_copy = None
    image_median = None
    padded_img = None
    height : int = 0
    width : int = 0
    pad_size : int = 0
    kernel_size : int = 0
    min_intensity: int = 0
    max_intensity: int = 0 
    non_extreme_count: int = 0
    middle_intensity: int = 0

    def padImage(self):
        self.padded_img = np.pad(self.image_noisy, ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size)), 'reflect')

    def unpaddImage(self):
        self.image_copy = self.image_copy[self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]

    def isNonExtremePixelExist(self, x, y, pad_size):
        x_start, x_end = (x - pad_size//2), (x + pad_size//2)
        y_start, y_end = (y - pad_size//2), (y + pad_size//2)
        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                if self.padded_img[i, j] != 0 and self.padded_img[i, j] != 255:
                    return True
        return False

    def decideKernel(self, x, y):    
        if self.isNonExtremePixelExist(x, y, self.pad_size):
            return self.pad_size
        elif self.isNonExtremePixelExist(x, y, self.pad_size + 2):
            return self.pad_size + 2
        elif self.isNonExtremePixelExist(x, y, self.pad_size + 4):
            return self.pad_size + 4
        return self.pad_size + 6

    def findMinMax(self, x, y):
        min_intensity = 255
        max_intensity = 0
        non_extreme_count = 0
        x_start, x_end = (x - self.kernel_size//2), (x + self.kernel_size//2)
        y_start, y_end = (y - self.kernel_size//2), (y + self.kernel_size//2)
        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                if self.padded_img[i, j] != 0 and self.padded_img[i, j] != 255:
                    min_intensity = min(self.padded_img[i, j], min_intensity)
                    max_intensity = max(self.padded_img[i, j], max_intensity)
                    non_extreme_count = non_extreme_count + 1

        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.non_extreme_count = non_extreme_count


    def findMiddle(self, x, y):
        summation = 0
        x_start, x_end = (x - self.kernel_size//2), (x + self.kernel_size//2)
        y_start, y_end = (y - self.kernel_size//2), (y + self.kernel_size//2)        
        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                if self.padded_img[i, j] != 0 and self.padded_img[i, j] != 255:
                    dist_min = abs(self.padded_img[i, j] - self.min_intensity)
                    dist_max = abs(self.max_intensity - self.padded_img[i, j])
                    summation += self.min_intensity if dist_min > dist_max else self.max_intensity

        return summation//self.non_extreme_count

    def findRealIntensity(self, x, y):
        summation = 0
        x_start, x_end = (x - self.kernel_size//2), (x + self.kernel_size//2)
        y_start, y_end = (y - self.kernel_size//2), (y + self.kernel_size//2)
        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                if self.padded_img[i, j] != 0 and self.padded_img[i, j] != 255:

                    dist_min = abs(self.padded_img[i, j] - self.min_intensity)
                    dist_max = abs(self.max_intensity - self.padded_img[i, j])
                    dist_mid = abs(self.middle_intensity - self.padded_img[i, j])
                    chosen = min(dist_max, min(dist_min, dist_mid))
                    
                    if chosen == dist_min:
                        summation += self.min_intensity
                    elif chosen == dist_max:
                        summation += self.max_intensity
                    elif chosen == dist_mid:
                        summation += self.middle_intensity

        return summation//self.non_extreme_count

    def twoStepMedian(self):
        self.image_copy = np.zeros_like(self.padded_img)
        height, width = self.padded_img.shape
        y_end = height - self.pad_size + 1
        x_end = width - self.pad_size + 1

        for x in range(self.pad_size, x_end):
            for y in range(self.pad_size, y_end):
                if self.padded_img[x, y] != 0 and self.padded_img[x, y] != 255:
                    self.image_copy[x, y] = self.padded_img[x, y]
                    continue
                
                self.kernel_size = self.decideKernel(x, y)
                self.findMinMax(x, y)

                if self.non_extreme_count == 0:
                    continue
                
                self.findMiddle(x, y)
                repaired_intensity = self.findRealIntensity(x, y)
                self.image_copy[x, y] = int(repaired_intensity)
                
    def cvMedian(self):
        self.image_median = cv2.medianBlur(self.image_noisy, 9)

    def enhancedMedian(self, image_original, image_noisy, noise_level):
        self.image_original = cv2.imread(image_original, cv2.IMREAD_GRAYSCALE)
        self.image_noisy = cv2.imread(image_noisy, cv2.IMREAD_GRAYSCALE)
        self.pad_size = 7
        self.padImage()
        self.image_copy = self.padded_img.copy()

        self.twoStepMedian()
        self.cvMedian()
        self.unpaddImage()
        images = [self.image_original, self.image_noisy, self.image_copy, self.image_median, noise_level]
        self.image_list.append(images)


if __name__ == "__main__":
    filter = EnhancedMedian()
    noise_level = [10,20,30,40,45,50,60,70,80,90]
    for i in range(len(noise_level)):
        filter.enhancedMedian('lena_original.png', f'Noisy/lena_noisy_{noise_level[i]}.png', noise_level[i])
        displayPlots(filter.image_list[-1])
