from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def load_pgm_map(file_path):
    image = Image.open(file_path)
    image = image.convert("L")
    width, height = image.size
    grid_map = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            pixel_value = image.getpixel((x, y))
            if pixel_value > 210:
                grid_map[y][x] = 1
    grid_map = np.array(grid_map)
    return grid_map

def downsample_map(grid_map, shape, downsample_factor):
    height, width = shape[0], shape[1]
    new_height = height // downsample_factor
    new_width = width // downsample_factor
    print(f'new_height:{new_height}, new_width:{new_width}!')

    downsampled_map = np.zeros((new_height, new_width))
    for i in range(new_height):
        for j in range(new_width):
            downsampled_map[i, j] = 1 -np.min(grid_map[i*downsample_factor : min((i+1)*downsample_factor, height), j*downsample_factor : min((j+1)*downsample_factor, width)])
    print(f'downsampled_map shape:{downsampled_map.shape}!')
    return downsampled_map

def get_Map():
    grid_map = load_pgm_map('test.pgm')
    downsampled_map = downsample_map(grid_map, grid_map.shape, 4)
    downsampled_map = downsampled_map[10:45, 0:50]
    desired_shape = (60, 60)
    pad_width_y = desired_shape[0] - downsampled_map.shape[0]
    pad_width_x = desired_shape[1] - downsampled_map.shape[1]
    np.random.seed(0) 
    random_pad_y1 = np.random.randint(0, pad_width_y + 1)
    random_pad_y2 = pad_width_y - random_pad_y1
    random_pad_x1 = np.random.randint(0, pad_width_x + 1)
    random_pad_x2 = pad_width_x - random_pad_x1
    pad_width = ((random_pad_y1, random_pad_y2), (random_pad_x1, random_pad_x2))
    expanded_map = np.pad(downsampled_map, pad_width, mode='constant', constant_values=1)
    return expanded_map

def plot_get_map():
    map_data = get_Map()
    plt.imshow(map_data, cmap='Greys', origin='lower')
    plt.show()

if __name__ == '__main__':
    plot_get_map()
