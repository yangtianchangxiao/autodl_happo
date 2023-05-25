import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Set the map size
map_size = 100

# Create an empty map
map_data = np.zeros((map_size, map_size))

# Define a set of random points for the boundary
boundary_points = np.array([
    [0, 0],
    [20, 50],
    [60, 70],
    [80, 30],
    [50, 10],
    [0, 0]
])

# Create boundary lines
for i in range(len(boundary_points) - 1):
    p1 = boundary_points[i]
    p2 = boundary_points[i + 1]

    # If you want curved lines, you can use interpolation
    f = interp1d([p1[0], p2[0]], [p1[1], p2[1]], kind='quadratic')
    x_range = np.linspace(p1[0], p2[0], num=50)
    y_range = f(x_range)

    for x, y in zip(x_range, y_range):
        map_data[int(x), int(y)] = 1

# Plot the map
plt.imshow(map_data, cmap='gray', origin='lower')
plt.show()
