import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import polygonize, unary_union

def generate_grid(width, height, num_horizontal_lines, num_vertical_lines):
    horizontal_lines = [LineString([(0, y), (width, y)]) for y in np.linspace(0, height, num_horizontal_lines)]
    vertical_lines = [LineString([(x, 0), (x, height)]) for x in np.linspace(0, width, num_vertical_lines)]

    # Perturb the lines
    perturbed_lines = []
    for line in horizontal_lines + vertical_lines:
        new_coords = []
        for point in line.coords[:-1]:
            perturbed_point = (point[0] + random.uniform(-width/20, width/20), point[1] + random.uniform(-height/20, height/20))
            new_coords.append(perturbed_point)
        new_coords.append(line.coords[-1])  # Keep the last point unchanged
        perturbed_lines.append(LineString(new_coords))

    return perturbed_lines

def create_polygons(lines, boundary_polygon):
    lines = unary_union(lines)
    polygons = list(polygonize(lines))
    filtered_polygons = [polygon for polygon in polygons if polygon.within(boundary_polygon)]
    return filtered_polygons

def plot_map(width, height, boundary_polygon, polygons):
    fig, ax = plt.subplots(figsize=(width/10, height/10))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')

    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.fill(x, y, 'k')

    x, y = boundary_polygon.exterior.xy
    ax.plot(x, y, 'r')  # Plot the red border
    plt.show()

width, height = 100, 100
num_horizontal_lines = 10
num_vertical_lines = 10

boundary_points = [(0, 0), (width, 0), (width, height), (0, height)]
perturbed_boundary_points = [(p[0] + random.uniform(-width/20, width/20), p[1] + random.uniform(-height/20, height/20)) for p in boundary_points]
boundary_polygon = Polygon(perturbed_boundary_points)

lines = generate_grid(width, height, num_horizontal_lines, num_vertical_lines)
polygons = create_polygons(lines, boundary_polygon)
plot_map(width, height, boundary_polygon, polygons)
