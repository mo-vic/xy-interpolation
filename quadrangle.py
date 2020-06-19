import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt


def solve(vertices, coordinates, axis=0):
    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]
    x3, y3 = vertices[3]
    y, x = coordinates

    if axis == 0:
        a = y - y3
        b = y2 - y3
        c = x - x3
        d = x2 - x3
        e = y2 - y3 - y1 + y0
        f = y3 - y0
        g = x2 - x3 - x1 + x0
        h = x3 - x0
    else:
        a = y - y1
        b = y2 - y1
        c = x - x1
        d = x2 - x1
        e = y3 - y0 - y2 + y1
        f = y0 - y1
        g = x3 - x0 - x2 + x1
        h = x0 - x1

    A = d * e - b * g
    B = a * g - b * h - c * e + d * f
    C = a * h - c * f

    r = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
    r[r < 0] = 0

    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--max", type=float, default=255, help="Maximum value.")
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    # vertices_x = np.random.randint(0, args.max, 4)
    # vertices_y = np.random.randint(0, args.max, 4)
    vertices_x = [10, 224, 180, 30]
    vertices_y = [10, 30, 250, 180]
    plt.subplot(1, 2, 1).scatter(vertices_x, vertices_y)
    plt.subplot(1, 2, 2).scatter(vertices_x, vertices_y)

    vertices = [np.expand_dims(np.stack([vertices_x, vertices_y], axis=1).astype(np.int32), axis=1)]
    canvas = np.zeros((args.max, args.max), dtype=np.uint8)
    cv2.drawContours(canvas, vertices, contourIdx=0, color=1, thickness=-1)

    coordinates = np.where(canvas)

    vertices = np.stack([vertices_x, vertices_y], axis=1)
    # x-axis
    u = solve(vertices, coordinates, axis=0)
    # y-axis
    v = solve(vertices, coordinates, axis=1)

    canvas = canvas.astype(np.float32)
    canvas1 = canvas.copy()
    canvas1[coordinates[0], coordinates[1]] = u
    plt.subplot(1, 2, 1).imshow(canvas1, cmap="jet", vmin=0, vmax=1)
    canvas2 = canvas.copy()
    canvas2[coordinates[0], coordinates[1]] = v
    plt.subplot(1, 2, 2).imshow(canvas2, cmap="jet", vmin=0, vmax=1)

    vertices_x.append(vertices_x[0])
    vertices_y.append(vertices_y[0])
    plt.subplot(1, 2, 1).plot(vertices_x, vertices_y)
    plt.subplot(1, 2, 2).plot(vertices_x, vertices_y)
    plt.show()


if __name__ == '__main__':
    main()
