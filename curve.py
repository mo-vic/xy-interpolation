'''
Description: Cubic bezier curve interpolation
Author: movic
Date:2020-06-19
'''

import argparse

import bezier
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--max", type=float, default=512, help="Maximum value.")
    parser.add_argument("--num", type=int, default=512, help="Number of points.")
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)

    ctrl_pt1 = np.array([[32, 80, 8, 48], [32, 64, 384, 448]])
    curve1 = bezier.Curve(ctrl_pt1, degree=3)
    curve1_x, curve1_y = curve1.evaluate_multi(np.linspace(0.0, 1.0, args.num))
    plt.subplot(1, 2, 1).plot(curve1_x, curve1_y)
    plt.subplot(1, 2, 2).plot(curve1_x, curve1_y)
    ctrl_pt2 = np.array([[48, 256, 320, 448], [448, 480, 352, 480]])
    curve2 = bezier.Curve(ctrl_pt2, degree=3)
    curve2_x, curve2_y = curve2.evaluate_multi(np.linspace(0.0, 1.0, args.num))
    plt.subplot(1, 2, 1).plot(curve2_x, curve2_y)
    plt.subplot(1, 2, 2).plot(curve2_x, curve2_y)
    ctrl_pt3 = np.array([[32, 256, 320, 464], [32, 64, 224, 48]])
    curve3 = bezier.Curve(ctrl_pt3, degree=3)
    curve3_x, curve3_y = curve3.evaluate_multi(np.linspace(0.0, 1.0, args.num))
    plt.subplot(1, 2, 1).plot(curve3_x, curve3_y)
    plt.subplot(1, 2, 2).plot(curve3_x, curve3_y)
    ctrl_pt4 = np.array([[464, 464, 320, 448], [48, 64, 224, 480]])
    curve4 = bezier.Curve(ctrl_pt4, degree=3)
    curve4_x, curve4_y = curve4.evaluate_multi(np.linspace(0.0, 1.0, args.num))
    plt.subplot(1, 2, 1).plot(curve4_x, curve4_y)
    plt.subplot(1, 2, 2).plot(curve4_x, curve4_y)

    canvas = np.zeros((args.max, args.max), dtype=np.float32)
    ts = np.linspace(0.0, 1.0, 2048)
    for t in ts:
        pt1 = curve2.evaluate(t).squeeze()
        pt2 = curve3.evaluate(t).squeeze()
        ctrl_1 = (1 - t) * ctrl_pt1[:, 1] + t * ctrl_pt4[:, 1]
        ctrl_2 = (1 - t) * ctrl_pt1[:, 2] + t * ctrl_pt4[:, 2]
        ctrl = list(zip(pt1, ctrl_2, ctrl_1, pt2))
        curve = bezier.Curve(ctrl, degree=3)
        coordinates = curve.evaluate_multi(np.linspace(0.0, 1.0, args.num)).astype(np.int32)
        canvas[coordinates[1], coordinates[0]] = t

    plt.subplot(1, 2, 1).imshow(canvas, cmap="jet", vmin=0, vmax=1)

    canvas = np.zeros((args.max, args.max), dtype=np.float32)
    ts = np.linspace(0.0, 1.0, 2048)
    for t in ts:
        pt1 = curve4.evaluate(t).squeeze()
        pt2 = curve1.evaluate(t).squeeze()
        ctrl_1 = (1 - t) * ctrl_pt3[:, 1] + t * ctrl_pt2[:, 1]
        ctrl_2 = (1 - t) * ctrl_pt3[:, 2] + t * ctrl_pt2[:, 2]
        ctrl = list(zip(pt1, ctrl_2, ctrl_1, pt2))
        curve = bezier.Curve(ctrl, degree=3)
        coordinates = curve.evaluate_multi(np.linspace(0.0, 1.0, args.num)).astype(np.int32)
        canvas[coordinates[1], coordinates[0]] = t
    plt.subplot(1, 2, 2).imshow(canvas, cmap="jet", vmin=0, vmax=1)

    plt.show()


if __name__ == '__main__':
    main()
