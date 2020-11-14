"""Combine testing results of the two models to get final accuracy."""

import argparse
import numpy as np
np.set_printoptions(precision=5, linewidth=120)
def check_accuracy(RGB_class_accuracy, Depth_class_accuracy, Skeleton_class_accuracy):
    if Skeleton_class_accuracy != "":
        RGB = np.load(RGB_class_accuracy)
        Depth = np.load(Depth_class_accuracy)
        Skeleton = np.load(Skeleton_class_accuracy)
        RGB_weight = np.where((np.diag(RGB) > np.diag(Depth)) & (np.diag(RGB) > np.diag(Skeleton)), 2, 1)
        Skeleton_weight = np.where((np.diag(Skeleton) > np.diag(RGB)) & (np.diag(Skeleton) > np.diag(Depth)), 2, 1)
        Depth_weight = np.where((np.diag(Depth) >= np.diag(RGB)) & (np.diag(Depth) >= np.diag(Skeleton)), 2, 1)
        print(np.diag(RGB) / RGB.sum(axis=1))
        print(np.diag(Depth) / Depth.sum(axis=1))
        print(np.diag(Skeleton) / Skeleton.sum(axis=1))
        return RGB_weight, Depth_weight, Skeleton_weight
    else:
        RGB = np.load(RGB_class_accuracy)
        Depth = np.load(Depth_class_accuracy)
        RGB_weight = np.where(np.diag(RGB) > np.diag(Depth), 2, 1)
        Depth_weight = np.where(np.diag(Depth) >= np.diag(RGB), 2, 1)
        print(np.diag(RGB) / RGB.sum(axis=1))
        print(np.diag(Depth) / Depth.sum(axis=1))
        return RGB_weight, Depth_weight

def main():
    parser = argparse.ArgumentParser(description="combine predictions")
    parser.add_argument('--RGB', type=str, required=True, help='RGB score file.')
    parser.add_argument('--Depth', type=str, required=True, help='Depth score file.')
    parser.add_argument('--Skeleton', type=str, required=False, default = "", help='Skeleton score file.')

    parser.add_argument('--wRGB', type=float, default=1.0, help='RGB weight.')
    parser.add_argument('--wDepth', type=float, default=1.0, help='Depth weight.')
    parser.add_argument('--wSkeleton', type=float, default=1.0, help='Skeleton weight.')

    args = parser.parse_args()

    if args.Skeleton == "":
        RGB_weight, Depth_weight = check_accuracy(args.RGB.replace('scores.npz', 'cm.npy'), args.Depth.replace('scores.npz', 'cm.npy'), "")
        with np.load(args.RGB, allow_pickle = True, encoding="latin1") as RGB:
            with np.load(args.Depth, allow_pickle = True, encoding="latin1") as Depth:
                n = len(RGB['names'])

                RGB_score = np.array([score[0][0] for score in RGB['scores']])
                Depth_score = np.array([score[0][0] for score in Depth['scores']])
                RGB_label = np.array([score[1] for score in RGB['scores']])
                Depth_label = np.array([score[1] for score in Depth['scores']])
                
                assert np.alltrue(RGB_label == Depth_label)

                combined_score = RGB_score * args.wRGB + Depth_score * args.wDepth
                #combined_score = RGB_score * RGB_weight + Depth_score * Depth_weight

                accuracy = float(sum(np.argmax(combined_score, axis=1) == RGB_label)) / n
                print('Accuracy: %f (%d).\t(with RGB weight: %.3f, Depth weight: %.3f)' % (accuracy, n, args.wRGB, args.wDepth))
    else:
        RGB_weight, Depth_weight, Skeleton_weight = check_accuracy(args.RGB.replace('scores.npz', 'cm.npy'), args.Depth.replace('scores.npz', 'cm.npy'), args.Skeleton.replace('scores.npz', 'cm.npy'))
        with np.load(args.RGB) as RGB:
            with np.load(args.Depth) as Depth:
                with np.load(args.Skeleton) as Skeleton:
                    n = len(RGB['names'])

                    RGB_score = np.array([score[0][0] for score in RGB['scores']])
                    Depth_score = np.array([score[0][0] for score in Depth['scores']])
                    Skeleton_score = np.array([score[0][0] for score in Skeleton['scores']])
                    
                    RGB_label = np.array([score[1] for score in RGB['scores']])
                    Depth_label = np.array([score[1] for score in Depth['scores']])
                    Skeleton_label = np.array([score[1] for score in Skeleton['scores']])
                    
                    assert np.alltrue(RGB_label == Depth_label)

                    combined_score = RGB_score * args.wRGB + Depth_score * args.wDepth + Skeleton_score * args.wSkeleton
                    combined_score = RGB_score * RGB_weight + Depth_score * Depth_weight + Skeleton_score * Skeleton_weight

                    accuracy = float(sum(np.argmax(combined_score, axis=1) == RGB_label)) / n
                    print('Accuracy: %f (%d).\t(with RGB weight: %.3f, Depth weight: %.3f, Skeleton weight: %.3f)' % (accuracy, n, args.wRGB, args.wDepth, args.wSkeleton))

if __name__ == '__main__':
    main()
