import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--train_or_test", type=str, help="Visualize data train or data test", required=True)
    parser.add_argument("--gt_dir", type=str, help="path to ground truth file", required=True)
    parser.add_argument("--root_dir", type=str, help="path to Frame folder", required=True)

    return parser.parse_args()

def draw_bounding_box(image, class_id, bb_left, bb_top, bb_width, bb_height, label_colors, label):
    color = label_colors.get(class_id, (0, 0, 0))
    cv2.rectangle(image, (bb_left, bb_top), (bb_left + bb_width, bb_top + bb_height), color, 6)
    label_name = label[class_id]
    cv2.putText(image, label_name, (bb_left, bb_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

    return image

def visualize(train_or_test,gt_dir, root_dir, label_colors, label):
    with open(gt_dir, "r") as file:
        data = file.readlines()
    for line in data:
        if train_or_test == 'train':
            video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id = map(int, line.strip().split(','))
        if train_or_test == 'test':
            video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id, conf = map(float, line.strip().split(','))
            video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id = int(video_id), int(frame), int(bb_left), int(bb_top), int(bb_width), int(bb_height), int(class_id)   

        # Extract image path
        video_folder_path = os.path.join(root_dir, '0' * (3-len(str(video_id))) + f"{video_id}")
        image_path = os.path.join(video_folder_path, f"frame{frame-1}.jpg")

        # Draw bounding box
        original_image = cv2.imread(image_path)
        image = draw_bounding_box(original_image, int(class_id), int(bb_left), int(bb_top), int(bb_width), int(bb_height), label_colors, label)

        # Save image
        output_image_path = os.path.join(root_dir,'0' * (3-len(str(video_id))) + f"{video_id}", f"frame{frame-1}.jpg")
        cv2.imwrite(output_image_path, image)


if __name__ == '__main__':
    args = get_args()

    # Define labels and colors for bounding box
    label = {
    1: 'motorbike',
    2: 'DHelmet',
    3: 'DNoHelmet',
    4: 'P1Helmet',
    5: 'P1NoHelmet',
    6: 'P2Helmet',
    7: 'P2NoHelmet',
    8: 'P0Helmet',
    9: 'P0NoHelmet'
    }
    label_colors = {
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (0, 0, 255),
        4: (255, 255, 0),
        5: (255, 0, 255),
        6: (0, 255, 255),
        7: (128, 128, 128),
        8: (255, 165, 0),
        9: (255, 255, 255)
    }
    # Define labels and colors for bounding box
    label_5 = {
    1: 'motorbike',
    2: 'D',
    3: 'P1',
    4: 'P2',
    5: 'P0',
    }
    label_colors_5 = {
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (0, 0, 255),
        4: (255, 255, 0),
        5: (255, 0, 255),
    }

    visualize(train_or_test=args.train_or_test,gt_dir=args.gt_dir, root_dir=args.root_dir, label_colors=label_colors, label=label)