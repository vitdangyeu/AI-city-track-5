import pandas as pd
import os
import cv2

from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--input_csv_dir", type=str, help="path to ground truth file csv", required=True)
    parser.add_argument("--image_folder_dir", type=str, help="path to folder of image", required=True)
    parser.add_argument("--output_dir", type=str, help="path to folder of image output", required=True)

    return parser.parse_args()

def cut_boundingbox(intput_csv_dir, image_folder_dir, output_dir):
    df = pd.read_csv(intput_csv_dir)
    for index, row in df.iterrows():
        image_path = os.path.join(image_folder_dir,'0' * (3-len(str(row['video_id']))) + f"{row['video_id']}", f"frame{row['frame']-1}.jpg")
        image = cv2.imread(image_path)
        # Cut Image
        x = row['bb_left']
        y = row['bb_top']
        width = row['bb_width']
        height = row['bb_height']
        cropped_image = image[y:y+height, x:x+width]
        cv2.imwrite(os.path.join(output_dir, f"{row['id']}.jpg"), cropped_image)
        if (index+1) % 1000 == 0:
            print(f"We have done {index + 1} image.")

if __name__ == '__main__':
    args = get_args()
    cut_boundingbox(args.input_csv_dir, args.image_folder_dir, args.output_dir)