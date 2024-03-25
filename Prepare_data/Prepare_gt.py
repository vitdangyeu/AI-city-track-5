import pandas as pd
import json
import os

def prepare_gt_resnet(gt_dir, output_dir):
    data = []
    with open(gt_dir, "r") as file:
        for line in file:
            line = line.strip().split(',')
            line = [int(x) for x in line]
            if line[6] == 1:
                continue
            elif line[6] == 2 or line[6] == 4 or line[6] == 6 or line[6] == 8:
                line[6] = 1
            else:
                line[6] = 0
            data.append(line)
    with open(output_dir, "w") as file:
        for line in data:
            line = [str(x) for x in line]
            file.write(','.join(line) + '\n')
    
def prepare_gt_dert_resnet(gt_dir, output_dir):
    data = []
    with open(gt_dir, "r") as file:
        for line in file:
            line = line.strip().split(',')
            line = [int(x) for x in line]
            if line[6] == 2 or line[6] == 3:
                line[6] = 2
            elif line[6] == 4 or line[6] == 5:
                line[6] = 3
            elif line[6] == 6 or line[6] == 7:
                line[6] = 4
            elif line[6] == 8 or line[6] == 9:
                line[6] = 5
            data.append(line)
    with open(output_dir, "w") as file:
        for line in data:
            line = [str(x) for x in line]
            file.write(','.join(line) + '\n')

def prepare_gt_test(video_root_dir, output_dir):
    data = []
    videos = sorted(os.listdir(video_root_dir))
    for video in videos:
        images = os.listdir(os.path.join(video_root_dir, video))
        images.sort(key=lambda x: int(x.split('.')[0][5:]))
        for img in images:
            img = img.split('.')[0].split('frame')[1]
            data.append([str(int(video)), str(int(img))])

    with open(output_dir, 'w') as file:
        for row in data:
            file.write(','.join(row) + '\n')

def gt_to_resnet_csv(gt_dir, output_dir):
    # Read file
    data = []
    with open(gt_dir, "r") as file:
        for line in file:
            line = line.strip().split(',')
            data.append(line)
    # Create Dataframe
    df = pd.DataFrame(data, columns=["video_id", "frame", "bb_left", "bb_top", "bb_width", "bb_height", "class_id"])
    df.insert(0, 'id', range(len(df))) # Add 'id' column
    df.to_csv(output_dir, index=False) # Save csv file
        
def gt_to_dert_csv(gt_dir, output_dir):
    # Read file
    data = []
    with open(gt_dir, "r") as file:
        for line in file:
            line = line.strip().split(',')
            data.append(line)

    # Create Dataframe
    df = pd.DataFrame(data, columns=["video_id", "frame", "bb_left", "bb_top", "bb_width", "bb_height", "class_id"])
    df[["bb_left", "bb_top", "bb_width", "bb_height", "class_id"]] = df[["bb_left", "bb_top", "bb_width", "bb_height", "class_id"]].astype(int)

    # Merge_columns
    def merge_columns(row):
        return {'x': row['bb_left'], 'y': row['bb_top'], 'width': row['bb_width'], 'height': row['bb_height']}
    df['bounding_box'] = df.apply(merge_columns, axis=1)

    # Group same frame
    result = []
    for _, group in df.groupby(["video_id", "frame"]):
        bounding_boxes = list(group['bounding_box'])
        class_ids = list(group['class_id'])
        result.append({
            'video_id': int(group['video_id'].iloc[0]),
            'frame': int(group['frame'].iloc[0]),
            'bounding_box': bounding_boxes,
            'class_id': class_ids
        })

    # Create new dataframe
    df_result = pd.DataFrame(result)
    df_result = df_result[['video_id', 'frame', 'bounding_box', 'class_id']]
    df_result = df_result.sort_values(by=["video_id", "frame"]) # Sort resutl
    df_result.insert(0, 'id', range(len(df_result))) # Add 'id' column
    df_result.to_csv(output_dir, index=False) # Save csv file

def gt_to_prediction_csv(gt_dir, output_dir):
    # Read file
    data = []
    with open(gt_dir, "r") as file:
        for line in file:
            line = line.strip().split(',')
            data.append(line)
    # Create Dataframe
    df = pd.DataFrame(data, columns=["video_id", "frame"])
    df.insert(0, 'id', range(len(df))) # Add 'id' column
    df.to_csv(output_dir, index=False) # Save csv file

# # Create file csv for dert model
# gt_dir = "D:\Competition\CVPR_2023\Track_5\Data\gt.txt"
# output_dir = "D:\Competition\CVPR_2023\Track_5\Compare\Data\Ground_truth\gt_dert.csv"
# gt_to_dert_csv(gt_dir, output_dir)

# Create file csv for dert and resnet model
# gt_dir_track5 = "D:\Competition\CVPR_2023\Track_5\Compare\Data\gt.txt"
# output_dir_gt = "D:\Competition\CVPR_2023\Track_5\Compare\Data\gt_dert_resnet.txt"
# gt_dir = "D:\Competition\CVPR_2023\Track_5\Compare\Data\gt_dert_resnet.txt"
# output_dir_csv = "D:\Competition\CVPR_2023\Track_5\Compare\Data\Ground_truth\gt_dert_resnet.csv"
# prepare_gt_dert_resnet(gt_dir_track5, output_dir_gt)
# gt_to_dert_csv(gt_dir, output_dir_csv)

# Create file csv for dert and resnet model
# gt_dir_track5 = "D:\Competition\CVPR_2023\Track_5\Compare\Data\gt.txt"
# output_dir_gt = "D:\Competition\CVPR_2023\Track_5\Compare\Data\gt_resnet.txt"
# gt_dir = "D:\Competition\CVPR_2023\Track_5\Compare\Data\gt_resnet.txt"
# output_dir_csv = "D:\Competition\CVPR_2023\Track_5\Compare\Data\Ground_truth\gt_resnet.csv"
# prepare_gt_resnet(gt_dir_track5, output_dir_gt)
# gt_to_resnet_csv(gt_dir, output_dir_csv)

# Create file csv for dert and resnet model
video_root_dir = 'D:\Competition\CVPR_2023\Track_5\Data\Frame'
output_dir_gt = 'D:\Competition\CVPR_2023\Track_5\Data\gt_yolo.txt'
output_dir_csv = 'D:\Competition\CVPR_2023\Track_5\Data\Ground_truth\gt_yolo.csv'

prepare_gt_test(video_root_dir, output_dir_gt)
gt_to_prediction_csv(output_dir_gt, output_dir_csv)

