import pandas as pd
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--input_dir", type=str, help="path to ground truth file", required=True)
    parser.add_argument("--train_dir", type=str, help="path to train dir", required=True)
    parser.add_argument("--val_dir", type=str, help="path to val dir", required=True)

    return parser.parse_args()

def split_data(input_dir, train_dir, val_dir):
    df = pd.read_csv(input_dir)

    # Chia thành hai DataFrame dựa trên giá trị của cột "video_id"
    df_train = df[df['video_id'] <= 90]
    df_val = df[df['video_id'] > 90]

    # Lưu hai DataFrame này thành hai file CSV riêng biệt
    df_train.to_csv(train_dir, index=False)
    df_val.to_csv(val_dir, index=False)

if __name__ == '__main__':
    args = get_args()
    split_data(args.input_dir, args.train_dir, args.val_dir)
