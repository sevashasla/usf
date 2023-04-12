import os
import argparse
import shutil

def my_clear_ngp(path):
    dirs = os.listdir(path)
    for d in dirs:
        curr_d = os.path.join(path, d)
        if not os.path.exists(curr_d) or not os.path.isdir(curr_d):
            continue
        shutil.rmtree(os.path.join(curr_d, 'run'), ignore_errors=True)
        shutil.rmtree(os.path.join(curr_d, 'checkpoints'), ignore_errors=True)
        shutil.rmtree(os.path.join(curr_d, 'validation'), ignore_errors=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to ngp dir")
    args = parser.parse_args()
    my_clear_ngp(args.path)

main()
