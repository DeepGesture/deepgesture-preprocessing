import argparse
import yaml
from pprint import pprint
from easydict import EasyDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepGesturePreprocessing')
    parser.add_argument('--config', default='./configs/DeepGesturePreprocessing.yml')
    parser.add_argument('--bvh_path', type=str, default='../data/bvh')
    parser.add_argument('--wav_path', type=str, default='../data/wav')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)
    print("Hello")
