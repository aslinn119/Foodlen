"""CLI: python -m challenge_external.demo_intervals path/to/rgb.jpg --model clip"""
import argparse,json
from challenge_external.runtime import predict_intervals

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--model',choices=['mean','resnet18','clip'],default='clip')
    args=parser.parse_args()
    print(json.dumps(predict_intervals(args.image,args.model),ensure_ascii=False,indent=2,allow_nan=False))
