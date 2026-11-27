import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms as T
from model_v2 import CaptionModelV2
def predict_caption(image_path):
 cp=torch.load(Path(__file__).parent/"best_v2.pt",map_location="cpu",weights_only=False)
 m=CaptionModelV2(len(cp["vocab"]),pretrained=False);m.load_state_dict(cp["model"],strict=True);m.eval()
 tf=T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 with Image.open(image_path) as im:x=tf(im.convert("RGB")).unsqueeze(0)
 with torch.no_grad():words=m.greedy_decode(x,cp["vocab"],max_len=67)[0]
 return {"caption":" ".join(words),"model_id":"v2_vocab_repaired_20261118","epoch":cp["epoch"]}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("image");a=p.parse_args();print(predict_caption(a.image))
