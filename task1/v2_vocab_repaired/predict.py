import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms as T
from model_v2 import CaptionModelV2
ROOT=Path(__file__).resolve().parent
def predict_caption(image_path, checkpoint=None, device=None):
 torch.set_num_threads(4)
 device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
 cp=torch.load(checkpoint or ROOT/"best_v2.pt",map_location=device,weights_only=False)
 m=CaptionModelV2(len(cp["vocab"]),pretrained=False).to(device)
 m.load_state_dict(cp["model"],strict=True);m.eval()
 tf=T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 with Image.open(image_path) as im:x=tf(im.convert("RGB")).unsqueeze(0).to(device)
 with torch.no_grad():words=m.greedy_decode(x,cp["vocab"],max_len=cp["config"]["max_generation_tokens"])[0]
 return {"caption":" ".join(words),"model_id":"v2_vocab_repaired_20261118","epoch":cp["epoch"]}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("image");p.add_argument("--checkpoint");p.add_argument("--device",choices=["cpu","cuda"]);a=p.parse_args();print(predict_caption(a.image,a.checkpoint,a.device))
