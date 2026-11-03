import json, numpy as np, torch, pandas as pd
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, '.')
from src.models.resnet import build_resnet18_classifier
from src.preprocessing.transforms import build_transforms
import torch.nn.functional as F
ckpt=Path('checkpoints/baseline_resnet18_no_aug/best_validation_model.pt')
cls_map=json.load(open('results/experiments/baseline_resnet18_no_aug/metrics/class_to_index.json'))
cfg=json.load(open('configs/baseline_resnet18_no_aug.json'))
prep=cfg['preprocessing']
trans=build_transforms(prep,'test')
device=torch.device('cpu')
model=build_resnet18_classifier(len(cls_map))
model.load_state_dict(torch.load(ckpt,map_location=device)['model_state_dict'])
model.eval()
extractor=torch.nn.Sequential(*list(model.children())[:-1])
extractor.eval().to(device)
ref=np.load('results/experiments/input_validation/reference_embeddings.npz')['embeddings']
def emb_for_pil(pil):
    inp=trans(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        f=extractor(inp).squeeze()
        if f.dim()>1:
            f=f.view(f.size(0),-1).squeeze(0)
        f=F.normalize(f,p=2,dim=0).cpu().numpy()
    sims=f @ ref.T
    return float(sims.max())

import random
val_df=pd.read_csv('data/splits/seed_42_70_15_15/validation.csv')
random.seed(42)
sample=val_df.sample(10, random_state=42)
for _, row in sample.iterrows():
    img_path=Path(row['image_path'])
    if not img_path.exists():
        alt=Path(str(img_path).replace('data/raw/NEU-DET','data/raw/neu-det'))
        if alt.exists():
            img_path=alt
        else:
            print('missing', row['image_path'])
            continue
    pil=Image.open(img_path).convert('RGB')
    print(row['image_path'], f"{emb_for_pil(pil):.4f}", 'class',row['class'])

for p in ['C:/new.png']:
    if Path(p).exists():
        pil=Image.open(p).convert('RGB')
        print('unrelated C:/new.png', f"{emb_for_pil(pil):.4f}")

from PIL import Image as PILI
import tempfile
tmp=Path(tempfile.gettempdir()) / 'tmptest224.png'
for col in [(255,0,0),(0,255,0),(0,0,0),(255,255,255),(128,128,128)]:
    im=PILI.new('RGB',(224,224),color=col)
    im.save(tmp)
    pil=Image.open(tmp).convert('RGB')
    print(f'solid {col} {emb_for_pil(pil):.4f}')

for p in ['data/raw/neu-det/NEU-DET/train/images/crazing/crazing_37.jpg','data/raw/neu-det/NEU-DET/train/images/crazing/crazing_151.jpg','data/raw/neu-det/NEU-DET/train/images/inclusion/inclusion_20.jpg']:
    pil=Image.open(p).convert('RGB')
    print(p, f"{emb_for_pil(pil):.4f}")
