from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, '.')
from src.input_validation.validator import CONSERVATIVE_THRESHOLD, load_reference_embeddings, build_feature_extractor, extract_embedding
from src.models.resnet import build_resnet18_classifier
from src.preprocessing.transforms import build_transforms
import json, torch
ckpt=Path('checkpoints/baseline_resnet18_no_aug/best_validation_model.pt')
cls_map=json.load(open('results/experiments/baseline_resnet18_no_aug/metrics/class_to_index.json'))
cfg=json.load(open('configs/baseline_resnet18_no_aug.json'))
prep=cfg['preprocessing']
trans=build_transforms(prep,'test')
device=torch.device('cpu')
model=build_resnet18_classifier(len(cls_map))
model.load_state_dict(torch.load(ckpt,map_location=device)['model_state_dict'])
model.eval()
extractor=build_feature_extractor(model)
extractor.eval().to(device)
ref=load_reference_embeddings()['embeddings']
import csv
from datetime import datetime
LOG=Path('results/experiments/input_validation/validation_log.csv')
LOG.parent.mkdir(parents=True, exist_ok=True)
if LOG.exists():
    LOG.unlink()
def log(img_path_str):
    pil=Image.open(img_path_str).convert('RGB')
    emb=extract_embedding(pil, extractor, trans, device)
    sims=emb @ ref.T
    max_sim=float(sims.max())
    mean_sim=float(sims.mean())
    is_suitable=max_sim>=CONSERVATIVE_THRESHOLD
    rec={'timestamp':datetime.now().isoformat(),'image_name':Path(img_path_str).name,'is_suitable':is_suitable,'max_similarity':f'{max_sim:.4f}','threshold':CONSERVATIVE_THRESHOLD,'mean_similarity':f'{mean_sim:.4f}'}
    wh=not LOG.exists()
    with LOG.open('a', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rec.keys()))
        if wh: w.writeheader()
        w.writerow(rec)
    print(f"{img_path_str} -> suitable={is_suitable} max={max_sim:.4f}")
log('data/raw/neu-det/NEU-DET/train/images/crazing/crazing_37.jpg')
log('data/raw/neu-det/NEU-DET/train/images/crazing/crazing_151.jpg')
log('C:/new.png')
log('data/raw/neu-det/NEU-DET/train/images/inclusion/inclusion_20.jpg')
print('--- log ---')
print(open(LOG).read())
