"""
Run PromptPAR inference on ITCPR gallery images.
Usage:
    python infer_itcpr.py PA100k --use_mm_former --use_div --dir /path/to/PA100k_Checkpoint.pth --output preds_pa100k.json
    python infer_itcpr.py PETA   --use_mm_former --use_div --dir /path/to/PETA_Checkpoint.pth   --output preds_peta.json
"""
import sys
import os
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# ── attr vocabulary ────────────────────────────────────────────────────────────
PA100K_ATTRS = [
    'female',
    'age over 60', 'age 18 to 60', 'age less 18',
    'front', 'side', 'back',
    'hat', 'glasses',
    'hand bag', 'shoulder bag', 'backpack', 'hold objects in front',
    'short sleeve', 'long sleeve', 'upper stride', 'upper logo', 'upper plaid', 'upper splice',
    'lower stripe', 'lower pattern', 'long coat', 'trousers', 'shorts', 'skirt and dress', 'boots',
]  # 26

PETA_ATTRS = [
    'head hat', 'head muffler', 'head nothing', 'head sunglasses', 'head long hair',
    'upper casual', 'upper formal', 'upper jacket', 'upper logo', 'upper plaid',
    'upper short sleeve', 'upper thin stripes', 'upper t-shirt', 'upper other', 'upper v-neck',
    'lower Casual', 'lower Formal', 'lower Jeans', 'lower Shorts', 'lower Short Skirt', 'lower Trousers',
    'shoes Leather', 'shoes Sandals', 'shoes other', 'shoes sneaker',
    'attach Backpack', 'attach Other', 'attach messenger bag', 'attach nothing', 'attach plastic bags',
    'age less 30', 'age 30 45', 'age 45 60', 'age over 60',
    'male',
]  # 35

ATTR_VOCAB = {'PA100k': PA100K_ATTRS, 'PETA': PETA_ATTRS}


# ── extra CLI arg before importing project modules ──────────────────────────────
def _patch_argv(dataset: str) -> argparse.Namespace:
    """
    The project modules (clip/model.py, models/base_block.py) parse sys.argv at import
    time, so we must set it before the first import.  We keep --use_mm_former, --use_div
    as flags and rely on the defaults (vis_prompt=50, text_prompt=3, vis_depth=24,
    div_num=4, mm_layers=1) which match the released checkpoints.
    """
    extra = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    sys.argv = [sys.argv[0], dataset, '--use_mm_former', '--use_div', '--use_textprompt', '--use_GL'] + extra
    from config import argument_parser
    return argument_parser().parse_args()


def _parse_own_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('dataset', choices=['PA100k', 'PETA'])
    p.add_argument('--dir', required=True)
    p.add_argument('--output', default=None)
    p.add_argument('--itcpr_root', default='/home/mika/mnt/yomikawa-reid/repos/yomikawa-reid/datasets/ITCPR')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--threshold', type=float, default=0.45)
    known, _ = p.parse_known_args()
    return known


own = _parse_own_args()
args = _patch_argv(own.dataset)   # must happen before project imports

# ── now safe to import project modules ─────────────────────────────────────────
from clip.model import build_model
from models.base_block import TransformerClassifier

VIT_PRETRAIN = '/home/mika/mnt/yomikawa-reid/repos/yomikawa-reid/checkpoints/prompt_par/jx_vit_base_p16_224-80ecf9dd.pth'
_use_cuda = os.environ.get('INFER_DEVICE', 'cuda') == 'cuda' and torch.cuda.is_available()
DEVICE = 'cuda' if _use_cuda else 'cpu'
print(f'Using device: {DEVICE}')


# ── dataset ────────────────────────────────────────────────────────────────────
class ITCPRGalleryDataset(Dataset):
    def __init__(self, itcpr_root: str, transform):
        with open(os.path.join(itcpr_root, 'gallery.json')) as f:
            self.entries = json.load(f)
        self.itcpr_root = itcpr_root
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = os.path.join(self.itcpr_root, entry['file_path'])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image), entry['file_path']


# ── inference ──────────────────────────────────────────────────────────────────
def run():
    attr_words = ATTR_VOCAB[own.dataset]
    attr_num = len(attr_words)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ITCPRGalleryDataset(own.itcpr_root, transform)
    loader = DataLoader(dataset, batch_size=own.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    print(f'Gallery size: {len(dataset)} images  |  attr_num: {attr_num}')

    # load model
    print('Loading checkpoint:', own.dir)
    ckpt = torch.load(own.dir, map_location='cpu', weights_only=False)
    clip_model = build_model(ckpt['ViT_model']).to(DEVICE)

    model = TransformerClassifier(clip_model, attr_num, attr_words,
                                  pretrain_path=VIT_PRETRAIN)

    # fix vis_embed → visual_embed key rename
    state = ckpt['model_state_dict']
    fixed = {}
    for k, v in state.items():
        new_k = k.replace('vis_embed.', 'visual_embed.')
        fixed[new_k] = v

    missing, unexpected = model.load_state_dict(fixed, strict=False)
    if missing:
        print(f'  missing keys ({len(missing)}):', missing[:5])
    if unexpected:
        print(f'  unexpected keys ({len(unexpected)}):', unexpected[:5])

    model = model.to(DEVICE)
    model.eval()
    clip_model.eval()

    # run inference
    all_probs = []
    all_paths = []
    t0 = time.time()

    with torch.no_grad():
        for step, (imgs, paths) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            logits, _ = model(imgs, clip_model=clip_model)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_paths.extend(paths)
            if step % 50 == 0:
                print(f'  step {step}/{len(loader)}  ({time.time()-t0:.1f}s)')

    all_probs = np.concatenate(all_probs, axis=0)  # [N, attr_num]
    print(f'Done in {time.time()-t0:.1f}s')

    # build results
    results = {}
    for i, path in enumerate(all_paths):
        probs_dict = {attr: float(all_probs[i, j]) for j, attr in enumerate(attr_words)}
        attrs_pred = [attr for j, attr in enumerate(attr_words)
                      if all_probs[i, j] > own.threshold]
        results[path] = {
            'probs': probs_dict,
            'attrs': attrs_pred,
        }

    # save
    out_path = own.output or f'preds_itcpr_{own.dataset.lower()}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'dataset': own.dataset,
            'attr_words': attr_words,
            'threshold': own.threshold,
            'predictions': results,
        }, f)
    print(f'Saved {len(results)} predictions → {out_path}')

    # ── collapse analysis ──────────────────────────────────────────────────────
    print('\n=== Collapse Analysis ===')
    print(f'Images: {len(all_paths)},  Attributes: {attr_num}')

    mean_prob = all_probs.mean(axis=0)    # [attr_num]
    std_prob  = all_probs.std(axis=0)     # [attr_num]
    pos_rate  = (all_probs > own.threshold).mean(axis=0)  # [attr_num]

    print(f'\n{"Attribute":<32}  {"Mean":>6}  {"Std":>6}  {"PosRate":>8}')
    print('-' * 58)
    for j, attr in enumerate(attr_words):
        flag = '*** COLLAPSE' if std_prob[j] < 0.02 else ''
        print(f'{attr:<32}  {mean_prob[j]:6.3f}  {std_prob[j]:6.3f}  {pos_rate[j]:8.3f}  {flag}')

    print(f'\nOverall mean prob:  {mean_prob.mean():.4f}')
    print(f'Overall std  prob:  {std_prob.mean():.4f}')
    print(f'Collapsed attrs (std<0.02): {(std_prob < 0.02).sum()}')
    print(f'Dead attrs (pos_rate<0.01): {(pos_rate < 0.01).sum()}')
    print(f'Saturated attrs (pos_rate>0.99): {(pos_rate > 0.99).sum()}')

    # inter-image variance — collapsed if all images predict nearly identically
    img_std = all_probs.std(axis=1).mean()
    print(f'Mean per-image std across attrs: {img_std:.4f}')

    # save stats for comparison
    stats = {
        'dataset': own.dataset,
        'n_images': len(all_paths),
        'mean_prob_per_attr': mean_prob.tolist(),
        'std_prob_per_attr': std_prob.tolist(),
        'pos_rate_per_attr': pos_rate.tolist(),
        'attr_words': attr_words,
        'collapsed_attrs': [attr_words[j] for j in range(attr_num) if std_prob[j] < 0.02],
        'dead_attrs': [attr_words[j] for j in range(attr_num) if pos_rate[j] < 0.01],
        'saturated_attrs': [attr_words[j] for j in range(attr_num) if pos_rate[j] > 0.99],
    }
    stats_path = out_path.replace('.json', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Stats saved → {stats_path}')


if __name__ == '__main__':
    run()
