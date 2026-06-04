"""
SNN-PAR ANN-teacher inference on ITCPR gallery.

SNN-PAR is a dual-stream model (ANN teacher + SNN student).  The released
checkpoint ckpt_peta.pth contains only the ANN teacher weights (no snn_model
keys).  We run inference using the ANN branch only, which is the full
classifer used at test-time in the paper.

Usage:
    uv run python infer_itcpr_snn.py \
        --ckpt /home/mika/mnt/yomikawa-reid/repos/yomikawa-reid/checkpoints/snn_par/ckpt_peta.pth \
        --output /tmp/preds_snn.json
"""
import sys, os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# ── import ViT from SNN-PAR (needed for the exact Block/LayerNorm definitions)
SNN_PAR_DIR = os.path.join(os.path.dirname(__file__), '..', 'SNN-PAR')
sys.path.insert(0, os.path.abspath(SNN_PAR_DIR))
from models.vit import vit_base   # SNN-PAR's ViT-B/16

ITCPR_ROOT = '/home/mika/mnt/yomikawa-reid/repos/yomikawa-reid/datasets/ITCPR'
VIT_PRETRAIN = '/home/mika/mnt/yomikawa-reid/repos/yomikawa-reid/checkpoints/prompt_par/jx_vit_base_p16_224-80ecf9dd.pth'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── PETA attribute vocabulary (SNN-PAR peta.py) ───────────────────────────────
PETA_ATTRS = [
    'accessory hat', 'accessory muffler', 'accessory nothing',
    'accessory sunglasses', 'accessory long hair',
    'upper body casual', 'upper body formal', 'upper body jacket',
    'upper body logo', 'upper body plaid', 'upper body short sleeve',
    'upper body thin stripes', 'upper body t-shirt', 'upper body other',
    'upper body v-neck',
    'lower body Casual', 'lower body Formal', 'lower body Jeans',
    'lower body Shorts', 'lower body Short Skirt', 'lower body Trousers',
    'foot wear Leather shoes', 'foot wear Sandals shoes',
    'foot wear shoes', 'foot wear sneaker shoes',
    'carrying Backpack', 'carrying Other', 'carrying messenger bag',
    'carrying nothing', 'carrying plastic bags',
    'personal less 30', 'personal less 45', 'personal less 60',
    'personal larger 60', 'personal male',
]  # 35


# ── ANN-only model (teacher branch) ──────────────────────────────────────────
class SNNPARTeacher(nn.Module):
    """ANN teacher branch of SNN-PAR (ViT + word_embed + MM-former + classifier)."""

    def __init__(self, attr_num: int, word_vec: torch.Tensor,
                 pretrain_path: str = VIT_PRETRAIN, dim: int = 768):
        super().__init__()
        self.attr_num = attr_num

        # frozen text embeddings (sentence-transformer, computed once)
        self.register_buffer('word_vec', word_vec)   # [attr_num, 768]

        self.word_embed = nn.Linear(768, dim)
        self.vis_embed  = nn.Parameter(torch.zeros(1, 1, dim))
        self.tex_embed  = nn.Parameter(torch.zeros(1, 1, dim))

        # ViT-B/16 with 256×128 positional embedding (128 patches + CLS)
        self.vit = vit_base()
        self.vit.load_param(pretrain_path)

        # MM-former: last ViT block + its norm
        self.blocks = self.vit.blocks[-1:]
        self.norm   = self.vit.norm

        self.weight_layer = nn.ModuleList(
            [nn.Linear(dim, 1) for _ in range(attr_num)])
        self.bn = nn.BatchNorm1d(attr_num)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        B = imgs.shape[0]
        features  = self.vit(imgs)                              # [B, 129, 768]
        word_proj = self.word_embed(self.word_vec)              # [attr_num, 768]

        tex = word_proj.unsqueeze(0).expand(B, -1, -1) + self.tex_embed  # [B, 35, 768]
        vis = features + self.vis_embed                                    # [B, 129, 768]

        x = torch.cat([tex, vis], dim=1)                        # [B, 164, 768]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        logits = torch.cat([self.weight_layer[i](x[:, i, :])
                            for i in range(self.attr_num)], dim=1)  # [B, 35]
        return self.bn(logits)


# ── dataset ───────────────────────────────────────────────────────────────────
class ITCPRGallery(Dataset):
    def __init__(self, root: str, transform):
        with open(os.path.join(root, 'gallery.json')) as f:
            self.entries = json.load(f)
        self.root = root
        self.transform = transform

    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img = Image.open(os.path.join(self.root, e['file_path'])).convert('RGB')
        return self.transform(img), e['file_path']


# ── helpers ───────────────────────────────────────────────────────────────────
def compute_word_vecs(attrs):
    from sentence_transformers import SentenceTransformer
    print('  computing sentence embeddings …')
    st = SentenceTransformer('all-mpnet-base-v2')
    vecs = st.encode(attrs, convert_to_tensor=False)     # np [n, 768]
    return torch.tensor(vecs, dtype=torch.float32)


def load_model(ckpt_path: str, word_vec: torch.Tensor,
               attr_num: int) -> SNNPARTeacher:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd   = ckpt['state_dicts']

    model = SNNPARTeacher(attr_num, word_vec)

    # The checkpoint stores the ViT as self.vit.*, and blocks/norm at top level.
    # Our model matches this layout exactly.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'  missing keys  ({len(missing)}): {missing[:4]}')
    if unexpected:
        print(f'  unexpected keys ({len(unexpected)}): {unexpected[:4]}')
    return model


# ── main ──────────────────────────────────────────────────────────────────────
def run(args):
    print(f'Device: {DEVICE}')

    # image transform: 256×128, normalise [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = ITCPRGallery(ITCPR_ROOT, transform)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)
    print(f'Gallery: {len(dataset)} images  |  attr_num: {len(PETA_ATTRS)}')

    word_vec = compute_word_vecs(PETA_ATTRS)

    print('Loading checkpoint:', args.ckpt)
    model = load_model(args.ckpt, word_vec, len(PETA_ATTRS))
    model = model.to(DEVICE).eval()

    all_probs, all_paths = [], []
    t0 = time.time()
    with torch.no_grad():
        for step, (imgs, paths) in enumerate(loader):
            logits = model(imgs.to(DEVICE))
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_paths.extend(paths)
            if step % 50 == 0:
                print(f'  step {step}/{len(loader)}  ({time.time()-t0:.1f}s)')

    all_probs = np.concatenate(all_probs, axis=0)   # [N, 35]
    print(f'Done in {time.time()-t0:.1f}s')

    # build output
    results = {}
    for i, path in enumerate(all_paths):
        results[path] = {
            'probs': {a: float(all_probs[i, j]) for j, a in enumerate(PETA_ATTRS)},
            'attrs': [a for j, a in enumerate(PETA_ATTRS)
                      if all_probs[i, j] > args.threshold],
        }

    out = {'model': 'SNN-PAR (ANN teacher)', 'checkpoint': 'ckpt_peta.pth',
           'attr_words': PETA_ATTRS, 'threshold': args.threshold,
           'predictions': results}
    with open(args.output, 'w') as f:
        json.dump(out, f)
    print(f'Saved → {args.output}')

    # ── collapse analysis ──────────────────────────────────────────────────────
    print('\n=== Collapse Analysis ===')
    mean_p = all_probs.mean(0)
    std_p  = all_probs.std(0)
    pos_r  = (all_probs > args.threshold).mean(0)

    print(f'\n{"Attribute":<36}  {"Mean":>6}  {"Std":>6}  {"PosRate":>8}')
    print('-' * 62)
    for j, a in enumerate(PETA_ATTRS):
        flag = '  *** COLLAPSE' if std_p[j] < 0.02 else ''
        print(f'{a:<36}  {mean_p[j]:6.3f}  {std_p[j]:6.3f}  {pos_r[j]:8.3f}{flag}')

    print(f'\nOverall mean prob:            {mean_p.mean():.4f}')
    print(f'Overall std  prob:            {std_p.mean():.4f}')
    print(f'Collapsed attrs (std<0.02):   {(std_p<0.02).sum()}')
    print(f'Dead attrs (pos_rate<0.01):   {(pos_r<0.01).sum()}')
    print(f'Saturated attrs (pos_rate>0.99): {(pos_r>0.99).sum()}')
    print(f'Mean per-image std: {all_probs.std(1).mean():.4f}')

    from collections import Counter
    sets = Counter(tuple(sorted(v['attrs'])) for v in results.values())
    print(f'\nUnique prediction sets: {len(sets)}')
    print('Top-5:')
    for s, c in sets.most_common(5):
        print(f'  {100*c/len(results):.1f}%  {s}')

    # pairwise cosine similarity
    idx1 = np.random.choice(len(all_probs), 1000, replace=False)
    idx2 = np.random.choice(len(all_probs), 1000, replace=False)
    n1, n2 = all_probs[idx1], all_probs[idx2]
    cos = (n1 * n2).sum(1) / (np.linalg.norm(n1, 1) * np.linalg.norm(n2, 1) + 1e-8)
    print(f'Pairwise cosine similarity: {cos.mean():.4f} ± {cos.std():.4f}')

    # save stats
    stats = {
        'model': 'SNN-PAR',
        'n_images': len(all_paths),
        'mean_prob_per_attr': mean_p.tolist(),
        'std_prob_per_attr':  std_p.tolist(),
        'pos_rate_per_attr':  pos_r.tolist(),
        'attr_words': PETA_ATTRS,
        'collapsed_attrs':   [PETA_ATTRS[j] for j in range(len(PETA_ATTRS)) if std_p[j] < 0.02],
        'dead_attrs':        [PETA_ATTRS[j] for j in range(len(PETA_ATTRS)) if pos_r[j] < 0.01],
        'saturated_attrs':   [PETA_ATTRS[j] for j in range(len(PETA_ATTRS)) if pos_r[j] > 0.99],
        'unique_prediction_sets': len(sets),
        'pairwise_cosine_mean': float(cos.mean()),
    }
    stats_path = args.output.replace('.json', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Stats saved → {stats_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',       default='/home/mika/mnt/yomikawa-reid/repos/yomikawa-reid/checkpoints/snn_par/ckpt_peta.pth')
    p.add_argument('--output',     default='/tmp/preds_snn.json')
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--threshold',  type=float, default=0.45)
    run(p.parse_args())
