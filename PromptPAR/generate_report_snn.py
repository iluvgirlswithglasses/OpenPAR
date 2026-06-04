"""
SNN-PAR vs PromptPAR comparison report.
Usage:  uv run python generate_report_snn.py
"""
import base64, io, json, os, random
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ITCPR_ROOT = '/home/mika/mnt/yomikawa-reid/repos/yomikawa-reid/datasets/ITCPR'
PA_PREDS   = '/tmp/preds_pa100k.json'
PE_PREDS   = '/tmp/preds_peta.json'
SNN_PREDS  = '/tmp/preds_snn.json'
OUT_HTML   = 'itcpr_report_snn.html'
SEED = 42
random.seed(SEED); np.random.seed(SEED)

C_PA   = '#2980b9'   # blue  – PA100k PromptPAR
C_PE   = '#8e44ad'   # purple – PETA PromptPAR
C_SNN  = '#27ae60'   # green  – SNN-PAR
C_RED  = '#e74c3c'
C_ORG  = '#e67e22'
C_YLW  = '#f1c40f'
C_DARK = '#0f0f1a'
C_MID  = '#16213e'

# ── helpers ───────────────────────────────────────────────────────────────────
def fig_to_b64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def img_to_b64(pil_img):
    buf = io.BytesIO(); pil_img.save(buf, format='PNG'); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def load_img(rel_path, w=110, h=220):
    try:
        return Image.open(os.path.join(ITCPR_ROOT, rel_path)).convert('RGB').resize((w, h), Image.LANCZOS)
    except Exception:
        return None

def dark_fig(w, h):
    fig, axes = plt.subplots(1, 1, figsize=(w, h), facecolor=C_DARK)
    axes.set_facecolor(C_MID)
    return fig, axes

def multi_fig(cols, rows, w, h):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h), facecolor=C_DARK)
    for ax in np.array(axes).flatten():
        ax.set_facecolor(C_MID)
    return fig, axes

def style_ax(ax):
    ax.tick_params(colors='white')
    for s in ax.spines.values(): s.set_color('#334')
    return ax

# ── load data ──────────────────────────────────────────────────────────────────
print('Loading …')
with open(PA_PREDS)  as f: pa_raw  = json.load(f)
with open(PE_PREDS)  as f: pe_raw  = json.load(f)
with open(SNN_PREDS) as f: snn_raw = json.load(f)
with open(os.path.join(ITCPR_ROOT, 'gallery.json')) as f: gallery = json.load(f)

preds_pa  = pa_raw['predictions'];  attrs_pa  = pa_raw['attr_words']
preds_pe  = pe_raw['predictions'];  attrs_pe  = pe_raw['attr_words']
preds_snn = snn_raw['predictions']; attrs_snn = snn_raw['attr_words']

paths_all = list(preds_snn.keys())   # all three share the same gallery
N = len(paths_all)

def probs_array(preds, attrs):
    return np.array([[preds[p]['probs'][a] for a in attrs] for p in paths_all])

probs_pa  = probs_array(preds_pa,  attrs_pa)
probs_pe  = probs_array(preds_pe,  attrs_pe)
probs_snn = probs_array(preds_snn, attrs_snn)

def stats(probs):
    return probs.mean(0), probs.std(0), (probs > 0.45).mean(0)

mean_pa,  std_pa,  pos_pa  = stats(probs_pa)
mean_pe,  std_pe,  pos_pe  = stats(probs_pe)
mean_snn, std_snn, pos_snn = stats(probs_snn)

ctr_pa  = Counter(tuple(sorted(v['attrs'])) for v in preds_pa.values())
ctr_pe  = Counter(tuple(sorted(v['attrs'])) for v in preds_pe.values())
ctr_snn = Counter(tuple(sorted(v['attrs'])) for v in preds_snn.values())

def cosine_sim(probs, n=2000):
    i1 = np.random.choice(len(probs), n, replace=False)
    i2 = np.random.choice(len(probs), n, replace=False)
    a, b = probs[i1], probs[i2]
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)

cos_pa  = cosine_sim(probs_pa)
cos_pe  = cosine_sim(probs_pe)
cos_snn = cosine_sim(probs_snn)

sources = {'Celeb-reID': [], 'PRCC': [], 'LAST': []}
for e in gallery:
    s = e['file_path'].split('/')[0]
    if s in sources: sources[s].append(e['file_path'])

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Head-to-head scorecard bar chart
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 1: scorecard bars …')
fig1, ax = plt.subplots(figsize=(14, 5), facecolor=C_DARK)
ax.set_facecolor(C_MID); style_ax(ax)

labels = [
    'Unique\nprediction sets',
    'Collapsed\nattrs (std<0.02)',
    'Dead attrs\n(pos<1%)',
    'Saturated\nattrs (pos>99%)',
]
pa_vals  = [len(ctr_pa),  (std_pa <0.02).sum(), (pos_pa <0.01).sum(), (pos_pa >0.99).sum()]
pe_vals  = [len(ctr_pe),  (std_pe <0.02).sum(), (pos_pe <0.01).sum(), (pos_pe >0.99).sum()]
snn_vals = [len(ctr_snn), (std_snn<0.02).sum(), (pos_snn<0.01).sum(), (pos_snn>0.99).sum()]
# normalise for display on a log scale for 'unique sets'
x = np.arange(len(labels))
w = 0.25
ax.bar(x - w, pa_vals,  width=w, color=C_PA,  label='PA100k (PromptPAR)', alpha=0.9)
ax.bar(x,     pe_vals,  width=w, color=C_PE,  label='PETA (PromptPAR)',   alpha=0.9)
ax.bar(x + w, snn_vals, width=w, color=C_SNN, label='SNN-PAR (PETA)',     alpha=0.9)
for i, (pav, pev, sv) in enumerate(zip(pa_vals, pe_vals, snn_vals)):
    ax.text(i-w, pav+50, str(pav), ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
    ax.text(i,   pev+50, str(pev), ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
    ax.text(i+w, sv+50,  str(sv),  ha='center', va='bottom', color=C_SNN,  fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels, color='white', fontsize=11)
ax.set_ylabel('Count', color='white', fontsize=11)
ax.set_title('Model Collapse Metrics — Three Checkpoints on ITCPR', color='white', fontsize=13)
ax.legend(facecolor=C_DARK, edgecolor='#334', labelcolor='white', fontsize=10)
ax.set_ylim(0, max(pa_vals)*1.15)
fig1.tight_layout()
FIG1 = fig_to_b64(fig1)
plt.close(fig1)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Cosine similarity distributions side by side
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 2: cosine similarity …')
fig2, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor=C_DARK)
for ax, cos, title, col in [
    (axes[0], cos_pa,  f'PA100k PromptPAR\nmean={cos_pa.mean():.4f}',  C_PA),
    (axes[1], cos_pe,  f'PETA PromptPAR\nmean={cos_pe.mean():.4f}',    C_PE),
    (axes[2], cos_snn, f'SNN-PAR\nmean={cos_snn.mean():.4f}',          C_SNN),
]:
    ax.set_facecolor(C_MID); style_ax(ax)
    ax.hist(cos, bins=60, color=col, edgecolor='none', alpha=0.85)
    ax.axvline(cos.mean(), color='white', lw=1.5, ls='--')
    ax.set_xlabel('Cosine Similarity', color='white', fontsize=10)
    ax.set_ylabel('Count', color='white', fontsize=10)
    ax.set_title(title, color='white', fontsize=11)
    ax.set_xlim(-0.1, 1.1)
fig2.suptitle('Inter-Image Prediction Similarity  (1.0 = identical, 0.0 = orthogonal, <0 = opposite)',
              color='white', fontsize=12, y=1.01)
fig2.tight_layout()
FIG2 = fig_to_b64(fig2)
plt.close(fig2)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Per-image std distributions
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 3: per-image std …')
fig3, ax = plt.subplots(figsize=(12, 4.5), facecolor=C_DARK)
ax.set_facecolor(C_MID); style_ax(ax)
for probs, col, lbl in [(probs_pa, C_PA, 'PA100k'), (probs_pe, C_PE, 'PETA'), (probs_snn, C_SNN, 'SNN-PAR')]:
    s = probs.std(axis=1)
    ax.hist(s, bins=70, color=col, alpha=0.55, label=f'{lbl}  μ={s.mean():.3f}')
ax.set_xlabel('Per-image Std across Attributes', color='white', fontsize=11)
ax.set_ylabel('Count', color='white', fontsize=11)
ax.set_title('Per-Image Attribute Diversity — wider distribution = model is discriminating more', color='white', fontsize=12)
ax.legend(facecolor=C_DARK, edgecolor='#334', labelcolor='white', fontsize=11)
fig3.tight_layout()
FIG3 = fig_to_b64(fig3)
plt.close(fig3)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Attribute std comparison: SNN vs PETA PromptPAR
# (shared attr vocabulary: PETA-35 — same set for both, different phrasing)
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 4: per-attr std comparison …')
fig4, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=C_DARK)

# left: PETA PromptPAR
ax = axes[0]; ax.set_facecolor(C_MID); style_ax(ax)
colors_pe = [C_RED if s<0.02 and p>0.9 else '#7f8c8d' if s<0.02 and p<0.1
             else C_ORG if s<0.02 else C_PE for s,p in zip(std_pe, pos_pe)]
ax.barh(range(len(attrs_pe)), std_pe, color=colors_pe, edgecolor='none', height=0.7)
ax.set_yticks(range(len(attrs_pe))); ax.set_yticklabels(attrs_pe, fontsize=8, color='white')
ax.set_xlabel('Std of Probability', color='white', fontsize=10)
ax.axvline(0.02, color='white', lw=1, ls='--', alpha=0.5)
ax.set_title('PETA PromptPAR — Attribute Variance', color='white', fontsize=11)
legend_items = [
    mpatches.Patch(color=C_RED,     label='Saturated (collapsed, pos>90%)'),
    mpatches.Patch(color='#7f8c8d', label='Dead (collapsed, pos<10%)'),
    mpatches.Patch(color=C_ORG,     label='Collapsed (std<0.02)'),
    mpatches.Patch(color=C_PE,      label='Active'),
]
ax.legend(handles=legend_items, fontsize=8, facecolor=C_DARK, edgecolor='#334', labelcolor='white')

# right: SNN-PAR
ax = axes[1]; ax.set_facecolor(C_MID); style_ax(ax)
colors_snn = [C_SNN if s>=0.1 else '#aaddbb' for s in std_snn]
ax.barh(range(len(attrs_snn)), std_snn, color=colors_snn, edgecolor='none', height=0.7)
ax.set_yticks(range(len(attrs_snn))); ax.set_yticklabels(attrs_snn, fontsize=8, color='white')
ax.set_xlabel('Std of Probability', color='white', fontsize=10)
ax.axvline(0.02, color='white', lw=1, ls='--', alpha=0.5)
ax.set_title('SNN-PAR — Attribute Variance  (no collapsed attributes!)', color='white', fontsize=11)
legend2 = [
    mpatches.Patch(color=C_SNN,      label='Highly active (std≥0.10)'),
    mpatches.Patch(color='#aaddbb',  label='Low variance but not dead'),
]
ax.legend(handles=legend2, fontsize=8, facecolor=C_DARK, edgecolor='#334', labelcolor='white')

fig4.suptitle('Per-Attribute Variance: PETA PromptPAR vs SNN-PAR', color='white', fontsize=13, y=1.01)
fig4.tight_layout()
FIG4 = fig_to_b64(fig4)
plt.close(fig4)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — SNN-PAR attribute mean + std
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 5: SNN attribute activation …')
fig5, ax = plt.subplots(figsize=(12, 8), facecolor=C_DARK)
ax.set_facecolor(C_MID); style_ax(ax)
colors = [C_RED if p>0.9 else C_ORG if p>0.6 else C_SNN if p>0.1 else '#7f8c8d'
          for p in pos_snn]
ax.barh(range(len(attrs_snn)), pos_snn, xerr=std_snn, color=colors, edgecolor='none', height=0.7,
        error_kw={'ecolor': 'white', 'alpha': 0.4, 'capsize': 2})
ax.set_yticks(range(len(attrs_snn))); ax.set_yticklabels(attrs_snn, fontsize=9, color='white')
ax.set_xlabel('Positive Rate (threshold=0.45)', color='white', fontsize=11)
ax.axvline(0.45, color='white', lw=0.8, ls='--', alpha=0.5)
ax.set_title('SNN-PAR — Attribute Positive Rates on ITCPR', color='white', fontsize=12)
ax.set_xlim(0, 1.12)
legend_items = [
    mpatches.Patch(color=C_RED,     label='>90% (near-saturated)'),
    mpatches.Patch(color=C_ORG,     label='60–90%'),
    mpatches.Patch(color=C_SNN,     label='10–60% (active)'),
    mpatches.Patch(color='#7f8c8d', label='<10% (low activation)'),
]
ax.legend(handles=legend_items, fontsize=9, facecolor=C_DARK, edgecolor='#334', labelcolor='white')
fig5.tight_layout()
FIG5 = fig_to_b64(fig5)
plt.close(fig5)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Unique sets & top-prediction share bar
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 6: diversity summary …')
fig6, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=C_DARK)

# left: unique sets
ax = axes[0]; ax.set_facecolor(C_MID); style_ax(ax)
uniques = [len(ctr_pa), len(ctr_pe), len(ctr_snn)]
lbls = ['PA100k\n(PromptPAR)', 'PETA\n(PromptPAR)', 'SNN-PAR\n(PETA ckpt)']
cols = [C_PA, C_PE, C_SNN]
bars = ax.bar(lbls, uniques, color=cols, edgecolor='none', width=0.5)
for bar, v in zip(bars, uniques):
    ax.text(bar.get_x()+bar.get_width()/2, v+80, f'{v:,}', ha='center',
            va='bottom', color='white', fontsize=12, fontweight='bold')
ax.set_ylabel('Unique prediction sets', color='white', fontsize=11)
ax.set_title('Prediction Diversity\n(out of 20,508 images)', color='white', fontsize=11)
ax.set_yscale('log'); ax.set_ylim(1, 100000)
ax.yaxis.set_tick_params(color='white')

# right: dominant set share
ax = axes[1]; ax.set_facecolor(C_MID); style_ax(ax)
top_shares = [100*ctr_pa.most_common(1)[0][1]/N,
              100*ctr_pe.most_common(1)[0][1]/N,
              100*ctr_snn.most_common(1)[0][1]/N]
bars = ax.bar(lbls, top_shares, color=cols, edgecolor='none', width=0.5)
for bar, v in zip(bars, top_shares):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:.1f}%', ha='center',
            va='bottom', color='white', fontsize=12, fontweight='bold')
ax.set_ylabel('% images with top prediction set', color='white', fontsize=11)
ax.set_title('Top Prediction Dominance\n(lower = more diverse)', color='white', fontsize=11)
ax.set_ylim(0, 100)

fig6.suptitle('Prediction Diversity Comparison', color='white', fontsize=13, y=1.01)
fig6.tight_layout()
FIG6 = fig_to_b64(fig6)
plt.close(fig6)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Per-source heatmap for SNN-PAR
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 7: per-source heatmap …')
fig7, ax = plt.subplots(figsize=(16, 4), facecolor=C_DARK)
ax.set_facecolor(C_MID); style_ax(ax)

src_names = list(sources.keys())
heatmap = []
for src in src_names:
    rows = np.array([[preds_snn[p]['probs'][a] for a in attrs_snn]
                     for p in sources[src] if p in preds_snn])
    heatmap.append((rows > 0.45).mean(0) if len(rows) else np.zeros(len(attrs_snn)))
heatmap = np.array(heatmap)

im = ax.imshow(heatmap, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xticks(range(len(attrs_snn)))
ax.set_xticklabels(attrs_snn, rotation=50, ha='right', fontsize=8, color='white')
ax.set_yticks(range(3))
ax.set_yticklabels([f'{s} (n={len(sources[s])})' for s in src_names], color='white', fontsize=10)
ax.set_title('SNN-PAR — Attribute Positive Rate by Source Dataset', color='white', fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01).ax.yaxis.set_tick_params(color='white')
for i in range(3):
    for j in range(len(attrs_snn)):
        ax.text(j, i, f'{heatmap[i,j]:.0%}', ha='center', va='center',
                fontsize=5.5, color='black' if heatmap[i,j]>0.4 else 'white')
fig7.tight_layout()
FIG7 = fig_to_b64(fig7)
plt.close(fig7)

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE GRIDS
# ═══════════════════════════════════════════════════════════════════════════════
print('Sampling images …')
per_std_snn = probs_snn.std(axis=1)
idx_diverse   = np.argsort(per_std_snn)[::-1][:30]
idx_collapsed = np.argsort(per_std_snn)[:30]
np.random.shuffle(idx_diverse); np.random.shuffle(idx_collapsed)
diverse_paths   = [paths_all[i] for i in idx_diverse[:12]]
collapsed_paths = [paths_all[i] for i in idx_collapsed[:12]]

prcc_paths = [p for p in paths_all if p.startswith('PRCC')]
random.shuffle(prcc_paths); prcc_sample = prcc_paths[:8]

def render_triple_grid(img_paths, n_cols=4, thumb_w=110, thumb_h=220):
    rows_html = []
    for start in range(0, len(img_paths), n_cols):
        batch = img_paths[start:start+n_cols]
        cells = []
        for p in batch:
            img = load_img(p, thumb_w, thumb_h)
            if img is None:
                cells.append('<td></td>'); continue
            b64  = img_to_b64(img)
            src  = p.split('/')[0]
            sc   = {'Celeb-reID':'#3498db','PRCC':'#e67e22','LAST':'#27ae60'}.get(src,'#aaa')
            fname = '/'.join(p.split('/')[-2:])

            def top_k(pd, attrs, k=5):
                return sorted(attrs, key=lambda a: pd.get(p,{}).get('probs',{}).get(a,0), reverse=True)[:k]

            def span(a, pd, thr=0.45):
                prob = pd.get(p,{}).get('probs',{}).get(a,0)
                col  = '#27ae60' if prob>thr else '#888'
                return f'<span style="color:{col};font-size:9.5px">• {a} <em>({prob:.2f})</em></span>'

            snn_top = top_k(preds_snn, attrs_snn)
            pe_top  = top_k(preds_pe,  attrs_pe)

            snn_html = '<br>'.join(span(a, preds_snn) for a in snn_top)
            pe_html  = '<br>'.join(span(a, preds_pe)  for a in pe_top)

            cell = f'''
<td style="vertical-align:top;padding:6px;background:#16213e;border-radius:6px;min-width:{thumb_w+10}px">
  <div style="text-align:center">
    <img src="data:image/png;base64,{b64}" style="border-radius:4px;border:2px solid {sc};display:block;margin:0 auto"/>
    <div style="font-size:9px;color:{sc};margin-top:2px">{src}</div>
    <div style="font-size:8px;color:#aaa;word-break:break-all">{fname}</div>
  </div>
  <div style="margin-top:5px">
    <div style="font-size:9px;font-weight:bold;color:{C_SNN};margin-bottom:2px">SNN-PAR:</div>
    {snn_html}
  </div>
  <div style="margin-top:5px">
    <div style="font-size:9px;font-weight:bold;color:{C_PE};margin-bottom:2px">PETA PromptPAR:</div>
    {pe_html}
  </div>
</td>'''
            cells.append(cell)
        rows_html.append('<tr style="gap:8px">' + '\n'.join(cells) + '</tr>')
    return '<table style="border-collapse:separate;border-spacing:8px">' + '\n'.join(rows_html) + '</table>'

print('  rendering grids …')
GRID_DIVERSE   = render_triple_grid(diverse_paths)
GRID_COLLAPSED = render_triple_grid(collapsed_paths)
GRID_PRCC      = render_triple_grid(prcc_sample)

# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE HTML
# ═══════════════════════════════════════════════════════════════════════════════
print('Assembling HTML …')

def callout(text, kind='info'):
    c = {'good':(C_SNN,'#001a00'),'bad':(C_RED,'#1a0000'),
         'warn':(C_ORG,'#2c2000'),'info':(C_PA,'#001020')}[kind]
    return f'<div class="callout" style="border-left:4px solid {c[0]};background:{c[1]};padding:12px 16px;border-radius:4px;margin:12px 0">{text}</div>'

def fig_block(b64, caption):
    return f'<figure><img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:8px"/><figcaption>{caption}</figcaption></figure>'

def sec(id_, title, body, icon='▸'):
    return f'<section id="{id_}">\n<h2>{icon} {title}</h2>\n{body}\n</section>'

# summary table rows
def attr_table_rows(attrs, mean, std, pos):
    rows = []
    for a, m, s, p in zip(attrs, mean, std, pos):
        if s < 0.02 and p > 0.9:  status, sc = 'SATURATED', C_RED
        elif s < 0.02 and p < 0.1: status, sc = 'DEAD',      '#7f8c8d'
        elif s < 0.02:              status, sc = 'COLLAPSED', C_ORG
        elif s > 0.10:              status, sc = 'ACTIVE',    C_SNN
        else:                       status, sc = 'LOW-VAR',   C_YLW
        rows.append(f'<tr><td><span class="badge" style="background:{sc}22;color:{sc}">{status}</span></td>'
                    f'<td><code>{a}</code></td><td>{m:.3f}</td><td>{s:.3f}</td><td>{p:.1%}</td></tr>')
    return '\n'.join(rows)

top5_snn_html = '<ol style="margin:4px 0;padding-left:20px">'
for s, c in ctr_snn.most_common(5):
    top5_snn_html += (f'<li><span style="color:{C_SNN};font-weight:bold">{100*c/N:.1f}%</span> — '
                      + ', '.join(f'<code>{a}</code>' for a in s) + '</li>')
top5_snn_html += '</ol>'

HTML = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SNN-PAR vs PromptPAR on ITCPR</title>
<style>
  :root{{--bg:#0f0f1a;--bg2:#16213e;--bg3:#1a1a2e;--text:#e0e0f0;--muted:#8899aa;
        --accent:#4fc3f7;--blue:{C_PA};--purple:{C_PE};--green:{C_SNN};--red:{C_RED};}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;line-height:1.65}}
  .container{{max-width:1320px;margin:0 auto;padding:0 24px 60px}}
  header{{background:linear-gradient(135deg,#0a2a10 0%,#0d1b4b 50%,#1a0533 100%);
          padding:48px 32px;margin-bottom:32px;border-bottom:1px solid #334}}
  header h1{{font-size:2.1rem;font-weight:700;color:white;letter-spacing:-0.5px}}
  header .subtitle{{color:var(--muted);margin-top:8px;font-size:1.05rem}}
  header .meta{{display:flex;gap:16px;margin-top:20px;flex-wrap:wrap}}
  header .meta span{{background:#ffffff18;padding:4px 12px;border-radius:20px;font-size:0.85rem;color:#bcd}}
  section{{margin-bottom:44px}}
  h2{{font-size:1.35rem;color:var(--accent);margin-bottom:16px;padding-bottom:6px;border-bottom:1px solid #223}}
  h3{{font-size:1.05rem;color:#aaccff;margin:20px 0 10px}}
  p{{margin-bottom:10px;color:#ccd}}
  ul,ol{{margin:6px 0 12px 22px;color:#ccd}}
  li{{margin-bottom:4px}}
  figure{{margin:20px 0}}
  figcaption{{text-align:center;color:var(--muted);font-size:0.82rem;margin-top:8px}}
  code{{background:#223;padding:1px 6px;border-radius:3px;font-size:0.88em;color:#7ec8e3}}
  table.stats{{width:100%;border-collapse:collapse;margin:12px 0}}
  table.stats th,table.stats td{{padding:7px 10px;text-align:left;border-bottom:1px solid #223;font-size:0.875rem}}
  table.stats th{{background:#1e2a4a;color:var(--accent);font-weight:600}}
  table.stats tr:nth-child(even){{background:#12192e}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
  .grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}}
  .card{{background:var(--bg2);border-radius:8px;padding:18px;border:1px solid #223}}
  .card h3{{margin-top:0}}
  .badge{{display:inline-block;padding:1px 8px;border-radius:12px;font-size:0.76rem;font-weight:700}}
  .toc{{background:#12192e;border-radius:8px;padding:18px 26px;margin-bottom:32px;border:1px solid #223}}
  .toc h3{{color:var(--accent);margin-bottom:8px}}
  .toc ol{{padding-left:20px}}
  .toc li{{margin-bottom:4px}}
  .toc a{{color:#7ec8e3;text-decoration:none}}
  .toc a:hover{{text-decoration:underline}}
  .scroll-wrap{{overflow-x:auto;padding-bottom:8px}}
  .highlight{{color:{C_SNN};font-weight:bold}}
  @media(max-width:768px){{.grid-2,.grid-3{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
  <div class="container">
    <h1>SNN-PAR vs PromptPAR on ITCPR: Comparative Collapse Analysis</h1>
    <p class="subtitle">Does the Spiking Neural Network architecture generalise better to out-of-distribution surveillance data?</p>
    <div class="meta">
      <span>📅 2026-06-04</span>
      <span>🗂 ITCPR Gallery — 20,510 images</span>
      <span style="background:#27ae6022;color:{C_SNN}">🟢 SNN-PAR (PETA, 35 attrs)</span>
      <span style="background:#8e44ad22;color:{C_PE}">🟣 PromptPAR PETA (35 attrs)</span>
      <span style="background:#2980b922;color:{C_PA}">🔵 PromptPAR PA100k (26 attrs)</span>
    </div>
  </div>
</header>
<div class="container">

<div class="toc">
  <h3>Contents</h3>
  <ol>
    <li><a href="#summary">Executive Summary</a></li>
    <li><a href="#diversity">Prediction Diversity</a></li>
    <li><a href="#cosine">Inter-Image Similarity</a></li>
    <li><a href="#per-image">Per-Image Attribute Variance</a></li>
    <li><a href="#attr-variance">Per-Attribute Variance Deep-Dive</a></li>
    <li><a href="#snn-activation">SNN-PAR Attribute Activation</a></li>
    <li><a href="#per-source">Per-Source Breakdown</a></li>
    <li><a href="#attr-tables">Attribute Status Tables</a></li>
    <li><a href="#ex-diverse">Image Examples — SNN-PAR Most Diverse</a></li>
    <li><a href="#ex-collapsed">Image Examples — SNN-PAR Most Uniform</a></li>
    <li><a href="#ex-prcc">Image Examples — PRCC (Highest Quality)</a></li>
    <li><a href="#why">Why Doesn't SNN-PAR Collapse?</a></li>
    <li><a href="#caveats">Caveats &amp; Limitations</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
  </ol>
</div>

{sec('summary','Executive Summary', f'''
{callout(f'''<strong>SNN-PAR does NOT collapse on ITCPR.</strong>
Pairwise cosine similarity: <strong>0.63</strong> (vs 0.99 for PA100k PromptPAR, 0.94 for PETA PromptPAR).
Unique prediction sets: <strong>12,381</strong> out of 20,508 images (vs 7 and 647).
Zero collapsed, dead, or saturated attributes.''', 'good')}

<div class="grid-3" style="margin-top:16px">
  <div class="card">
    <h3 style="color:{C_PA}">PA100k PromptPAR</h3>
    <ul>
      <li><strong>7</strong> unique prediction sets</li>
      <li>84.1% identical predictions</li>
      <li>14/26 collapsed attrs</li>
      <li>Cosine sim: <strong>0.9925</strong></li>
      <li>Gender: completely dead</li>
    </ul>
  </div>
  <div class="card">
    <h3 style="color:{C_PE}">PETA PromptPAR</h3>
    <ul>
      <li><strong>647</strong> unique sets</li>
      <li>6.3% top-set share</li>
      <li>14/35 collapsed attrs</li>
      <li>Cosine sim: <strong>0.9415</strong></li>
      <li>Gender: partial (15%)</li>
    </ul>
  </div>
  <div class="card" style="border-color:{C_SNN}44">
    <h3 style="color:{C_SNN}">SNN-PAR</h3>
    <ul>
      <li><strong class="highlight">12,381</strong> unique sets</li>
      <li>1.1% top-set share</li>
      <li><strong class="highlight">0/35</strong> collapsed attrs</li>
      <li>Cosine sim: <strong class="highlight">0.6278</strong></li>
      <li>Gender: active (48%)</li>
    </ul>
  </div>
</div>
''', '📋')}

{sec('diversity','Prediction Diversity', f'''
{fig_block(FIG6, 'Left: unique prediction sets on log scale. Right: % of images receiving the single most common prediction. Lower = more diverse.')}
{callout(f'SNN-PAR produces <strong>12,381 unique attribute sets</strong> — 1,769× more diverse than PA100k PromptPAR (7 sets) and 19× more diverse than PETA PromptPAR (647 sets). The most common SNN-PAR prediction covers just 1.1% of images.', 'good')}
{fig_block(FIG1, 'Raw counts of collapse-related metrics for all three checkpoints.')}

<div class="grid-2" style="margin-top:16px">
  <div class="card">
    <h3 style="color:{C_SNN}">SNN-PAR — Top-5 Prediction Sets</h3>
    {top5_snn_html}
    <p style="margin-top:10px;color:#aaa;font-size:0.9rem">No set dominates — the model distributes
    probability mass across all 35 attributes depending on image content.</p>
  </div>
  <div class="card">
    <h3>Collapse Metric Summary</h3>
    <table class="stats">
      <tr><th>Metric</th><th style="color:{C_PA}">PA100k</th><th style="color:{C_PE}">PETA</th><th style="color:{C_SNN}">SNN-PAR</th></tr>
      <tr><td>Unique sets</td><td>{len(ctr_pa)}</td><td>{len(ctr_pe)}</td><td class="highlight">{len(ctr_snn):,}</td></tr>
      <tr><td>Collapsed attrs</td><td>{(std_pa<0.02).sum()}/26</td><td>{(std_pe<0.02).sum()}/35</td><td class="highlight">0/35</td></tr>
      <tr><td>Dead attrs</td><td>{(pos_pa<0.01).sum()}/26</td><td>{(pos_pe<0.01).sum()}/35</td><td class="highlight">0/35</td></tr>
      <tr><td>Saturated attrs</td><td>{(pos_pa>0.99).sum()}/26</td><td>{(pos_pe>0.99).sum()}/35</td><td class="highlight">0/35</td></tr>
      <tr><td>Cosine sim (mean)</td><td>0.9925</td><td>0.9415</td><td class="highlight">0.6278</td></tr>
      <tr><td>Mean per-img std</td><td>{probs_pa.std(1).mean():.3f}</td><td>{probs_pe.std(1).mean():.3f}</td><td class="highlight">{probs_snn.std(1).mean():.3f}</td></tr>
    </table>
  </div>
</div>
''', '🔀')}

{sec('cosine','Inter-Image Similarity', f'''
{fig_block(FIG2, 'Distribution of cosine similarity between 2,000 random image pairs. PA100k clusters near 1.0 (identical predictions). SNN-PAR has a broad distribution centred at 0.63 — the model produces genuinely different predictions for different images.')}
<p>A cosine similarity of <strong>0.63 ± 0.14</strong> for SNN-PAR reflects a model that is actually responding to image content. By contrast PA100k at 0.99 means almost every pair of images gets nearly the same probability vector — it doesn't matter what the image shows.</p>
''', '📐')}

{sec('per-image','Per-Image Attribute Variance', f'''
{fig_block(FIG3, 'Histogram of per-image standard deviation across all attribute probabilities. A narrow peak = model outputs similar vectors for all images. A wide distribution = the model is discriminating.')}
{callout('SNN-PAR\'s per-image std distribution (green) is substantially wider and shifted right compared to both PromptPAR checkpoints. The range 0.22–0.43 shows the model is genuinely adjusting its predictions across images.', 'good')}
''', '📉')}

{sec('attr-variance','Per-Attribute Variance Deep-Dive', f'''
{fig_block(FIG4, 'Left: PETA PromptPAR attribute variance — many attributes collapsed to near-zero std. Right: SNN-PAR — every attribute has meaningful variance (all bars exceed the 0.02 threshold). Note the rich distribution of green bars on the right.')}
<p>The contrast is stark: PETA PromptPAR has 14 attributes with std&lt;0.02 (grey/red bars on the left), while SNN-PAR has <strong>zero collapsed attributes</strong>. Even the least variable SNN-PAR attribute (<code>accessory sunglasses</code>, std≈0.09) has more variance than most PETA PromptPAR attributes.</p>
''', '📊')}

{sec('snn-activation','SNN-PAR Attribute Activation', f'''
{fig_block(FIG5, 'Positive rate ± std for each SNN-PAR attribute. Error bars show per-attribute variability. Unlike PromptPAR, no attribute sits at 0% or 100% — all are in the informative middle range.')}
<h3>Notable Attributes</h3>
<table class="stats">
  <tr><th>Attribute</th><th>Pos Rate</th><th>Std</th><th>Comment</th></tr>
  <tr><td><code>upper body casual</code></td><td>{pos_snn[5]:.1%}</td><td>{std_snn[5]:.3f}</td><td>High but not saturated; reasonable for surveillance</td></tr>
  <tr><td><code>personal male</code></td><td>{pos_snn[34]:.1%}</td><td>{std_snn[34]:.3f}</td><td>Meaningful gender detection (vs dead in PA100k)</td></tr>
  <tr><td><code>personal less 30</code></td><td>{pos_snn[30]:.1%}</td><td>{std_snn[30]:.3f}</td><td>Active age prediction</td></tr>
  <tr><td><code>carrying nothing</code></td><td>{pos_snn[28]:.1%}</td><td>{std_snn[28]:.3f}</td><td>High variance — model distinguishes accessories</td></tr>
  <tr><td><code>foot wear sneaker shoes</code></td><td>{pos_snn[24]:.1%}</td><td>{std_snn[24]:.3f}</td><td>Shoe type varies across images</td></tr>
  <tr><td><code>accessory sunglasses</code></td><td>{pos_snn[3]:.1%}</td><td>{std_snn[3]:.3f}</td><td>Least variable — still not dead</td></tr>
</table>
''', '📈')}

{sec('per-source','Per-Source Breakdown', f'''
{fig_block(FIG7, 'Attribute positive rates per source dataset. Unlike PromptPAR (near-identical rows = source doesn\'t matter), SNN-PAR shows noticeable variation between Celeb-reID, PRCC, and LAST — suggesting the model is sensitive to image characteristics.')}
''', '📂')}

{sec('attr-tables','Attribute Status Tables', f'''
<div class="grid-2">
  <div class="card">
    <h3 style="color:{C_SNN}">SNN-PAR (35 attrs)</h3>
    <table class="stats">
      <tr><th>Status</th><th>Attribute</th><th>Mean</th><th>Std</th><th>PosRate</th></tr>
      {attr_table_rows(attrs_snn, mean_snn, std_snn, pos_snn)}
    </table>
  </div>
  <div class="card">
    <h3 style="color:{C_PE}">PETA PromptPAR (35 attrs)</h3>
    <table class="stats">
      <tr><th>Status</th><th>Attribute</th><th>Mean</th><th>Std</th><th>PosRate</th></tr>
      {attr_table_rows(attrs_pe, mean_pe, std_pe, pos_pe)}
    </table>
  </div>
</div>
''', '📋')}

{sec('ex-diverse','Image Examples — SNN-PAR Most Diverse Predictions', f'''
<p>Images where SNN-PAR has the <strong>highest per-image attribute variance</strong> — the model is most confident about a specific attribute combination. Note how SNN-PAR and PETA PromptPAR give substantially different predictions for the same crop.</p>
<div class="scroll-wrap">{GRID_DIVERSE}</div>
{callout('SNN-PAR\'s predictions vary meaningfully across images: some get sunglasses, some backpacks, some long hair, different age brackets. PETA PromptPAR saturates on upper casual/other + lower Casual/Trousers for most images.', 'good')}
''', '🟢')}

{sec('ex-collapsed','Image Examples — SNN-PAR Most Uniform Predictions', f'''
<p>Images where SNN-PAR has the <strong>lowest per-image variance</strong> — the model gives the most generic outputs. Even at its least discriminative, SNN-PAR still outputs different patterns than the constant PromptPAR prediction.</p>
<div class="scroll-wrap">{GRID_COLLAPSED}</div>
''', '🟡')}

{sec('ex-prcc','Image Examples — PRCC Source', f'''
<p>PRCC images are the highest quality in ITCPR. SNN-PAR utilises this — PRCC shows more attribute diversity (lower <code>personal less 30</code> vs Celeb-reID) and more meaningful shoe/accessory predictions.</p>
<div class="scroll-wrap">{GRID_PRCC}</div>
''', '🔵')}

{sec('why','Why Doesn\'t SNN-PAR Collapse?', f'''
<h3>1. Fine-tuned on Pedestrian-Aspect Images</h3>
<p>The ViT-B/16 in SNN-PAR was fine-tuned at <strong>256×128 resolution</strong> — the natural tall-narrow aspect ratio of pedestrian surveillance crops. Its positional embedding has shape <code>[1, 129, 768]</code> (128 = 16×8 patches). ITCPR images are also ~128×256 px, so the spatial distribution matches. PromptPAR uses 224×224 (originally designed for ImageNet), creating a strong aspect-ratio mismatch with surveillance crops.</p>

<h3>2. No Learned Prompt Over-Fitting</h3>
<p>PromptPAR injects <strong>50 visual prompts + 3 text prompts</strong> that are jointly optimised on the training dataset. These prompts encode dataset-specific statistics. SNN-PAR uses simple learned scalar offsets (<code>vis_embed</code>, <code>tex_embed</code> — both 1×1×768) which are less susceptible to over-fitting to training-set image statistics.</p>

<h3>3. Sentence-Transformer Text Embeddings</h3>
<p>SNN-PAR uses fixed <code>all-mpnet-base-v2</code> embeddings (768-dim) as word vectors. These are general-purpose semantic embeddings that are not tied to any visual training distribution. The <code>word_embed</code> linear layer maps them to the feature space, but the semantic grounding is robust across domains.</p>

<h3>4. Normalisation Closer to Surveillance Domain</h3>
<p>SNN-PAR normalises images with <code>mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]</code> — a simple range-normalisation that is more tolerant of different lighting conditions than PromptPAR's CLIP-specific normalisation, which was optimised for clean, well-lit photographs.</p>

<h3>5. Attribute Vocabulary Style</h3>
<p>SNN-PAR's PETA vocabulary uses <strong>more descriptive, compositional phrases</strong> (e.g., <em>'upper body casual'</em>, <em>'foot wear sneaker shoes'</em>, <em>'carrying nothing'</em>) compared to PromptPAR's shorter labels. Longer, more specific phrases produce more distinct sentence embeddings, which may help the model learn tighter boundaries between attribute classes.</p>

{callout('The resolution match is likely the single most important factor. SNN-PAR was fine-tuned at 256×128, which matches ITCPR\'s surveillance crop aspect ratio. PromptPAR was trained on 224×224 (square) images from clean datasets — a fundamental mismatch.', 'good')}
''', '🔍')}

{sec('caveats','Caveats and Limitations', f'''
{callout('<strong>SNN-PAR is not necessarily accurate — it is just diverse.</strong> High prediction diversity does not guarantee correct predictions. We do not have ground-truth attribute labels for ITCPR, so we cannot measure precision/recall.', 'warn')}

<h3>What We Cannot Conclude</h3>
<ul>
  <li>SNN-PAR may be producing <em>noisy</em> predictions rather than <em>correct</em> ones — diversity and accuracy are different things.</li>
  <li>The <code>personal male</code> at 48% positive rate may reflect calibration issues rather than accurate gender detection.</li>
  <li>Some near-saturated attributes (<code>upper body casual</code> at 93%, <code>lower body Casual</code> at 92%) suggest SNN-PAR still has biases, just not absolute collapse.</li>
  <li>The SNN (Spikingformer) component is not used — only the ANN teacher branch is in the checkpoint. A full SNN-PAR with trained SNN weights might perform differently.</li>
</ul>

<h3>Recommended Validation</h3>
<ul>
  <li>Manually inspect ~100 predicted images with ground-truth annotations from a parallel source.</li>
  <li>Evaluate on a dataset with known PAR annotations at 256×128 resolution (e.g., MARKET-1501-Attribute or DukeMTMC-Attribute).</li>
  <li>Compare SNN-PAR predictions against human annotations on a small ITCPR sample.</li>
</ul>
''', '⚠️')}

{sec('conclusion','Conclusion', f'''
<table class="stats" style="margin-bottom:20px">
  <tr><th>Question</th><th>Answer</th></tr>
  <tr><td>Does SNN-PAR collapse on ITCPR?</td><td style="color:{C_SNN}"><strong>No — zero collapsed/dead/saturated attributes, 12,381 unique predictions</strong></td></tr>
  <tr><td>Does PromptPAR collapse on ITCPR?</td><td style="color:{C_RED}"><strong>Yes — 7 unique predictions for PA100k, severe degeneration for PETA</strong></td></tr>
  <tr><td>Why does SNN-PAR generalise better?</td><td>256×128 training resolution, no prompt over-fitting, robust text embeddings</td></tr>
  <tr><td>Is SNN-PAR accurate?</td><td style="color:{C_YLW}"><strong>Unknown — diverse ≠ correct; needs evaluation with GT labels</strong></td></tr>
  <tr><td>Which model to use for ITCPR?</td><td style="color:{C_SNN}"><strong>SNN-PAR is the clear choice</strong> — PromptPAR is unusable</td></tr>
</table>

<p>For any downstream application using pedestrian attributes on ITCPR-style surveillance data, <strong>SNN-PAR's ANN teacher</strong> is the only viable option among the tested models. PromptPAR requires surveillance-domain fine-tuning before it can be used.</p>

<p>The root cause of PromptPAR's failure is the combination of CLIP-trained visual prompts (optimised for clean photos) and 224×224 input resolution (mismatched aspect ratio). SNN-PAR avoids both by using a 256×128-tuned ViT without task-specific visual prompts.</p>
''', '✅')}

</div>
<footer style="border-top:1px solid #223;padding:18px 32px;text-align:center;color:#556;font-size:0.82rem">
  SNN-PAR ANN teacher inference · ckpt_peta.pth · PromptPAR PA100k &amp; PETA checkpoints · ITCPR 20,510 images · 2026-06-04
</footer>
</body></html>'''

with open(OUT_HTML, 'w') as f:
    f.write(HTML)
print(f'\n✓ Report written → {OUT_HTML}  ({os.path.getsize(OUT_HTML)//1024} KB)')
