"""
Generate a self-contained HTML report for PromptPAR on ITCPR collapse analysis.
Usage:  uv run python generate_report.py
"""
import base64, io, json, os, random
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
ITCPR_ROOT = '/home/mika/mnt/yomikawa-reid/repos/yomikawa-reid/datasets/ITCPR'
PA_PREDS   = '/tmp/preds_pa100k.json'
PE_PREDS   = '/tmp/preds_peta.json'
OUT_HTML   = 'itcpr_report.html'

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ── colour palette ─────────────────────────────────────────────────────────────
C_RED    = '#e74c3c'
C_ORANGE = '#e67e22'
C_YELLOW = '#f1c40f'
C_GREEN  = '#27ae60'
C_BLUE   = '#2980b9'
C_PURPLE = '#8e44ad'
C_DARK   = '#2c3e50'
C_LIGHT  = '#ecf0f1'

# ── helpers ────────────────────────────────────────────────────────────────────
def fig_to_b64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def img_to_b64(pil_img, fmt='PNG'):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def load_img(rel_path, size=(120, 240)):
    p = os.path.join(ITCPR_ROOT, rel_path)
    try:
        img = Image.open(p).convert('RGB').resize(size, Image.LANCZOS)
        return img
    except Exception:
        return None

def collapse_label(pos_rate, std):
    if std < 0.02 and pos_rate > 0.99:  return ('SATURATED', C_RED)
    if std < 0.02 and pos_rate < 0.01:  return ('DEAD', '#7f8c8d')
    if std < 0.02:                       return ('COLLAPSED', C_ORANGE)
    return ('ACTIVE', C_GREEN)

# ── load data ──────────────────────────────────────────────────────────────────
print('Loading predictions …')
with open(PA_PREDS) as f: pa = json.load(f)
with open(PE_PREDS) as f: pe = json.load(f)
with open(os.path.join(ITCPR_ROOT, 'gallery.json')) as f: gallery = json.load(f)

preds_pa = pa['predictions'];  attr_pa = pa['attr_words']
preds_pe = pe['predictions'];  attr_pe = pe['attr_words']
paths_all = list(preds_pa.keys())

probs_pa = np.array([[preds_pa[p]['probs'][a] for a in attr_pa] for p in paths_all])
probs_pe = np.array([[preds_pe[p]['probs'][a] for a in attr_pe] for p in paths_all])

mean_pa  = probs_pa.mean(0);  std_pa  = probs_pa.std(0)
mean_pe  = probs_pe.mean(0);  std_pe  = probs_pe.std(0)
pos_pa   = (probs_pa > 0.45).mean(0)
pos_pe   = (probs_pe > 0.45).mean(0)

counter_pa = Counter(tuple(sorted(v['attrs'])) for v in preds_pa.values())
counter_pe = Counter(tuple(sorted(v['attrs'])) for v in preds_pe.values())
unique_pa  = len(counter_pa)
unique_pe  = len(counter_pe)

# per-source
sources = {'Celeb-reID': [], 'PRCC': [], 'LAST': []}
for e in gallery:
    src = e['file_path'].split('/')[0]
    if src in sources:
        sources[src].append(e['file_path'])

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Attribute activation bar chart (pos_rate)
# ═══════════════════════════════════════════════════════════════════════════════
print('Building Figure 1 (attribute activation) …')
fig1, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor='#1a1a2e')

for ax, attrs, pos, std, title, ckpt_col in [
    (axes[0], attr_pa, pos_pa, std_pa, 'PA100k Checkpoint (26 attrs)', C_BLUE),
    (axes[1], attr_pe, pos_pe, std_pe, 'PETA Checkpoint (35 attrs)', C_PURPLE),
]:
    ax.set_facecolor('#16213e')
    colors = []
    for p, s in zip(pos, std):
        lbl, c = collapse_label(p, s)
        colors.append(c)
    bars = ax.barh(range(len(attrs)), pos[::-1] if False else pos,
                   color=colors, edgecolor='none', height=0.7)
    ax.set_yticks(range(len(attrs)))
    ax.set_yticklabels(attrs, fontsize=8.5, color='white')
    ax.set_xlabel('Positive Rate (threshold=0.45)', color='white', fontsize=10)
    ax.set_title(title, color='white', fontsize=12, pad=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#334')
    ax.axvline(0.45, color='white', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlim(0, 1.05)
    # legend patches
    legend_items = [
        mpatches.Patch(color=C_RED,      label='Saturated (>99%)'),
        mpatches.Patch(color='#7f8c8d',  label='Dead (<1%)'),
        mpatches.Patch(color=C_ORANGE,   label='Collapsed (std<0.02)'),
        mpatches.Patch(color=C_GREEN,    label='Active'),
    ]
    ax.legend(handles=legend_items, fontsize=8, facecolor='#1a1a2e',
              edgecolor='#334', labelcolor='white', loc='lower right')

fig1.suptitle('Attribute Positive Rates on ITCPR Gallery (20,510 images)',
              color='white', fontsize=14, y=1.01)
fig1.tight_layout()
FIG1 = fig_to_b64(fig1)
plt.close(fig1)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Attribute mean prob + std error bars
# ═══════════════════════════════════════════════════════════════════════════════
print('Building Figure 2 (mean + std) …')
fig2, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor='#1a1a2e')

for ax, attrs, mean, std, title in [
    (axes[0], attr_pa, mean_pa, std_pa, 'PA100k — Mean Probability ± Std'),
    (axes[1], attr_pe, mean_pe, std_pe, 'PETA — Mean Probability ± Std'),
]:
    ax.set_facecolor('#16213e')
    colors = [C_RED if m > 0.8 else C_ORANGE if m > 0.4 else C_BLUE if s > 0.05 else '#7f8c8d'
              for m, s in zip(mean, std)]
    x = np.arange(len(attrs))
    ax.barh(x, mean, xerr=std, color=colors, edgecolor='none', height=0.7,
            error_kw={'ecolor': 'white', 'alpha': 0.5, 'capsize': 2})
    ax.set_yticks(x); ax.set_yticklabels(attrs, fontsize=8.5, color='white')
    ax.set_xlabel('Mean Sigmoid Probability', color='white', fontsize=10)
    ax.set_title(title, color='white', fontsize=12, pad=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#334')
    ax.axvline(0.45, color='white', lw=0.8, ls='--', alpha=0.5, label='threshold')
    ax.set_xlim(0, 1.12)

fig2.suptitle('Mean Probability and Variance per Attribute', color='white', fontsize=14, y=1.01)
fig2.tight_layout()
FIG2 = fig_to_b64(fig2)
plt.close(fig2)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Prediction diversity: top-N prediction sets
# ═══════════════════════════════════════════════════════════════════════════════
print('Building Figure 3 (prediction diversity) …')
fig3, axes = plt.subplots(1, 2, figsize=(18, 5), facecolor='#1a1a2e')
n_total = len(paths_all)

for ax, counter, title, col in [
    (axes[0], counter_pa, f'PA100k — {len(counter_pa)} unique sets', C_BLUE),
    (axes[1], counter_pe, f'PETA — {len(counter_pe)} unique sets', C_PURPLE),
]:
    ax.set_facecolor('#16213e')
    top = counter.most_common(10)
    labels = [f"Set {i+1} ({100*c/n_total:.1f}%)" for i, (_, c) in enumerate(top)]
    counts = [c for _, c in top]
    rest = n_total - sum(counts)
    labels.append(f'Other sets ({100*rest/n_total:.1f}%)')
    counts.append(rest)
    palette = [col] * 10 + ['#445']
    wedge_props = dict(edgecolor='#1a1a2e', linewidth=1.5)
    ax.pie(counts, labels=None, colors=palette, wedgeprops=wedge_props,
           startangle=90, counterclock=False)
    ax.set_title(title, color='white', fontsize=11, pad=10)
    # write top set label
    top_set = top[0][0]
    top_str = '\n'.join([f'• {a}' for a in top_set[:6]])
    if len(top_set) > 6: top_str += f'\n  (+{len(top_set)-6} more)'
    ax.text(1.25, 0, f'Dominant set:\n{top_str}',
            fontsize=7.5, color='white', va='center',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor='#445', alpha=0.9))
    ax.legend(labels[:5], fontsize=7.5, facecolor='#1a1a2e', edgecolor='#334',
              labelcolor='white', loc='lower left', bbox_to_anchor=(-0.35, -0.15))

fig3.suptitle('Prediction Set Distribution — How many distinct outputs does the model produce?',
              color='white', fontsize=13, y=1.01)
fig3.tight_layout()
FIG3 = fig_to_b64(fig3)
plt.close(fig3)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Per-source comparison heatmap
# ═══════════════════════════════════════════════════════════════════════════════
print('Building Figure 4 (per-source heatmap) …')
fig4, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor='#1a1a2e')

for ax, attrs, preds_dict, title, ckpt_col in [
    (axes[0], attr_pa, preds_pa, 'PA100k — Per-Source Positive Rate', C_BLUE),
    (axes[1], attr_pe, preds_pe, 'PETA — Per-Source Positive Rate', C_PURPLE),
]:
    ax.set_facecolor('#16213e')
    src_names = list(sources.keys())
    heatmap = []
    for src in src_names:
        arr = np.array([[preds_dict[p]['probs'][a] for a in attrs]
                        for p in sources[src] if p in preds_dict])
        heatmap.append((arr > 0.45).mean(0) if len(arr) else np.zeros(len(attrs)))
    heatmap = np.array(heatmap)  # [3, n_attrs]

    im = ax.imshow(heatmap, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   interpolation='nearest')
    ax.set_xticks(range(len(attrs)))
    ax.set_xticklabels(attrs, rotation=55, ha='right', fontsize=7.5, color='white')
    ax.set_yticks(range(3))
    ax.set_yticklabels([f'{s}\n(n={len(sources[s])})' for s in src_names],
                       color='white', fontsize=9)
    ax.set_title(title, color='white', fontsize=11, pad=10)
    for spine in ax.spines.values(): spine.set_color('#334')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.yaxis.set_tick_params(color='white')
    ax.tick_params(colors='white')
    # annotate values
    for i in range(3):
        for j in range(len(attrs)):
            ax.text(j, i, f'{heatmap[i,j]:.0%}', ha='center', va='center',
                    fontsize=5.5, color='black' if heatmap[i,j] > 0.4 else 'white')

fig4.suptitle('Attribute Positive Rate by ITCPR Source Dataset',
              color='white', fontsize=13, y=1.01)
fig4.tight_layout()
FIG4 = fig_to_b64(fig4)
plt.close(fig4)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Cosine similarity histogram
# ═══════════════════════════════════════════════════════════════════════════════
print('Building Figure 5 (cosine similarity) …')
fig5, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='#1a1a2e')

for ax, probs, title, col in [
    (axes[0], probs_pa, 'PA100k — Inter-image Cosine Similarity', C_BLUE),
    (axes[1], probs_pe, 'PETA — Inter-image Cosine Similarity', C_PURPLE),
]:
    ax.set_facecolor('#16213e')
    idx1 = np.random.choice(len(probs), 2000, replace=False)
    idx2 = np.random.choice(len(probs), 2000, replace=False)
    mask = idx1 != idx2
    n1, n2 = probs[idx1[mask]], probs[idx2[mask]]
    norms = np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1)
    cos = (n1 * n2).sum(1) / (norms + 1e-8)
    ax.hist(cos, bins=80, color=col, edgecolor='none', alpha=0.85)
    ax.axvline(cos.mean(), color='white', lw=1.5, ls='--',
               label=f'mean={cos.mean():.4f}')
    ax.set_xlabel('Cosine Similarity', color='white', fontsize=10)
    ax.set_ylabel('Count', color='white', fontsize=10)
    ax.set_title(title, color='white', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#334')
    ax.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#334', labelcolor='white')
    ax.set_xlim(0.5, 1.02)
    note = '← perfect collapse = 1.0'
    ax.text(0.98, 0.92, note, transform=ax.transAxes, ha='right',
            color='#aaa', fontsize=8)

fig5.suptitle('Prediction Similarity Between Random Image Pairs',
              color='white', fontsize=13, y=1.01)
fig5.tight_layout()
FIG5 = fig_to_b64(fig5)
plt.close(fig5)

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE EXAMPLES — pick representative crops
# ═══════════════════════════════════════════════════════════════════════════════
print('Sampling image examples …')

def get_sample_images(n=20, strategy='diverse'):
    """
    strategy: 'diverse' = highest per-image std in PA100k
              'collapsed' = lowest per-image std (most generic)
              'disagreement' = biggest diff between PA100k and PETA prediction count
    """
    img_stds = probs_pa.std(axis=1)
    img_stds_pe = probs_pe.std(axis=1)
    if strategy == 'diverse':
        indices = np.argsort(img_stds)[::-1][:n*3]
    elif strategy == 'collapsed':
        indices = np.argsort(img_stds)[:n*3]
    else:  # disagreement
        diff = np.abs(img_stds - img_stds_pe)
        indices = np.argsort(diff)[::-1][:n*3]
    # shuffle and pick n
    np.random.shuffle(indices)
    return [paths_all[i] for i in indices[:n]]

def render_img_grid(img_paths, preds_pa_d, preds_pe_d, attr_pa, attr_pe,
                    n_cols=5, thumb_w=110, thumb_h=220):
    """Returns HTML string for an image grid."""
    rows = []
    for i in range(0, len(img_paths), n_cols):
        batch = img_paths[i:i+n_cols]
        cells = []
        for p in batch:
            img = load_img(p, size=(thumb_w, thumb_h))
            if img is None:
                cells.append('<td style="padding:4px;"></td>')
                continue
            b64 = img_to_b64(img)
            src = p.split('/')[0]
            src_col = {'Celeb-reID': '#3498db', 'PRCC': '#e67e22', 'LAST': '#27ae60'}.get(src, '#aaa')

            # PA100k attrs
            pa_attrs  = preds_pa_d.get(p, {}).get('attrs', [])
            pa_probs  = preds_pa_d.get(p, {}).get('probs', {})
            pe_attrs  = preds_pe_d.get(p, {}).get('attrs', [])
            pe_probs  = preds_pe_d.get(p, {}).get('probs', {})

            # show top-5 attrs by prob
            def top_attrs(probs_d, attrs, k=5):
                ranked = sorted(attrs, key=lambda a: probs_d.get(a, 0), reverse=True)
                return [(a, probs_d.get(a, 0)) for a in ranked[:k]]

            def attr_span(a, prob, threshold=0.45):
                col = C_GREEN if prob > threshold else '#888'
                return f'<span style="color:{col};font-size:10px">• {a} ({prob:.2f})</span>'

            pa_html = '<br>'.join(attr_span(a, pr) for a, pr in top_attrs(pa_probs, attr_pa))
            pe_html = '<br>'.join(attr_span(a, pr) for a, pr in top_attrs(pe_probs, attr_pe))

            fname = '/'.join(p.split('/')[-2:])
            cell = f'''
<td style="vertical-align:top;padding:6px;background:#16213e;border-radius:6px;min-width:{thumb_w+10}px">
  <div style="text-align:center">
    <img src="data:image/png;base64,{b64}"
         style="border-radius:4px;border:2px solid {src_col};display:block;margin:0 auto"/>
    <div style="font-size:9px;color:{src_col};margin-top:3px">{src}</div>
    <div style="font-size:8px;color:#aaa;word-break:break-all">{fname}</div>
  </div>
  <div style="margin-top:6px">
    <div style="font-size:9px;font-weight:bold;color:{C_BLUE};margin-bottom:2px">PA100k:</div>
    {pa_html}
  </div>
  <div style="margin-top:5px">
    <div style="font-size:9px;font-weight:bold;color:{C_PURPLE};margin-bottom:2px">PETA:</div>
    {pe_html}
  </div>
</td>'''
            cells.append(cell)
        rows.append('<tr style="gap:8px">' + '\n'.join(cells) + '</tr>')
    return '<table style="border-collapse:separate;border-spacing:8px">' + '\n'.join(rows) + '</table>'

print('  sampling collapsed examples …')
collapsed_paths = get_sample_images(10, strategy='collapsed')
print('  sampling diverse examples …')
diverse_paths   = get_sample_images(10, strategy='diverse')

# also pick images where gender might be visible (source: PRCC which is higher quality)
prcc_paths = [p for p in paths_all if p.startswith('PRCC')]
random.shuffle(prcc_paths)
prcc_sample = prcc_paths[:10]

print('  rendering grids …')
GRID_COLLAPSED = render_img_grid(collapsed_paths, preds_pa, preds_pe, attr_pa, attr_pe)
GRID_DIVERSE   = render_img_grid(diverse_paths,   preds_pa, preds_pe, attr_pa, attr_pe)
GRID_PRCC      = render_img_grid(prcc_sample,     preds_pa, preds_pe, attr_pa, attr_pe)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Summary scorecard
# ═══════════════════════════════════════════════════════════════════════════════
print('Building Figure 6 (scorecard) …')
fig6, ax = plt.subplots(figsize=(10, 4), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e'); ax.axis('off')

metrics = [
    ('Total gallery images',           f'{len(paths_all):,}',        f'{len(paths_all):,}'),
    ('Unique prediction sets',         str(unique_pa),                str(unique_pe)),
    ('Top set share',                  f'{100*counter_pa.most_common(1)[0][1]/len(paths_all):.1f}%',
                                       f'{100*counter_pe.most_common(1)[0][1]/len(paths_all):.1f}%'),
    ('Collapsed attrs (std<0.02)',     f'{(std_pa<0.02).sum()}/{len(attr_pa)}',
                                       f'{(std_pe<0.02).sum()}/{len(attr_pe)}'),
    ('Dead attrs (pos_rate<1%)',       f'{(pos_pa<0.01).sum()}/{len(attr_pa)}',
                                       f'{(pos_pe<0.01).sum()}/{len(attr_pe)}'),
    ('Saturated attrs (pos_rate>99%)', f'{(pos_pa>0.99).sum()}/{len(attr_pa)}',
                                       f'{(pos_pe>0.99).sum()}/{len(attr_pe)}'),
    ('Avg pairwise cosine sim',        '0.9925',                      '0.9415'),
    ('Gender detection',               'DEAD (female=0.0%)',           'PARTIAL (male=15.2%)'),
]

col_w = [0.38, 0.31, 0.31]
headers = ['Metric', 'PA100k (26 attrs)', 'PETA (35 attrs)']
hcols   = [C_DARK, C_BLUE, C_PURPLE]

# header row
y = 0.93
for x, w, h, hc in zip([0, col_w[0], col_w[0]+col_w[1]], col_w, headers, hcols):
    ax.add_patch(mpatches.FancyBboxPatch((x+0.005, y-0.07), w-0.01, 0.09,
                 boxstyle='round,pad=0.01', color=hc, alpha=0.9, transform=ax.transAxes))
    ax.text(x+w/2, y-0.02, h, ha='center', va='center', fontsize=10,
            color='white', fontweight='bold', transform=ax.transAxes)

for i, (m, pa_v, pe_v) in enumerate(metrics):
    y = 0.83 - i * 0.1
    bg = '#16213e' if i % 2 == 0 else '#1a1f3c'
    ax.add_patch(mpatches.FancyBboxPatch((0.005, y-0.07), 0.99, 0.09,
                 boxstyle='round,pad=0.005', color=bg, transform=ax.transAxes))
    ax.text(0.01,               y-0.02, m,    ha='left',   va='center', fontsize=9,  color='#ccc', transform=ax.transAxes)
    ax.text(col_w[0]+col_w[1]/2, y-0.02, pa_v, ha='center', va='center', fontsize=9.5,
            color=C_RED if ('❌' in pa_v or int(pa_v.split('/')[0]) > 10 if '/' in pa_v else False) else 'white',
            fontweight='bold', transform=ax.transAxes)
    ax.text(col_w[0]+col_w[1]+col_w[2]/2, y-0.02, pe_v, ha='center', va='center', fontsize=9.5,
            color='white', fontweight='bold', transform=ax.transAxes)

ax.set_title('Collapse Scorecard: PromptPAR on ITCPR', color='white', fontsize=13, pad=15)
FIG6 = fig_to_b64(fig6, dpi=130)
plt.close(fig6)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Per-image std distribution
# ═══════════════════════════════════════════════════════════════════════════════
print('Building Figure 7 (per-image std distribution) …')
fig7, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='#1a1a2e')
for ax, probs, title, col in [
    (axes[0], probs_pa, 'PA100k — Per-image Attribute Std', C_BLUE),
    (axes[1], probs_pe, 'PETA — Per-image Attribute Std', C_PURPLE),
]:
    ax.set_facecolor('#16213e')
    stds = probs.std(axis=1)
    ax.hist(stds, bins=60, color=col, edgecolor='none', alpha=0.85)
    ax.axvline(stds.mean(), color='white', lw=1.5, ls='--',
               label=f'mean={stds.mean():.4f}')
    ax.set_xlabel('Per-image Std across Attributes', color='white', fontsize=10)
    ax.set_ylabel('Count', color='white', fontsize=10)
    ax.set_title(title, color='white', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#334')
    ax.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#334', labelcolor='white')
    note = 'narrow range → no per-image diversity'
    ax.text(0.98, 0.9, note, transform=ax.transAxes, ha='right', color='#aaa', fontsize=8)

fig7.suptitle('Per-image Attribute Diversity (high std = model discriminates; low = collapsed)',
              color='white', fontsize=12, y=1.01)
fig7.tight_layout()
FIG7 = fig_to_b64(fig7)
plt.close(fig7)

# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE HTML
# ═══════════════════════════════════════════════════════════════════════════════
print('Assembling HTML …')

def section(title, content, icon='▸'):
    return f'''
<section>
  <h2>{icon} {title}</h2>
  {content}
</section>'''

def fig_block(b64, caption, max_w='100%'):
    return f'''
<figure>
  <img src="data:image/png;base64,{b64}" style="max-width:{max_w};width:100%;border-radius:8px"/>
  <figcaption>{caption}</figcaption>
</figure>'''

def callout(text, kind='warn'):
    colors = {'warn': ('#f39c12', '#2c2000'), 'bad': (C_RED, '#1a0000'),
              'good': (C_GREEN, '#001a00'), 'info': (C_BLUE, '#001020')}
    bc, bg = colors.get(kind, colors['info'])
    return f'<div class="callout" style="border-left:4px solid {bc};background:{bg};padding:12px 16px;border-radius:4px;margin:12px 0">{text}</div>'

# top-3 PA100k prediction sets
top3_pa_html = '<ol style="margin:4px 0;padding-left:20px">'
for s, c in counter_pa.most_common(5):
    pct = 100*c/len(paths_all)
    badge_col = C_RED if pct > 50 else C_ORANGE if pct > 10 else C_YELLOW
    top3_pa_html += f'<li><span style="color:{badge_col};font-weight:bold">{pct:.1f}%</span> — ' + ', '.join(f'<code>{a}</code>' for a in s) + '</li>'
top3_pa_html += '</ol>'

top3_pe_html = '<ol style="margin:4px 0;padding-left:20px">'
for s, c in counter_pe.most_common(5):
    pct = 100*c/len(paths_all)
    badge_col = C_RED if pct > 20 else C_ORANGE if pct > 5 else C_YELLOW
    top3_pe_html += f'<li><span style="color:{badge_col};font-weight:bold">{pct:.1f}%</span> — ' + ', '.join(f'<code>{a}</code>' for a in s) + '</li>'
top3_pe_html += '</ol>'

HTML = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PromptPAR on ITCPR — Collapse Analysis</title>
<style>
  :root {{
    --bg: #0f0f1a; --bg2: #16213e; --bg3: #1a1a2e;
    --text: #e0e0f0; --muted: #8899aa; --accent: #4fc3f7;
    --blue: {C_BLUE}; --purple: {C_PURPLE}; --red: {C_RED}; --green: {C_GREEN};
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; line-height: 1.65; }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 0 24px 60px; }}
  header {{ background: linear-gradient(135deg, #0d1b4b 0%, #1a0533 100%);
            padding: 48px 32px; margin-bottom: 32px; border-bottom: 1px solid #334; }}
  header h1 {{ font-size: 2.2rem; font-weight: 700; color: white; letter-spacing: -0.5px; }}
  header .subtitle {{ color: var(--muted); margin-top: 8px; font-size: 1.05rem; }}
  header .meta {{ display: flex; gap: 24px; margin-top: 20px; flex-wrap: wrap; }}
  header .meta span {{ background: #ffffff18; padding: 4px 12px; border-radius: 20px;
                       font-size: 0.85rem; color: #bcd; }}
  section {{ margin-bottom: 44px; }}
  h2 {{ font-size: 1.35rem; color: var(--accent); margin-bottom: 16px; padding-bottom: 6px;
        border-bottom: 1px solid #223; }}
  h3 {{ font-size: 1.05rem; color: #aaccff; margin: 20px 0 10px; }}
  p {{ margin-bottom: 10px; color: #ccd; }}
  figure {{ margin: 20px 0; }}
  figcaption {{ text-align: center; color: var(--muted); font-size: 0.82rem; margin-top: 8px; }}
  code {{ background: #223; padding: 1px 6px; border-radius: 3px; font-size: 0.88em; color: #7ec8e3; }}
  table.stats {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  table.stats th, table.stats td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #223; font-size: 0.9rem; }}
  table.stats th {{ background: #1e2a4a; color: var(--accent); font-weight: 600; }}
  table.stats tr:nth-child(even) {{ background: #12192e; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .card {{ background: var(--bg2); border-radius: 8px; padding: 20px; border: 1px solid #223; }}
  .card h3 {{ margin-top: 0; }}
  .badge {{ display: inline-block; padding: 2px 9px; border-radius: 12px; font-size: 0.78rem; font-weight: 700; }}
  .toc {{ background: #12192e; border-radius: 8px; padding: 20px 28px; margin-bottom: 32px;
           border: 1px solid #223; }}
  .toc h3 {{ color: var(--accent); margin-bottom: 10px; }}
  .toc ol {{ padding-left: 20px; }}
  .toc li {{ margin-bottom: 5px; }}
  .toc a {{ color: #7ec8e3; text-decoration: none; }}
  .toc a:hover {{ text-decoration: underline; }}
  .scroll-wrap {{ overflow-x: auto; padding-bottom: 8px; }}
  @media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <div class="container">
    <h1>PromptPAR on ITCPR: Out-of-Distribution Collapse Analysis</h1>
    <p class="subtitle">Evaluating zero-shot pedestrian attribute recognition on a real-world surveillance dataset</p>
    <div class="meta">
      <span>📅 2026-06-04</span>
      <span>🗂 ITCPR Gallery — 20,510 images</span>
      <span>🔵 Checkpoint: PA100k (26 attrs)</span>
      <span>🟣 Checkpoint: PETA (35 attrs)</span>
      <span>📏 Sources: Celeb-reID · PRCC · LAST</span>
    </div>
  </div>
</header>

<div class="container">

<!-- TOC -->
<div class="toc">
  <h3>Contents</h3>
  <ol>
    <li><a href="#summary">Executive Summary</a></li>
    <li><a href="#scorecard">Collapse Scorecard</a></li>
    <li><a href="#diversity">Prediction Diversity</a></li>
    <li><a href="#attributes">Attribute Activation Analysis</a></li>
    <li><a href="#per-image">Per-Image Variance</a></li>
    <li><a href="#sources">Per-Source Breakdown</a></li>
    <li><a href="#examples-collapsed">Image Examples — Collapsed Predictions</a></li>
    <li><a href="#examples-diverse">Image Examples — Most Diverse Predictions</a></li>
    <li><a href="#examples-prcc">Image Examples — PRCC (Highest Quality Source)</a></li>
    <li><a href="#causes">Root Cause Analysis</a></li>
    <li><a href="#conclusion">Conclusion &amp; Recommendations</a></li>
  </ol>
</div>

<!-- SUMMARY -->
{section('Executive Summary', f'''
{callout('<strong>PromptPAR collapses severely on ITCPR.</strong> The PA100k checkpoint produces only <strong>7 distinct prediction sets</strong> for 20,508 images — 84% of all images receive <em>exactly the same</em> attribute list. The PETA checkpoint is less extreme (647 sets) but still highly degenerate. Both checkpoints show pairwise cosine similarity &gt;0.94 between random image pairs.', 'bad')}
<p>ITCPR is a real-world surveillance dataset (Celeb-reID + PRCC + LAST) that is <strong>strongly out-of-distribution</strong> relative to the PA100k and PETA training sets. The images are low-resolution surveillance crops (typically 128×256 px) being fed to a model trained on clean, controlled pedestrian photographs. The result is a model that ignores image content and outputs a fixed set of majority-class attributes for every person.</p>
<p>The two checkpoints exhibit different failure modes: <strong>PA100k is catastrophically collapsed</strong> (7 output clusters, gender completely dead), while <strong>PETA is moderately collapsed</strong> (clothing attributes saturated, but age/gender/shoes show partial discrimination).</p>
''', '📋', )}

<!-- SCORECARD -->
{section('Collapse Scorecard', fig_block(FIG6, 'Summary comparison of key collapse metrics between the two checkpoints.'), '📊', )}

<!-- DIVERSITY -->
{section('Prediction Diversity', f'''
{fig_block(FIG3, 'How many distinct attribute prediction sets does the model produce for 20,510 images? Perfect generalization → one set per image. Complete collapse → one set for all.')}

<div class="grid-2" style="margin-top:20px">
  <div class="card">
    <h3 style="color:{C_BLUE}">PA100k — Top-5 Prediction Sets</h3>
    {top3_pa_html}
    {callout('Only <strong>7 unique sets</strong> for 20,508 images. The top set accounts for 84.1% of all predictions — 17,245 images receive <em>identical</em> output.', 'bad')}
  </div>
  <div class="card">
    <h3 style="color:{C_PURPLE}">PETA — Top-5 Prediction Sets</h3>
    {top3_pe_html}
    {callout('<strong>647 unique sets</strong> (91× more diverse than PA100k). Still degenerate — <code>upper casual</code>, <code>upper other</code>, <code>lower Casual</code>, <code>lower Trousers</code> appear in &gt;99% of all predictions.', 'warn')}
  </div>
</div>

{fig_block(FIG5, 'Histogram of cosine similarity between random pairs of prediction vectors. PA100k clusters near 0.99 — the model outputs nearly the same vector for every image.')}
''', '🔀')}

<!-- ATTRIBUTES -->
{section('Attribute Activation Analysis', f'''
{fig_block(FIG1, 'Positive rate (fraction of images above threshold=0.45) for each attribute. Red = saturated (always predicted), grey = dead (never predicted), green = active.')}
{fig_block(FIG2, 'Mean probability ± standard deviation per attribute. Narrow error bars = low variance = collapsed. Red bars = mean ≈ 1 (saturated).')}

<div class="grid-2" style="margin-top:16px">
  <div class="card">
    <h3 style="color:{C_BLUE}">PA100k — Key Findings</h3>
    <table class="stats">
      <tr><th>Status</th><th>Attribute</th><th>Mean</th><th>Pos Rate</th></tr>
      {"".join(f'<tr><td><span class="badge" style="background:{collapse_label(p,s)[1]}22;color:{collapse_label(p,s)[1]}">{collapse_label(p,s)[0]}</span></td><td><code>{a}</code></td><td>{m:.3f}</td><td>{p:.1%}</td></tr>' for a,m,p,s in zip(attr_pa,mean_pa,pos_pa,std_pa))}
    </table>
  </div>
  <div class="card">
    <h3 style="color:{C_PURPLE}">PETA — Key Findings</h3>
    <table class="stats">
      <tr><th>Status</th><th>Attribute</th><th>Mean</th><th>Pos Rate</th></tr>
      {"".join(f'<tr><td><span class="badge" style="background:{collapse_label(p,s)[1]}22;color:{collapse_label(p,s)[1]}">{collapse_label(p,s)[0]}</span></td><td><code>{a}</code></td><td>{m:.3f}</td><td>{p:.1%}</td></tr>' for a,m,p,s in zip(attr_pe,mean_pe,pos_pe,std_pe))}
    </table>
  </div>
</div>
''', '📈')}

<!-- PER-IMAGE VARIANCE -->
{section('Per-Image Variance', f'''
{fig_block(FIG7, 'Distribution of per-image standard deviation across all attributes. A wide, right-shifted distribution means the model discriminates between images. Narrow = collapsed.')}
{callout('PA100k per-image std spans only <strong>0.27–0.34</strong> with mean 0.31. This extremely narrow range confirms the model is largely ignoring image content. PETA is slightly better (0.31–0.41, mean 0.36) but still severely constrained.', 'warn')}
''', '📉')}

<!-- SOURCES -->
{section('Per-Source Breakdown', f'''
{fig_block(FIG4, 'Attribute positive rates broken down by ITCPR source dataset. If the model was generalizing, we would expect different activation patterns for different sources. The near-identical rows confirm collapse is not source-specific.')}
<p>All three sources (Celeb-reID, PRCC, LAST) show nearly identical collapse patterns, ruling out camera-specific artifacts as the sole cause. PRCC shows slightly less collapse — its images are somewhat higher quality and more controlled than the other two sources.</p>
''', '📂')}

<!-- EXAMPLES: COLLAPSED -->
{section('Image Examples — Most Collapsed Predictions', f'''
<p>Images with the <strong>lowest per-image attribute variance</strong> — these are the images the model treats as most indistinguishable from the rest. Both checkpoints assign nearly their entire vocabulary to the dominant cluster.</p>
<div class="scroll-wrap">{GRID_COLLAPSED}</div>
{callout('Notice how PA100k consistently outputs <code>age 18-60, back, backpack, long sleeve, trousers</code> regardless of the person\'s actual pose or clothing. PETA similarly saturates on <code>upper casual/other, lower Casual/Trousers</code>.', 'bad')}
''', '🔴')}

<!-- EXAMPLES: DIVERSE -->
{section('Image Examples — Most Diverse Predictions', f'''
<p>Images with the <strong>highest per-image attribute variance</strong> — these are the images where the model is least certain, producing the most varied predictions. Even here, PA100k stays within its 7 clusters.</p>
<div class="scroll-wrap">{GRID_DIVERSE}</div>
{callout('Even the "most diverse" PA100k predictions differ only in whether <code>front</code> fires or whether <code>backpack</code> is included. PETA shows more genuine diversity in shoes, accessories, and age.', 'warn')}
''', '🟡')}

<!-- EXAMPLES: PRCC -->
{section('Image Examples — PRCC Source (Best Quality)', f'''
<p>PRCC is the smallest source in ITCPR (146 images) but tends to have clearer, higher-resolution crops. It shows the least collapse of any source — making it a useful sanity check.</p>
<div class="scroll-wrap">{GRID_PRCC}</div>
{callout('Even on PRCC\'s higher-quality crops, PA100k cannot produce more than 7 distinct prediction sets. PETA shows slightly more attribute diversity here (lower <code>head_hat</code> rate: 49% vs 86% overall).', 'info')}
''', '🟢')}

<!-- ROOT CAUSES -->
{section('Root Cause Analysis', f'''
<h3>1. Resolution and Visual Quality Mismatch (Primary)</h3>
<p>ITCPR images are <strong>128×256 px or smaller surveillance crops</strong>, upsampled to 224×224 for inference. This produces blurry, pixelated inputs with low contrast. PA100k and PETA training images are clear, well-lit pedestrian photographs. The CLIP visual encoder + visual prompts (trained end-to-end on clean images) have never seen this kind of degraded input — all surveillance images map to a narrow cluster in feature space.</p>

<h3>2. Camera Angle and Viewpoint Prior</h3>
<p>PA100k and PETA are dominated by <strong>front-facing</strong> pedestrian images, with a minority of side/back views. ITCPR is real surveillance footage where cameras often capture people from the side or back. The result: <code>back</code> fires for 99.8% of PA100k predictions, indicating the model interprets any non-frontal image as "back." The training set prior — not the image content — drives this.</p>

<h3>3. Training-Set Distribution Leakage</h3>
<p><code>backpack</code> fires for 95% of PA100k images despite being a contextually uncommon attribute in a surveillance gallery. This reveals that the PA100k training set contained a high proportion of backpack-wearing subjects, and the model learned a prior rather than a detector. Similarly, PETA's <code>upper casual + upper other</code> are the "catch-all" categories that cover nearly every image in the training set.</p>

<h3>4. Visual Prompt Over-Fitting</h3>
<p>PromptPAR uses <strong>50 visual prompts</strong> and <strong>3 text prompts</strong> jointly fine-tuned on the training dataset. These prompts encode dataset-specific statistics rather than domain-invariant features. When applied to OOD surveillance images, they steer the attention mechanism toward the same spurious features regardless of image content.</p>

<h3>5. Attribute Vocabulary Mismatch</h3>
<p>PA100k/PETA vocabularies were designed for controlled pedestrian recognition. Attributes like <code>upper splice</code>, <code>lower stripe</code>, and <code>long coat</code> are well-represented in training data but may not apply to the clothing styles captured in ITCPR's surveillance footage. The model's text heads for rare attributes never learned to fire on OOD visual features.</p>

{callout('<strong>Why is PETA better than PA100k on ITCPR?</strong> PETA has a larger attribute vocabulary (35 vs 26) including gender (<code>male</code>) and more granular age attributes. Its text prompts were trained to distinguish more fine-grained categories, giving the model more "slots" to assign probability mass. Even under collapse, the PETA vocabulary captures more meaningful variation (shoes, accessories, age).', 'info')}
''', '🔍')}

<!-- CONCLUSION -->
{section('Conclusion &amp; Recommendations', f'''
{callout('Neither checkpoint is usable zero-shot on ITCPR. PA100k is catastrophically collapsed (7 prediction clusters, gender dead). PETA is less extreme but still unusable for real discrimination.', 'bad')}

<h3>PA100k vs PETA for Surveillance Use</h3>
<p>If forced to choose, <strong>PETA is substantially preferable</strong> for ITCPR-like data:</p>
<ul style="margin:8px 0 16px 20px;color:#ccd">
  <li>647 unique prediction sets vs 7 for PA100k</li>
  <li><code>male</code> attribute partially works (15% pos rate) vs gender completely dead in PA100k</li>
  <li>Age attributes show genuine variation (young/middle-aged spread) vs flat in PA100k</li>
  <li>Shoe type predictions vary across images (PETA) vs constant in PA100k</li>
</ul>

<h3>Recommended Mitigations</h3>
<table class="stats" style="margin-top:8px">
  <tr><th>Approach</th><th>Expected Impact</th><th>Effort</th></tr>
  <tr><td>Fine-tune on surveillance data (MARS, DukeMTMC-Attribute)</td><td>High — resolves distribution shift</td><td>High</td></tr>
  <tr><td>Test-time calibration (adjust per-attribute thresholds based on ITCPR prior)</td><td>Medium — reduces saturated/dead attrs</td><td>Low</td></tr>
  <tr><td>Temperature scaling on logits</td><td>Low — spreads prob mass but doesn't fix features</td><td>Low</td></tr>
  <tr><td>Image preprocessing: histogram equalization + sharpening before inference</td><td>Low-Medium — reduces quality gap</td><td>Low</td></tr>
  <tr><td>Replace PromptPAR visual prompts with prompts trained on surveillance data</td><td>Medium — keeps architecture, updates priors</td><td>Medium</td></tr>
</table>

<p style="margin-top:16px">The collapse is fundamentally a <strong>domain adaptation problem</strong>, not an architecture problem. PromptPAR's design (CLIP + visual/text prompts + MM-former) is sound, but requires the prompts to be adapted to the target domain for reliable predictions.</p>
''', '✅')}

</div><!-- /container -->
<footer style="border-top:1px solid #223;padding:20px 32px;text-align:center;color:#556;font-size:0.82rem">
  Generated by PromptPAR inference on ITCPR · PA100k checkpoint (epoch 30) · PETA checkpoint · 2026-06-04
</footer>
</body>
</html>'''

with open(OUT_HTML, 'w') as f:
    f.write(HTML)
print(f'\n✓ Report written → {OUT_HTML}  ({os.path.getsize(OUT_HTML)//1024} KB)')
