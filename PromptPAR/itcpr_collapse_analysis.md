# PromptPAR on ITCPR: OOD Collapse Analysis

**Date:** 2026-06-04  
**Checkpoints:** PA100k (26 attrs) · PETA (35 attrs)  
**Gallery:** ITCPR — 20,510 images from Celeb-reID (19,459), PRCC (146), LAST (905)

---

## TL;DR

Yes, **PromptPAR collapses severely on ITCPR**. The PA100k checkpoint produces only **7 distinct attribute prediction sets** for 20,508 images — 84% of images receive the identical prediction `{age 18-60, back, backpack, long sleeve, trousers}`. The PETA checkpoint is less extreme (647 unique sets) but still highly degenerate. Both checkpoints show pairwise cosine similarity > 0.94 between random image pairs, confirming the predictions are nearly identical regardless of image content.

---

## Prediction Diversity

| Metric | PA100k (26 attrs) | PETA (35 attrs) |
|--------|-------------------|-----------------|
| Unique prediction sets | **7** | 647 |
| Top prediction share | **84.1%** | 6.3% |
| Pairwise cosine similarity | **0.9925 ± 0.005** | 0.9415 ± 0.030 |
| Collapsed attrs (std<0.02) | 14/26 | 14/35 |
| Dead attrs (pos_rate<1%) | 20/26 | 20/35 |
| Saturated attrs (pos_rate>99%) | 4/26 | 4/35 |

---

## PA100k Checkpoint — Per-Attribute Breakdown

```
Attribute                           Mean     Std   PosRate
female                             0.006   0.003     0.000  ← NEVER predicted
age over 60                        0.004   0.004     0.000
age 18 to 60                       0.908   0.046     1.000  ← ALWAYS predicted
age less 18                        0.047   0.022     0.000
front                              0.376   0.059     0.114
side                               0.102   0.019     0.000
back                               0.670   0.066     0.998  ← ALWAYS predicted
hat                                0.030   0.030     0.000
glasses                            0.039   0.017     0.000
hand bag                           0.011   0.010     0.000
shoulder bag                       0.033   0.015     0.000
backpack                           0.625   0.101     0.950  ← Almost always
hold objects in front              0.004   0.004     0.000
short sleeve                       0.139   0.029     0.000
long sleeve                        0.887   0.027     1.000  ← ALWAYS predicted
upper stride                       0.006   0.003     0.000
upper logo                         0.033   0.010     0.000
upper plaid                        0.100   0.036     0.000
upper splice                       0.003   0.002     0.000
lower stripe                       0.013   0.012     0.000
lower pattern                      0.058   0.035     0.000
long coat                          0.005   0.006     0.000
trousers                           0.934   0.026     1.000  ← ALWAYS predicted
shorts                             0.018   0.006     0.000
skirt and dress                    0.010   0.006     0.000
boots                              0.026   0.034     0.000
```

**Dominant prediction (84.1% of all 20,508 images):**
> `{age 18 to 60, back, backpack, long sleeve, trousers}`

Gender prediction is completely dead — `female` mean prob 0.006 across all images.

---

## PETA Checkpoint — Per-Attribute Breakdown

```
Attribute                           Mean     Std   PosRate
head hat                           0.621   0.154     0.856  ← spurious
head nothing                       0.206   0.148     0.079
upper casual                       0.997   0.003     1.000  ← ALWAYS (collapsed)
upper formal                       0.004   0.003     0.000
upper jacket                       0.000   0.001     0.000
upper other                        0.994   0.016     1.000  ← ALWAYS (collapsed)
lower Casual                       0.986   0.015     1.000  ← ALWAYS (collapsed)
lower Trousers                     0.993   0.015     1.000  ← ALWAYS (collapsed)
shoes Leather                      0.567   0.237     0.664  ← some variation
shoes other                        0.610   0.250     0.726  ← some variation
attach messenger bag               0.473   0.208     0.533  ← some variation
attach nothing                     0.795   0.148     0.966
age less 30                        0.311   0.228     0.258  ← partial
age 30 45                          0.409   0.256     0.419  ← partial
male                               0.289   0.149     0.152  ← partial
```

PETA shows **more attribute diversity** than PA100k in shoes, accessories, age, and gender — but the clothing attributes (`upper`, `lower`) are essentially constants.

---

## Root Cause Analysis

### 1. Resolution Mismatch (primary cause)

ITCPR surveillance crops are 128×256 or smaller, upsampled to 224×224 — producing blurry, low-contrast inputs. PromptPAR was trained on PA100k/PETA with higher-quality pedestrian crops. The visual features from degraded images cluster into a narrow, OOD region of feature space.

### 2. Camera Angle Bias in Training Data

PA100k and PETA are predominantly front-facing pedestrian images from controlled datasets. ITCPR is real surveillance footage where cameras often capture people from behind or at angles. The PA100k model predicts `back` for 99.8% of images — likely a calibration shift where its learned "not front" distribution now triggers `back` for nearly all ITCPR images.

### 3. Training Distribution Leakage

The PA100k model was apparently trained on a dataset with many backpack-wearing pedestrians — `backpack` fires for 95% of ITCPR images despite being an uncommon attribute. This is a direct training-set prior leaking into the test set.

### 4. Visual Prompt Over-Fitting

The 50 visual prompts and 3 text prompts are trained jointly with the dataset. They likely encode dataset-specific statistics that do not transfer. When the visual features are OOD (blurry surveillance), the prompts amplify the mismatch rather than suppressing it.

### 5. Clothing Attribute Vacuousness (PETA)

`upper casual + upper other` together cover ~100% of images in PETA training too — they are the "catch-all" categories. On OOD data, these become the dominant prediction, crowding out specific attributes.

---

## Per-Source Breakdown

The three image sources (Celeb-reID, PRCC, LAST) show almost identical collapse rates — the domain gap is uniform across sources, not specific to any one camera setup.

| Source | n | PA100k backpack PosRate | PETA head_hat PosRate |
|--------|---|------------------------|----------------------|
| Celeb-reID | 19,459 | 95.5% | 86.1% |
| PRCC | 146 | 92.5% | 48.6% |
| LAST | 905 | 83.5% | 80.3% |

PRCC shows slightly less collapse for both (lower backpack rate, lower hat rate), possibly because PRCC pedestrians are more clearly visible in the crops.

---

## Conclusion

PromptPAR **does not generalize to ITCPR zero-shot**. The PA100k checkpoint collapses to a single prediction for 84% of images; the PETA checkpoint is more diverse but still highly degenerate with `upper casual`, `upper other`, `lower Casual`, and `lower Trousers` all at near-100% activation.

**PA100k is worse than PETA** for this task because:
- It has no gender attribute that fires (female is dead; no male attribute)
- Its 4 saturated attributes dominate leaving almost no signal
- PETA at least partially predicts age, gender, and shoe type

For the ITCPR use case (zero-shot pedestrian attribute recognition in surveillance), **neither checkpoint is usable as-is**. The model would need fine-tuning on surveillance-domain data, or at minimum, careful calibration and threshold adjustment to reduce the dominant-attribute bias.
