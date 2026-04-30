# Nordeus Job Fair 2026 — Clan Tournament Winner Prediction

**Author:** Marija Gijic

## Problem

Predict which association (clan) wins a weekend tournament match in Top Eleven.  
Binary classification: `clan_winner` = 1 (clan_1 wins) or 2 (clan_2 wins).  
Evaluation metric: **Accuracy**.

## Key Game Mechanic

Each clan has 6 members who each play 2 matches (12 total per clan).  
Points per match: Win=3, Draw=1, Loss=0 — **multiplied by the player's individual `clan_multiplier`**.  
Matches are rank-sorted: best player vs best player, 2nd vs 2nd, etc.  
One inactive member contributes 0 points regardless of their quality — the **weakest-link effect**.

## Results

| Version | Model | CV Accuracy | Std |
|---|---|---|---|
| Baseline | XGBoost + LightGBM ensemble | 0.5851 | ±0.0136 |
| **Final** | **LightGBM + XGBRegressor blend** | **0.5836** | **±0.0074** |

The final model has **half the variance** of the baseline and is significantly more robust on hard/balanced matches (fold 2: 0.5624 → 0.5731).

## Key Findings

1. **Min training bonus** (weakest member's floor) — strongest predictor. 0 bonus → 45% win rate; 12+ bonus → 59% win rate
2. **Full attendance** (all 6 logged in today) — +9 percentage points win rate
3. **Ghost members** (>14 days inactive) — each one costs ~5% win rate
4. **Bonus uniformity** — uneven teams (high std) consistently underperform
5. **Training efficiency** — winners train 19% more sessions per active day
6. **Multiplier carry** (relying on one high-multiplier player) — **does not work**, confirmed by feature importance

> Core insight: **floor beats ceiling**. An active team with average quality beats an inactive team with star players.

## Modeling Approach

### Feature Engineering (99 match features)
- **Base aggregations** (41): mean/min/max/std of activity, bonus, quality, multiplier per clan
- **Per-rank matchup features** (48): sort each clan's 6 members by quality rank, expose position-specific stats — directly models the rank-sorted pairing mechanic
- **Ratio features** (8): clan_1/clan_2 for key metrics (captures relative advantage)
- **Derived** (2): `n_rank_advantages` (how many of 6 positions clan_1 wins), `sum_rank_pts_edge`
- All features are differenced (clan_1 − clan_2) so the model learns **relative advantage**

### Models
- **LightGBM classifier**: trained on binary win/loss labels
- **XGBRegressor on score differential**: trained on `clan_1_points − clan_2_points`. A 60–0 blowout gives stronger signal than a 10–9 narrow win — signal that binary labels discard
- **Blend**: simple average of the two probability estimates (lowest variance, best on hard matches)
- **Symmetry augmentation**: each match is doubled with clan_1 ↔ clan_2 swapped, enforcing antisymmetry

### Cross-Validation
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Augmentation applied only to training folds — validation folds are never seen during training

## BONUS: Smart Advisory Chatbot

`chatbot.py` is a standalone advisory system that tells clan leaders exactly which players to focus on and why.

**Features:**
- Diagnoses each player by name (ghost members, low training bonus, inactivity)
- Prioritises recommendations by expected impact on win rate
- Answers free-text questions via Claude API (tool use: `get_clan_report`, `compare_clans`)
- Falls back to rule-based system without any API key

**Usage:**
```bash
# Rule-based (no API key needed)
python chatbot.py --no-api

# Full Claude API chatbot
set ANTHROPIC_API_KEY=sk-ant-...
python chatbot.py

# Pre-load a specific clan
python chatbot.py --clan clan_5029188
```

**Example questions:**
- `What should clan_5029188 focus on before their next tournament?`
- `Compare clan_5029188 vs clan_5045813 — who is likely to win?`
- `Which member of clan_5029188 needs the most improvement?`

## Repository Structure

```
├── notebooks/
│   ├── 01_baseline_prototyping.ipynb   # Initial EDA and XGBoost baseline
│   ├── 02_improvements_analysis.ipynb  # Fold investigation, per-rank features, ensembles
│   └── 03_final_submission.ipynb       # Clean final model + advisory chatbot demo
├── chatbot.py                          # Standalone advisory chatbot (Claude API + rule-based)
├── clan_winner_predictions.csv         # Final predictions (23,646 rows)
└── README.md
```

> **Data files** are not included in the repository (provided by Nordeus Job Fair 2026).  
> Place them in a `data/` folder: `member_stats_training.csv`, `member_stats_test.csv`, `clan_matches_training.csv`, `clan_matches_test.csv`.  
> Update the `DATA` path variable in each notebook accordingly.

## Requirements

```bash
pip install pandas numpy matplotlib seaborn xgboost lightgbm scikit-learn scipy anthropic
```

Python 3.9+. Notebooks were developed on Google Colab.
