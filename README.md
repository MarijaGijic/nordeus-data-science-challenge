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
| **Final (`03_final_submission`)** | **XGBoost + LightGBM ensemble** | **0.5851** | **±0.0118** |
| Score regression (tested) | XGBRegressor on score differential | 0.5841 | ±0.0073 |
| Stacking (tested) | XGB + LGB + Score reg, OOF stacked | 0.5833 | ±0.0071 |

The XGB + LightGBM simple average on 41 features was the best by CV accuracy. More complex approaches (per-rank features, score regression, OOF stacking) were tested thoroughly in `02_improvements_analysis.ipynb` but did not beat this baseline on accuracy.

## Key Findings

1. **Min training bonus** (weakest member's floor) — strongest predictor. 0 bonus → 45% win rate; 12+ bonus → 59% win rate (+14pp)
2. **Full attendance** (all 6 logged in today) — +9 percentage points win rate
3. **Ghost members** (>14 days inactive) — each one costs ~5% win rate
4. **Bonus uniformity** — uneven teams (high spread) consistently underperform
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
- Augmentation applied only to training folds — validation folds are never augmented

## BONUS: Smart Advisory Chatbot

`chatbot.py` is a standalone advisory system that tells clan leaders exactly which players to focus on and why, grounded in win-rate data from 24,000+ real matches.

### How it works
1. Loads a clan's real player stats from the dataset
2. Compares each metric against EDA-derived benchmarks (e.g. `min_bonus ≥ 8` → 59% win rate)
3. Identifies specific players by name who are ghost members or have low training bonus
4. Produces a prioritised action plan
5. With Claude API: answers free-text questions conversationally using tool calls

### Usage
```bash
# Rule-based advisor (no API key needed)
python chatbot.py --no-api

# Full Claude API chatbot
set ANTHROPIC_API_KEY=sk-ant-...
python chatbot.py

# Pre-load a specific clan
python chatbot.py --clan clan_5029188
```

### Example output

```
==============================================================
  Advisory Report — clan_10184   [STRUGGLING CLAN]
==============================================================

[!!] GHOST MEMBERS (6 — inactive >14 days):
     user_14171            absent=57d  stars=3.1  bonus=0.0  multiplier=x2
     user_12224            absent=48d  stars=3.9  bonus=0.0  multiplier=x2
     user_41958            absent=47d  stars=4.2  bonus=0.0  multiplier=x2
     user_42052            absent=42d  stars=3.6  bonus=0.0  multiplier=x2
     user_52656            absent=26d  stars=4.8  bonus=0.0  multiplier=x2
     user_36188            absent=18d  stars=4.1  bonus=0.0  multiplier=x2
     → Each ghost costs ~6-18 pts. Contact or replace ASAP.

[!]  LOW TRAINING BONUS (6 member(s) below 5):
     → Training bonus is the #1 win predictor. Clans with floor ≥8 win 59% vs 45%.

[PRIORITY 1 — Fix These First]:
  • Min training bonus:       critical: 0.00 vs minimum 8.00
  • Ghost members (>14d):     critical: 6.00 vs threshold 1.00
  • Full attendance today:    critical: 0.00 vs minimum 0.50
  • Avg training sessions:    critical: 1.50 vs minimum 130.00
  • Training efficiency:      critical: 0.31 vs minimum 5.00

==============================================================
  Advisory Report — clan_1001   [STRONG CLAN]
==============================================================

[PRIORITY 2 — Room for Improvement]:
  • Total clan multiplier: 15.00 → target 18.00

[STRENGTHS — Keep These Up]:
  ✓ Min training bonus      ✓ Ghost members (0)
  ✓ Full attendance today   ✓ Avg training sessions
  ✓ Bonus uniformity        ✓ Squad quality
  ✓ Training efficiency     ✓ Weakest-link compound score

==============================================================
  Advisory Report — clan_10203  [TYPICAL CLAN]
==============================================================

[!!] GHOST MEMBERS (1 — inactive >14 days):
     user_12245   absent=31d  stars=6.4  bonus=0.0  multiplier=x2
     → Each ghost costs ~6-18 pts. Contact or replace ASAP.

[PRIORITY 1 — Fix These First]:
  • Min training bonus:    critical: 0.00 vs minimum 8.00
  • Full attendance today: critical: 0.00 vs minimum 0.50
  • Avg training sessions: critical: 44.00 vs minimum 130.00
  • Training efficiency:   critical: 1.76 vs minimum 5.00
```

### Example Q&A conversation

```
Q: What should clan_10184 focus on to improve their win chances?
A: The biggest drag is user_14171 — absent 57 days (ghost). With multiplier x2,
   they cost up to 12 potential points per tournament. Re-engage or replace them
   before the next match. user_12224 has the lowest training bonus (0.0). Data
   shows clans with bonus floor >= 8 win 59% of matches vs 45% for clans with
   floor near 0.

Q: How can clan_10203 improve their win rate before the tournament?
A: Top improvements by expected impact:
   1. eliminate 1 ghost member(s) (+5% win rate)
   2. raise min bonus floor from 0.0 to 8+ (+10-14% win rate)
   3. ensure all 6 members log in on tournament day (+9% win rate)

Q: Is clan_1001 in a good position for the next tournament?
A: This clan is in excellent shape — 0 ghost members, full attendance, high
   training bonus floor. Only marginal improvement possible: raising total
   clan multiplier from 15 to 18 via better League performance next season.
```

## Repository Structure

```
├── data/
│   ├── member_stats_training.csv    (291,456 rows × 15 cols)
│   ├── member_stats_test.csv        (283,752 rows × 15 cols)
│   ├── clan_matches_training.csv    (24,288 rows × 5 cols)
│   └── clan_matches_test.csv        (23,646 rows × 2 cols)
├── notebooks/
│   ├── 01_baseline_prototyping.ipynb   # EDA + XGBoost/LightGBM baseline (CV: 0.5851 ±0.0136)
│   ├── 02_improvements_analysis.ipynb  # Fold investigation, per-rank features, all experiments
│   └── 03_final_submission.ipynb       # Clean final model + advisory chatbot demo
├── chatbot.py                          # Standalone advisory chatbot (Claude API + rule-based)
├── clan_winner_predictions.csv         # Final predictions for all 23,646 test matches
└── README.md
```

> **Notebook paths:** all notebooks use `DATA = '../data/'` by default.  
> For Google Colab, switch to the commented Colab path at the top of each notebook.

## Requirements

```bash
pip install pandas numpy matplotlib seaborn xgboost lightgbm scikit-learn scipy anthropic
```

Python 3.9+. Notebooks developed and tested on Google Colab.
