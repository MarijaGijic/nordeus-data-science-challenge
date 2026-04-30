"""
Chatbot demo — runs rule-based advisor on 3 clan profiles and
prints a sample Q&A session. No ML packages required.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from chatbot import (
    load_data, build_clan_features,
    build_clan_report, format_report
)

print("Loading data...")
members_df, matches_df = load_data()
clan_agg = build_clan_features(members_df)
print(f"Loaded {len(clan_agg):,} clans.\n")

# --- Pick 3 representative clans ---
import pandas as pd, numpy as np

# Struggling: most ghost members
ghost_counts = members_df.groupby('clan_id')['days_since_last_active'].apply(lambda x: (x > 14).sum())
clan_struggling = ghost_counts[ghost_counts >= 3].sort_values(ascending=False).index[0]

# Strong: no ghosts + high min bonus + full attendance
strong = clan_agg[
    (clan_agg['ghost_count'] == 0) &
    (clan_agg['min_training_bonus'] > 12) &
    (clan_agg['full_attendance'] == 1)
]
clan_strong = strong.index[0] if len(strong) > 0 else clan_agg.index[0]

# Typical: find clan closest to mean on ghost_count + mean_training_bonus
mean_ghost = clan_agg['ghost_count'].mean()
mean_bonus = clan_agg['mean_training_bonus'].mean()
dist = ((clan_agg['ghost_count'] - mean_ghost).abs() +
        (clan_agg['mean_training_bonus'] - mean_bonus).abs())
clan_typical = dist.idxmin()


SEP = "=" * 62

# ── Demo 1: Advisory reports ─────────────────────────────────────
print(SEP)
print("  DEMO 1 — Advisory Reports for 3 Clan Profiles")
print(SEP)

for label, cid in [
    ("STRUGGLING CLAN (ghost-heavy)", clan_struggling),
    ("STRONG CLAN (elite profile)",   clan_strong),
    ("TYPICAL CLAN (average profile)", clan_typical),
]:
    print(f"\n>>> {label}")
    report = build_clan_report(cid, members_df, clan_agg)
    print(format_report(report))

# ── Demo 2: Simulated Q&A conversation ───────────────────────────
print(f"\n\n{SEP}")
print("  DEMO 2 — Simulated Conversation (rule-based)")
print(SEP)

def simple_qa(question, clan_id):
    report = build_clan_report(clan_id, members_df, clan_agg)
    feats  = report.get('features', {})
    players = members_df[members_df['clan_id'] == clan_id]

    q = question.lower()

    if 'worst' in q or 'weakest' in q or 'focus' in q or 'first' in q:
        ghosts = players[players['days_since_last_active'] > 14]
        low_b  = players[players['avg_training_bonus'] < 5].sort_values('avg_training_bonus')
        parts  = []
        if not ghosts.empty:
            p = ghosts.sort_values('days_since_last_active', ascending=False).iloc[0]
            parts.append(
                f"The biggest drag is {p['user_id']} — absent {int(p['days_since_last_active'])} days "
                f"(ghost). With multiplier x{int(p['clan_multiplier'])}, they cost up to "
                f"{int(p['clan_multiplier']) * 6} potential points per tournament. "
                f"Re-engage or replace them before the next match."
            )
        if not low_b.empty:
            p = low_b.iloc[0]
            parts.append(
                f"{p['user_id']} has the lowest training bonus ({p['avg_training_bonus']:.1f}). "
                f"Data shows clans with bonus floor ≥ 8 win 59% of matches vs 45% for clans with floor near 0."
            )
        return ' '.join(parts) if parts else "All members look healthy — focus on keeping everyone active this week."

    if 'improve' in q or 'win rate' in q or 'chance' in q:
        gc = int(feats.get('ghost_count', 0))
        mb = float(feats.get('min_training_bonus', 0))
        fa = int(feats.get('full_attendance', 0))
        tips = []
        if gc > 0:
            tips.append(f"eliminate {gc} ghost member(s) (+{gc*5}% win rate)")
        if mb < 8:
            tips.append(f"raise min bonus floor from {mb:.1f} to 8+ (+10–14% win rate)")
        if not fa:
            tips.append("ensure all 6 members log in on tournament day (+9% win rate)")
        if tips:
            return "Top improvements by expected impact:\n" + "\n".join(f"  {i+1}. {t}" for i, t in enumerate(tips))
        return "This clan is already in strong shape. Maintain attendance and keep training sessions consistent."

    if 'compare' in q or 'vs' in q:
        return "Use: ask_advisor('compare clan_X vs clan_Y') to get a head-to-head breakdown."

    return format_report(build_clan_report(clan_id, members_df, clan_agg))


questions = [
    (f"What should {clan_struggling} focus on to improve their win chances?",
     clan_struggling),
    (f"Which member of {clan_struggling} should they fix first?",
     clan_struggling),
    (f"How can {clan_typical} improve their win rate before the tournament?",
     clan_typical),
    (f"Is {clan_strong} in a good position for the next tournament?",
     clan_strong),
]

for q, cid in questions:
    print(f"\nQ: {q}")
    print(f"A: {simple_qa(q, cid)}")

print(f"\n\n{SEP}")
print("  Full interactive chatbot: python chatbot.py")
print(f"  With Claude API:         set ANTHROPIC_API_KEY=sk-ant-...")
print(SEP)
