"""
Nordeus Clan Tournament Advisory Chatbot
=========================================
Usage:
    python chatbot.py                         # interactive mode (asks for clan ID)
    python chatbot.py --clan clan_5029188     # start with a specific clan pre-loaded
    python chatbot.py --no-api               # rule-based only (no ANTHROPIC_API_KEY needed)

Requirements:
    pip install anthropic pandas numpy lightgbm xgboost scikit-learn

Set your API key:
    export ANTHROPIC_API_KEY=sk-ant-...       # Linux/macOS
    set ANTHROPIC_API_KEY=sk-ant-...          # Windows CMD
"""

import os
import sys
import json
import argparse
import textwrap

import pandas as pd
import numpy as np

# ── Data Loading ──────────────────────────────────────────────────────────────

DATA_DIR = "data/"

def load_data():
    members_train = pd.read_csv(DATA_DIR + "member_stats_training.csv")
    members_test  = pd.read_csv(DATA_DIR + "member_stats_test.csv")
    matches_train = pd.read_csv(DATA_DIR + "clan_matches_training.csv")
    all_members   = pd.concat([members_train, members_test], ignore_index=True)
    # tag training clan appearances as winner/loser for context
    results = {}
    for _, row in matches_train.iterrows():
        results[row['clan_1_id']] = 'winner' if row['clan_winner'] == 1 else 'loser'
        results[row['clan_2_id']] = 'winner' if row['clan_winner'] == 2 else 'loser'
    all_members['result'] = all_members['clan_id'].map(results)
    return all_members, matches_train


def build_clan_features(members_df: pd.DataFrame) -> pd.DataFrame:
    df = members_df.copy()
    df["weighted_quality"]    = df["clan_multiplier"] * df["avg_stars_top_11_players"]
    df["expected_score"]      = df["clan_multiplier"] * 3
    df["recency_ratio"]       = df["days_active_last_7_days"] / (df["days_active_last_28_days"] / 4 + 0.01)
    df["training_efficiency"] = df["training_count_last_28_days"] / (df["days_active_last_28_days"] + 0.01)
    df["is_inactive"]         = (df["days_since_last_active"] > 7).astype(int)
    df["is_ghost"]            = (df["days_since_last_active"] > 14).astype(int)

    g = df.groupby("clan_id")
    return pd.DataFrame({
        "mean_days_active_28":     g["days_active_last_28_days"].mean(),
        "min_days_active_28":      g["days_active_last_28_days"].min(),
        "mean_days_active_7":      g["days_active_last_7_days"].mean(),
        "min_days_active_7":       g["days_active_last_7_days"].min(),
        "max_days_since_active":   g["days_since_last_active"].max(),
        "mean_days_since_active":  g["days_since_last_active"].mean(),
        "inactive_count":          g["is_inactive"].sum(),
        "ghost_count":             g["is_ghost"].sum(),
        "full_attendance":         g["days_since_last_active"].max().eq(0).astype(int),
        "all_active_7":            g["days_active_last_7_days"].min().gt(0).astype(int),
        "mean_recency_ratio":      g["recency_ratio"].mean(),
        "min_recency_ratio":       g["recency_ratio"].min(),
        "mean_training_count":     g["training_count_last_28_days"].mean(),
        "min_training_count":      g["training_count_last_28_days"].min(),
        "training_efficiency":     g["training_efficiency"].mean(),
        "min_training_efficiency": g["training_efficiency"].min(),
        "mean_training_bonus":     g["avg_training_bonus"].mean(),
        "min_training_bonus":      g["avg_training_bonus"].min(),
        "max_training_bonus":      g["avg_training_bonus"].max(),
        "std_training_bonus":      g["avg_training_bonus"].std(),
        "bonus_cv":                g["avg_training_bonus"].std() / (g["avg_training_bonus"].mean() + 0.01),
        "mean_stars_top11":        g["avg_stars_top_11_players"].mean(),
        "min_stars_top11":         g["avg_stars_top_11_players"].min(),
        "mean_stars_top3":         g["avg_stars_top_3_players"].mean(),
        "sum_multiplier":          g["clan_multiplier"].sum(),
        "mean_multiplier":         g["clan_multiplier"].mean(),
        "sum_weighted_quality":    g["weighted_quality"].sum(),
        "sum_expected_score":      g["expected_score"].sum(),
        "payer_ratio":             g["is_payer_lifetime"].apply(lambda x: (x == True).mean()),
        "whale_count":             g["dynamic_payment_segment"].apply(lambda x: (x == "4) Whale").sum()),
        "mean_cohort_day":         g["cohort_day"].mean(),
        "min_bonus_x_min_activity": (g["avg_training_bonus"].min() * g["days_active_last_7_days"].min()),
    })


# ── Rule-Based Advisor ─────────────────────────────────────────────────────────

BENCHMARKS = {
    "min_training_bonus":       {"good": 8.0,  "great": 12.0,  "direction": "+", "label": "Min training bonus (weakest member)"},
    "ghost_count":              {"good": 1,    "great": 0,     "direction": "-", "label": "Ghost members (>14d inactive)"},
    "inactive_count":           {"good": 1,    "great": 0,     "direction": "-", "label": "Inactive members (>7d)"},
    "full_attendance":          {"good": 0.5,  "great": 1.0,   "direction": "+", "label": "Full attendance today (0/1)"},
    "mean_training_count":      {"good": 130,  "great": 150,   "direction": "+", "label": "Avg training sessions (28d)"},
    "bonus_cv":                 {"good": 0.5,  "great": 0.2,   "direction": "-", "label": "Bonus spread (lower = more uniform team)"},
    "mean_stars_top11":         {"good": 5.5,  "great": 6.5,   "direction": "+", "label": "Avg squad quality (stars)"},
    "training_efficiency":      {"good": 5.0,  "great": 7.0,   "direction": "+", "label": "Training efficiency (sessions/active day)"},
    "sum_multiplier":           {"good": 14,   "great": 18,    "direction": "+", "label": "Total clan multiplier"},
    "min_bonus_x_min_activity": {"good": 3.0,  "great": 7.0,   "direction": "+", "label": "Weakest-link compound score (bonus × activity)"},
}


def score_gap(value, bench: dict):
    good, great, direction = bench["good"], bench["great"], bench["direction"]
    if direction == "+":
        if value >= great: return "great", None
        if value >= good:  return "ok",    f"improve from {value:.2f} → target {great:.2f}"
        return "poor",                     f"critical gap: {value:.2f} vs minimum {good:.2f}"
    else:
        if value <= great: return "great", None
        if value <= good:  return "ok",    f"reduce from {value:.2f} → target {great:.2f}"
        return "poor",                     f"critical issue: {value:.2f} vs threshold {good:.2f}"


def build_clan_report(clan_id: str, members_df: pd.DataFrame, clan_agg: pd.DataFrame) -> dict:
    """Return a structured report dict (used by both rule-based advisor and Claude)."""
    players = members_df[members_df["clan_id"] == clan_id].copy()
    if players.empty:
        return {}

    if clan_id not in clan_agg.index:
        return {}

    feats = clan_agg.loc[clan_id].to_dict()

    ghosts   = players[players["days_since_last_active"] > 14].to_dict("records")
    inactive = players[(players["days_since_last_active"] > 7) &
                       (players["days_since_last_active"] <= 14)].to_dict("records")
    low_bonus = players[players["avg_training_bonus"] < 5].to_dict("records")

    issues, strengths = [], []
    for key, bench in BENCHMARKS.items():
        val = feats.get(key)
        if val is None:
            continue
        rating, gap = score_gap(val, bench)
        if rating == "poor":
            issues.append({"priority": "CRITICAL", "label": bench["label"], "gap": gap, "value": round(float(val), 2)})
        elif rating == "ok":
            issues.append({"priority": "OK", "label": bench["label"], "gap": gap, "value": round(float(val), 2)})
        else:
            strengths.append(bench["label"])

    return {
        "clan_id": clan_id,
        "member_count": len(players),
        "ghost_members": ghosts,
        "at_risk_members": inactive,
        "low_bonus_members": low_bonus,
        "issues": issues,
        "strengths": strengths,
        "features": {k: round(float(v), 3) for k, v in feats.items()},
    }


def format_report(report: dict) -> str:
    if not report:
        return "Clan not found in dataset."

    cid = report["clan_id"]
    lines = [f"\n{'='*60}", f"  Advisory Report — {cid}", f"{'='*60}"]

    # Ghost / inactive members
    if report["ghost_members"]:
        lines.append(f"\n[!!] GHOST MEMBERS ({len(report['ghost_members'])} — inactive >14 days):")
        for p in report["ghost_members"]:
            lines.append(f"     • {p['user_id']:20s}  absent={int(p['days_since_last_active'])}d  "
                         f"stars={p['avg_stars_top_11_players']:.1f}  bonus={p['avg_training_bonus']:.1f}")
        lines.append("     → Re-engage or replace these members. Each ghost costs ~6-18 pts.")

    if report["at_risk_members"]:
        lines.append(f"\n[!]  AT-RISK MEMBERS ({len(report['at_risk_members'])} — inactive 7-14 days):")
        for p in report["at_risk_members"]:
            lines.append(f"     • {p['user_id']:20s}  absent={int(p['days_since_last_active'])}d")
        lines.append("     → Contact before the tournament starts.")

    # Bonus floor
    if report["low_bonus_members"]:
        lines.append(f"\n[!]  LOW TRAINING BONUS ({len(report['low_bonus_members'])} member(s) below 5):")
        for p in report["low_bonus_members"]:
            lines.append(f"     • {p['user_id']:20s}  bonus={p['avg_training_bonus']:.1f}")
        lines.append("     → Training bonus is the strongest win predictor. Boost these players first.")

    # Prioritised issues
    critical = [i for i in report["issues"] if i["priority"] == "CRITICAL"]
    ok_issues = [i for i in report["issues"] if i["priority"] == "OK"]

    if critical:
        lines.append("\n[PRIORITY 1 — Fix These First]:")
        for i in critical:
            lines.append(f"  • {i['label']}: {i['gap']}")

    if ok_issues:
        lines.append("\n[PRIORITY 2 — Room for Improvement]:")
        for i in ok_issues:
            lines.append(f"  • {i['label']}: {i['gap']}")

    if report["strengths"]:
        lines.append("\n[STRENGTHS — Keep These Up]:")
        for s in report["strengths"]:
            lines.append(f"  ✓ {s}")

    return "\n".join(lines)


# ── Claude API Chatbot ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a data-driven Top Eleven clan tournament advisor. Your job is to help Association managers
boost their win chances using evidence from historical tournament data.

Key game mechanics:
- Each clan has 6 members who each play 2 matches (12 total matches per clan)
- Match points: Win=3, Draw=1, Loss=0 — MULTIPLIED by the player's individual clan_multiplier
- One inactive or weak member can cost the entire team 6-18 points per tournament

Evidence-based win factors (ranked by impact from our model):
1. Min training bonus (weakest member's floor) — strongest single predictor (#1)
   Clans with 0 bonus floor: 45% win rate → Clans with 12+ floor: 59% win rate
2. Full attendance (all 6 logged in today) — +9 percentage points win rate
3. Ghost members (>14 days inactive) — each ghost lowers win rate by ~5%
4. Bonus uniformity (low spread across team) — even teams outperform uneven ones
5. Training efficiency (sessions per active day) — winners train 19% more
6. Squad quality (stars) — matters, but less than engagement
7. Multiplier carry (relying on 1 high-multiplier player) — does NOT work

When given a clan report:
- Identify the most impactful 2-3 improvements for the SPECIFIC players who need attention
- Quantify impact where possible (e.g. "this could add +X% win rate")
- Be direct and actionable — avoid generic advice
- If asking about strategy, explain the multiplier mechanic and why floor beats ceiling

Respond concisely. Use bullet points. Stay focused on the data.
"""

TOOLS = [
    {
        "name": "get_clan_report",
        "description": "Fetch the full stats report for a clan by its ID. Returns member activity, bonus levels, ghost members, and key metrics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clan_id": {
                    "type": "string",
                    "description": "The clan ID to look up, e.g. 'clan_5029188'"
                }
            },
            "required": ["clan_id"]
        }
    },
    {
        "name": "compare_clans",
        "description": "Compare two clans head-to-head across all key metrics and predict the likely winner.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clan_1_id": {"type": "string", "description": "First clan ID"},
                "clan_2_id": {"type": "string", "description": "Second clan ID"}
            },
            "required": ["clan_1_id", "clan_2_id"]
        }
    },
    {
        "name": "list_weak_members",
        "description": "List members of a clan who most need improvement, sorted by their compound impact score.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clan_id": {"type": "string", "description": "The clan ID to analyse"}
            },
            "required": ["clan_id"]
        }
    }
]


class ClanAdvisorChatbot:
    def __init__(self, members_df: pd.DataFrame, clan_agg: pd.DataFrame, use_api: bool = True):
        self.members_df = members_df
        self.clan_agg   = clan_agg
        self.history    = []
        self.use_api    = use_api
        self.client     = None

        if use_api:
            try:
                import anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                if not api_key:
                    print("[WARNING] ANTHROPIC_API_KEY not set. Falling back to rule-based advisor.")
                    self.use_api = False
                else:
                    self.client = anthropic.Anthropic(api_key=api_key)
                    print("[OK] Claude API connected.")
            except ImportError:
                print("[WARNING] anthropic package not installed. Run: pip install anthropic")
                self.use_api = False

    # ── Tool execution ────────────────────────────────────────────────────────

    def _exec_tool(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "get_clan_report":
            report = build_clan_report(tool_input["clan_id"], self.members_df, self.clan_agg)
            if not report:
                return json.dumps({"error": f"Clan '{tool_input['clan_id']}' not found."})
            report_copy = {k: v for k, v in report.items() if k != "features"}
            report_copy["key_metrics"] = {
                "min_training_bonus": round(float(self.clan_agg.loc[tool_input["clan_id"], "min_training_bonus"]), 2),
                "ghost_count":        int(self.clan_agg.loc[tool_input["clan_id"], "ghost_count"]),
                "full_attendance":    int(self.clan_agg.loc[tool_input["clan_id"], "full_attendance"]),
                "mean_training_count": round(float(self.clan_agg.loc[tool_input["clan_id"], "mean_training_count"]), 1),
                "mean_stars_top11":   round(float(self.clan_agg.loc[tool_input["clan_id"], "mean_stars_top11"]), 2),
                "sum_multiplier":     round(float(self.clan_agg.loc[tool_input["clan_id"], "sum_multiplier"]), 1),
                "bonus_cv":           round(float(self.clan_agg.loc[tool_input["clan_id"], "bonus_cv"]), 3),
            }
            return json.dumps(report_copy, default=str)

        elif tool_name == "compare_clans":
            c1, c2 = tool_input["clan_1_id"], tool_input["clan_2_id"]
            if c1 not in self.clan_agg.index or c2 not in self.clan_agg.index:
                return json.dumps({"error": "One or both clans not found."})
            comparison = {}
            key_metrics = ["min_training_bonus", "ghost_count", "full_attendance",
                           "mean_training_count", "mean_stars_top11", "sum_multiplier",
                           "bonus_cv", "inactive_count", "training_efficiency"]
            for m in key_metrics:
                v1 = float(self.clan_agg.loc[c1, m])
                v2 = float(self.clan_agg.loc[c2, m])
                comparison[m] = {c1: round(v1, 2), c2: round(v2, 2),
                                 "advantage": c1 if v1 > v2 else c2 if v2 > v1 else "tie"}
            return json.dumps({"comparison": comparison}, default=str)

        elif tool_name == "list_weak_members":
            clan_id = tool_input["clan_id"]
            players = self.members_df[self.members_df["clan_id"] == clan_id].copy()
            if players.empty:
                return json.dumps({"error": "Clan not found."})
            players["impact_score"] = (
                players["avg_training_bonus"].rank(pct=True) * 0.4 +
                (1 - players["days_since_last_active"].rank(pct=True)) * 0.35 +
                players["training_count_last_28_days"].rank(pct=True) * 0.25
            )
            players_sorted = players.sort_values("impact_score").head(6)
            result = []
            for _, p in players_sorted.iterrows():
                result.append({
                    "user_id": p["user_id"],
                    "days_since_last_active": int(p["days_since_last_active"]),
                    "avg_training_bonus": round(p["avg_training_bonus"], 1),
                    "training_count_last_28d": int(p["training_count_last_28_days"]),
                    "avg_stars_top11": round(p["avg_stars_top_11_players"], 2),
                    "clan_multiplier": p["clan_multiplier"],
                })
            return json.dumps({"members_by_priority": result})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # ── Claude API loop ───────────────────────────────────────────────────────

    def _ask_claude(self, user_message: str) -> str:
        import anthropic
        self.history.append({"role": "user", "content": user_message})

        while True:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self.history
            )

            # Collect text and tool_use blocks
            tool_calls = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if not tool_calls:
                # No tool use — return final text
                reply = text_blocks[0].text if text_blocks else ""
                self.history.append({"role": "assistant", "content": response.content})
                return reply

            # Append assistant message with tool uses
            self.history.append({"role": "assistant", "content": response.content})

            # Execute all tools and append results
            tool_results = []
            for tc in tool_calls:
                result_str = self._exec_tool(tc.name, tc.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_str
                })

            self.history.append({"role": "user", "content": tool_results})

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _ask_rulebased(self, user_message: str) -> str:
        msg = user_message.lower()
        for token in msg.split():
            if token.startswith("clan_"):
                report = build_clan_report(token, self.members_df, self.clan_agg)
                if report:
                    return format_report(report)
        return (
            "Rule-based advisor: Please mention a clan ID (e.g. 'clan_5029188') in your question.\n"
            "Example: 'Advise clan_5029188 on how to win.'"
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def chat(self, message: str) -> str:
        if self.use_api and self.client:
            return self._ask_claude(message)
        return self._ask_rulebased(message)

    def reset(self):
        self.history = []


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nordeus Clan Advisory Chatbot")
    parser.add_argument("--clan", default=None, help="Start with a specific clan pre-loaded")
    parser.add_argument("--no-api", action="store_true", help="Use rule-based advisor only")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Nordeus Clan Tournament Advisory Chatbot")
    print("="*60)
    print("Loading data...")

    members_df, matches_df = load_data()
    clan_agg = build_clan_features(members_df)

    bot = ClanAdvisorChatbot(members_df, clan_agg, use_api=not args.no_api)

    print(f"\nLoaded {len(clan_agg)} clans.")
    print("\nYou can ask things like:")
    print("  • 'Analyse clan_5029188'")
    print("  • 'Compare clan_5029188 vs clan_5016270'")
    print("  • 'Which members of clan_5029188 need the most improvement?'")
    print("  • 'What should a clan focus on to win?'")
    print("  • 'reset' — clear conversation history")
    print("  • 'quit' / 'exit' — exit\n")

    if args.clan:
        print(f"Pre-loading report for {args.clan}...")
        report = build_clan_report(args.clan, members_df, clan_agg)
        if report:
            print(format_report(report))
            if bot.use_api:
                # Prime the conversation context with the report
                bot.history.append({"role": "user", "content": f"Here is the data for {args.clan}:\n{json.dumps({k: v for k, v in report.items() if k != 'features'}, default=str)}"})
                bot.history.append({"role": "assistant", "content": f"I've loaded the data for {args.clan}. What would you like to know?"})
                print(f"\nAssistant: I've loaded the data for {args.clan}. What would you like to know?\n")
        else:
            print(f"Clan {args.clan} not found.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            bot.reset()
            print("Conversation reset.\n")
            continue

        reply = bot.chat(user_input)
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()
