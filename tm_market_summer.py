#!/usr/bin/env python3
"""
Transfermarkt → CSV (Premier League transfers, no lxml)

Creates a CSV with columns:
Team, Player, Direction, Age, Position, Market Value,
Left/Joined, Club that left/joined, Fee

Example:
  python tm_transfers_to_csv_no_lxml.py --season 2025 --window s \
      --out premier_league_transfers_2025_summer.csv
"""

import argparse
import re
import sys
from typing import List, Dict

import requests
import pandas as pd
from bs4 import BeautifulSoup, FeatureNotFound

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.transfermarkt.us/",
}


def clean_text(html_node) -> str:
    """
    Visible text with whitespace collapsed.
    Also turns line breaks (e.g., 'End of loan\\nJun 30, 2025') into single spaces.
    """
    if html_node is None:
        return ""
    s = html_node.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", s).strip()


def player_from_cell(td) -> str:
    a = td.find("a")
    if a and a.get("title"):
        return a["title"].strip()
    return clean_text(td)


def club_from_cell(td) -> str:
    """
    Prefer anchor title (e.g., 'Real Sociedad'). If missing, fall back to text.
    """
    a = td.find("a")
    if a and a.get("title"):
        return a["title"].strip()
    return clean_text(td)


def parse_direction_table(table, team: str, direction: str) -> List[Dict]:
    """
    Parse one table (either the IN table or the OUT table) for a single team box.
    The column immediately after 'Market value' is a 2-col group (badge + text):
        - for IN table the header is 'Left'
        - for OUT table the header is 'Joined'
    We want the TEXT half of that group (the second <td> after MV).
    """
    # Build a map of header -> index to be resilient to small layout shifts
    headers = [clean_text(th).lower() for th in table.select("thead th")]
    head_idx = {h: i for i, h in enumerate(headers)}

    # Essentials (these positions are stable on Transfermarkt)
    idx_player = 0  # first column contains the player cell
    idx_age = head_idx.get("age", 1)
    idx_position = head_idx.get("position", 3)
    idx_mv = head_idx.get("market value", 5)

    # After Market Value, there is a 2-td group (badge + text), then Fee
    other_text_offset = 2   # MV + 2 → text cell for Left/Joined club
    fee_offset = 3          # MV + 3 → Fee cell

    rows: List[Dict] = []
    for tr in table.select("tbody tr"):
        tds = tr.find_all("td")
        if not tds or len(tds) <= idx_mv + fee_offset:
            continue

        try:
            player = player_from_cell(tds[idx_player])
            age = clean_text(tds[idx_age])
            position = clean_text(tds[idx_position])
            market_value = clean_text(tds[idx_mv])

            other_td = tds[idx_mv + other_text_offset] if len(tds) > idx_mv + other_text_offset else None
            other_club = club_from_cell(other_td) if other_td else ""

            fee_td = tds[idx_mv + fee_offset] if len(tds) > idx_mv + fee_offset else None
            fee = clean_text(fee_td) if fee_td else ""
        except Exception:
            continue  # skip any row that doesn't match expectations

        rows.append({
            "Team": team,
            "Player": player,
            "Direction": direction,                          # In / Out
            "Age": age,
            "Position": position,
            "Market Value": market_value,
            "Left/Joined": "Left" if direction == "In" else "Joined",
            "Club that left/joined": other_club,
            "Fee": fee,
        })

    return rows


def scrape(league: str, season: int, window: str, out_csv: str) -> None:
    url = f"https://www.transfermarkt.us/premier-league/transfers/wettbewerb/{league}?saison_id={season}&s_w={window}"
    r = requests.get(url, headers=HEADERS, timeout=40)
    r.raise_for_status()

    # Use built-in parser only (no lxml)
    try:
        soup = BeautifulSoup(r.text, "html.parser")
    except FeatureNotFound:
        print("html.parser not available in this Python build.", file=sys.stderr)
        sys.exit(1)

    all_rows: List[Dict] = []

    # Each club’s block looks like <div class="box"> ... <h2 ...>Club</h2> ... two tables ...
    for box in soup.select("div.box"):
        h2 = box.select_one("h2.content-box-headline")
        if not h2:
            continue
        anchors = h2.select("a[title]")
        if not anchors:
            continue
        team = anchors[-1].get("title") or anchors[-1].get_text(strip=True)

        tables = box.select("div.responsive-table > table")
        if not tables:
            continue

        # Convention: first table is IN, second table is OUT
        if len(tables) >= 1:
            all_rows += parse_direction_table(tables[0], team, direction="In")
        if len(tables) >= 2:
            all_rows += parse_direction_table(tables[1], team, direction="Out")

    if not all_rows:
        print("No rows parsed. The page layout may have changed or Cloudflare blocked the content.", file=sys.stderr)

    df = pd.DataFrame(all_rows, columns=[
        "Team", "Player", "Direction", "Age", "Position", "Market Value",
        "Left/Joined", "Club that left/joined", "Fee"
    ])
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows → {out_csv}")


def main():
    p = argparse.ArgumentParser(description="Scrape Transfermarkt transfers (no lxml).")
    p.add_argument("--league", default="GB1", help="League code (GB1 = Premier League)")
    p.add_argument("--season", type=int, default=2025, help="Season ID, e.g. 2025")
    p.add_argument("--window", default="s", choices=["s", "w", "a"],
                   help="Transfer window: s=summer, w=winter, a=all")
    p.add_argument("--out", default="transfers.csv", help="Output CSV path")
    args = p.parse_args()
    scrape(args.league, args.season, args.window, args.out)


if __name__ == "__main__":
    main()
