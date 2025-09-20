# tm_market_values.py
"""
Transfermarkt — Premier League market values (players + club aggregates)

Outputs
  1) tm_market_values_players.csv
     Columns: Rank, Player, Position, Nat, Age, Club, ValueEUR

  2) tm_market_values_clubs.csv (one file that merges two club sources)
     Columns:
       Rank, Club, ValueAug1EUR, CurrentValueEUR, ChangePct,
       ExpenditureEUR, Arrivals, IncomeEUR, Departures, BalanceEUR

Usage:
  python tm_market_values.py \
      --out-players tm_market_values_players.csv \
      --out-clubs   tm_market_values_clubs.csv \
      --season 2025  # for income/expenditure period 25/26 (from=to=2025)
"""

from __future__ import annotations
import argparse, random, re, sys, time
from typing import List, Dict, Optional
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ---------------- HTTP helpers ----------------

UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TM-Scraper/3.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_7) Safari/605.1.15",
]

def new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(UA_POOL),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Referer": "https://www.transfermarkt.us/",
    })
    return s

def get_html(url: str, ses: requests.Session, timeout: int = 35) -> str:
    ses.headers["User-Agent"] = random.choice(UA_POOL)
    r = ses.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

# ---------------- Value parsing/formatting ----------------

def _num_from_tm_eur(s: str) -> float | None:
    """
    TM formats: '€180.00m', '€900k', '€1.37bn', '-', '€500k', '€-120.18m'
    Return raw float euros (can be negative). None if cannot parse.
    """
    if not s:
        return None
    txt = s.strip().replace("€", "").replace(",", "").lower()
    sign = -1.0 if txt.startswith("-") else 1.0
    txt = txt.lstrip("+-")
    mult = 1.0
    if txt.endswith("bn"):
        mult, txt = 1_000_000_000.0, txt[:-2]
    elif txt.endswith("m"):
        mult, txt = 1_000_000.0, txt[:-1]
    elif txt.endswith("k"):
        mult, txt = 1_000.0, txt[:-1]
    try:
        return sign * float(txt) * mult
    except Exception:
        return None

def eur_string(val: float | None) -> str:
    if val is None:
        return ""
    return "€ {:,.2f}".format(float(val))

def pct_from_text(s: str) -> float | None:
    if not s:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*%", s)
    return float(m.group(1)) if m else None

# ---------------- Canonical club label ----------------

TEAM_ALIASES: Dict[str, str] = {
    "Arsenal FC": "Arsenal",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Manchester City": "Manchester City",
    "Manchester United": "Manchester United",
    "Newcastle United": "Newcastle United",
    "Brighton & Hove Albion": "Brighton and Hove Albion",
}
def canon_team(name: str) -> str:
    if not name:
        return ""
    s = re.sub(r"\s+FC$", "", name.strip())
    return TEAM_ALIASES.get(s, s)

# ---------------- Players table ----------------

PLAYERS_BASE = "https://www.transfermarkt.us/premier-league/marktwerte/wettbewerb/GB1"

def _detect_last_page(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    pages = []
    for a in soup.select("ul.tm-pagination a.tm-pagination__link"):
        m = re.search(r"/page/(\d+)", a.get("href") or "")
        if m:
            pages.append(int(m.group(1)))
    return max(pages) if pages else 1

def _parse_players_table(html: str) -> List[dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("div.responsive-table div.grid-view table.items")
    if not table:
        return []

    out: List[dict] = []
    for tr in table.select("tbody > tr"):
        # Only top-level tds (ignore nested inline-table cells)
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 6:
            continue

        # 0) Rank
        rk_text = tds[0].get_text(strip=True)
        if not re.fullmatch(r"\d+", rk_text or ""):
            continue
        rank = int(rk_text)

        # 1) Player + Position (inline-table inside this cell)
        player, pos = "", ""
        inl = tds[1].select_one("table.inline-table")
        if inl:
            a = inl.select_one(".hauptlink a")
            player = (a.get("title") or a.get_text(" ", strip=True)) if a else inl.get_text(" ", strip=True)
            trs = inl.select("tr")
            if len(trs) >= 2:
                pos = trs[1].get_text(" ", strip=True)
        player = player.strip()
        pos = pos.strip()

        # 2) Nationalities (flag titles)
        nats = []
        for im in tds[2].select("img"):
            val = im.get("title") or im.get("alt")
            if val:
                nats.append(val.strip())
        nat = " / ".join(nats) if nats else tds[2].get_text(" ", strip=True)

        # 3) Age
        age = None
        atext = tds[3].get_text(strip=True)
        if atext:
            m = re.search(r"\d+", atext)
            age = int(m.group(0)) if m else None

        # 4) Club (prefer link title, fallback to badge title/alt, else text)
        club = ""
        a = tds[4].select_one("a[title]")
        if a and a.get("title"):
            club = a.get("title").strip()
        if not club:
            im = tds[4].select_one("img")
            if im and (im.get("title") or im.get("alt")):
                club = (im.get("title") or im.get("alt")).strip()
        if not club:
            club = tds[4].get_text(" ", strip=True)
        club = canon_team(club)

        # 5) Market value
        aval = tds[5].select_one("a")
        raw_val = (aval.get_text(" ", strip=True) if aval else tds[5].get_text(" ", strip=True))
        val_eur = eur_string(_num_from_tm_eur(raw_val))

        out.append({
            "Rank": rank,
            "Player": player,
            "Position": pos,
            "Nat": nat,
            "Age": age,
            "Club": club,
            "ValueEUR": val_eur,
        })
    return out

def fetch_players(ses: requests.Session, sleep: float = 1.0, max_pages: int = 8) -> pd.DataFrame:
    html1 = get_html(PLAYERS_BASE, ses)
    rows = _parse_players_table(html1)
    last = min(_detect_last_page(html1), max_pages)
    for p in range(2, last + 1):
        time.sleep(max(0.0, sleep) + random.random() * 0.4)
        try:
            htmlp = get_html(f"{PLAYERS_BASE}/page/{p}", ses)
            rows += _parse_players_table(htmlp)
        except requests.RequestException as e:
            sys.stderr.write(f"[WARN] players page {p} failed: {e}\n")
            break
    df = pd.DataFrame(rows).sort_values("Rank").reset_index(drop=True)
    # ensure order
    cols = ["Rank","Player","Position","Nat","Age","Club","ValueEUR"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

# ---------------- Clubs: market values + income/expenditures ----------------

CLUB_VALUES_URL = "https://www.transfermarkt.us/premier-league/marktwerteverein/wettbewerb/GB1"
INOUT_URL_TMPL = ("https://www.transfermarkt.us/premier-league/einnahmenausgaben/wettbewerb/GB1/"
                  "ids/a/sa//saison_id/{y}/saison_id_bis/{y}/nat/0/pos//w_s//intern/0")

def parse_club_values(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("div.responsive-table div.grid-view table.items")
    rows = []
    if not table:
        return pd.DataFrame(columns=["Rank","Club","ValueAug1EUR","CurrentValueEUR","ChangePct"])
    for tr in table.select("tbody > tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 7:
            continue
        rk = tds[0].get_text(strip=True)
        if not re.fullmatch(r"\d+", rk or ""):
            continue
        rank = int(rk)
        # club name from link title
        club = ""
        a = tds[2].select_one("a[title]") or tds[2].select_one("a")
        if a:
            club = a.get("title") or a.get_text(" ", strip=True)
        club = canon_team(club)
        # Value Aug 1 (td[4]) and current value (td[5]), pct (td[6])
        v_aug = _num_from_tm_eur(tds[4].get_text(" ", strip=True))
        v_cur = _num_from_tm_eur(tds[5].get_text(" ", strip=True))
        pct  = pct_from_text(tds[6].get_text(" ", strip=True))
        rows.append({
            "Rank": rank,
            "Club": club,
            "ValueAug1EUR": eur_string(v_aug),
            "CurrentValueEUR": eur_string(v_cur),
            "ChangePct": pct if pct is not None else "",
        })
    return pd.DataFrame(rows).sort_values("Rank").reset_index(drop=True)

def parse_income_out(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("div.responsive-table div.grid-view table.items")
    cols = ["Club","ExpenditureEUR","Arrivals","IncomeEUR","Departures","BalanceEUR"]
    if not table:
        return pd.DataFrame(columns=cols)
    out = []
    for tr in table.select("tbody > tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 8:
            continue
        # club name in tds[2]
        a = tds[2].select_one("a[title]") or tds[2].select_one("a")
        club = canon_team(a.get("title") or a.get_text(" ", strip=True))
        exp = _num_from_tm_eur(tds[3].get_text(" ", strip=True))
        arr = tds[4].get_text(strip=True)
        arr = int(re.sub(r"[^\d]", "", arr)) if re.search(r"\d", arr) else None
        inc = _num_from_tm_eur(tds[5].get_text(" ", strip=True))
        dep = tds[6].get_text(strip=True)
        dep = int(re.sub(r"[^\d]", "", dep)) if re.search(r"\d", dep) else None
        bal = _num_from_tm_eur(tds[7].get_text(" ", strip=True))
        out.append({
            "Club": club,
            "ExpenditureEUR": eur_string(exp),
            "Arrivals": arr if arr is not None else "",
            "IncomeEUR": eur_string(inc),
            "Departures": dep if dep is not None else "",
            "BalanceEUR": eur_string(bal),
        })
    return pd.DataFrame(out)

def build_club_csv(ses: requests.Session, season_y: int) -> pd.DataFrame:
    # market values
    html_vals = get_html(CLUB_VALUES_URL, ses)
    df_vals = parse_club_values(html_vals)

    # income/expenditures for a specific period (25/26 etc.)
    html_io = get_html(INOUT_URL_TMPL.format(y=season_y), ses)
    df_io = parse_income_out(html_io)

    # merge on Club
    df = df_vals.merge(df_io, on="Club", how="left")
    # Ensure column order
    cols = ["Rank","Club","ValueAug1EUR","CurrentValueEUR","ChangePct",
            "ExpenditureEUR","Arrivals","IncomeEUR","Departures","BalanceEUR"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].sort_values("Rank").reset_index(drop=True)

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Transfermarkt PL market values (players + club aggregates)")
    ap.add_argument("--out-players", default="tm_market_values_players.csv")
    ap.add_argument("--out-clubs",   default="tm_market_values_clubs.csv")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--max-pages", type=int, default=8)
    ap.add_argument("--season", type=int, default=2025, help="For income/expenditure period (e.g. 2025 ⇒ 25/26)")
    args = ap.parse_args()

    ses = new_session()

    # Players
    players = fetch_players(ses, sleep=args.sleep, max_pages=args.max_pages)
    players.to_csv(args.out_players, index=False, encoding="utf-8")
    print(f"✔ Wrote {args.out_players} ({len(players)} rows)")
    if not players.empty:
        print(players.head(5).to_string(index=False))

    # Clubs (values + income/expenditures)
    clubs = build_club_csv(ses, args.season)
    clubs.to_csv(args.out_clubs, index=False, encoding="utf-8")
    print(f"✔ Wrote {args.out_clubs} ({len(clubs)} rows)")
    if not clubs.empty:
        print(clubs.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
