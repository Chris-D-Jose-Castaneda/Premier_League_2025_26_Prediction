# odds.py — Combined scraper
# - Google "Win probability": old logic (with consent banner handling) that worked for you
# - OneFootball "Who will win?": improved logic (MD-card names, hydration nudges, multiple selectors)
#
# Requirements:
#   pip install selenium pandas python-dateutil
#   Firefox + geckodriver in PATH
#
# Usage:
#   python odds.py                       # auto-pick next/active matchday
#   python odds.py --md 3                # force MD 3
#   python odds.py --csv fixtures_2025_26.csv --out md3_odds.csv --md 3

from __future__ import annotations
import argparse, random, re, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dateutil import parser as dtparse

from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from urllib.parse import quote_plus, urljoin


# =============================== Browser setup ===============================

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
    "Gecko/20100101 Firefox/121.0"
)

def headless_firefox() -> webdriver.Firefox:
    opts = FirefoxOptions()
    opts.add_argument("-headless")
    opts.set_preference("general.useragent.override", USER_AGENT)
    opts.set_preference("dom.webdriver.enabled", False)
    opts.set_preference("useAutomationExtension", False)
    try:
        drv = webdriver.Firefox(options=opts, service=FirefoxService())
    except WebDriverException as e:
        raise SystemExit(f"Could not start Firefox/Geckodriver: {e}")
    drv.set_page_load_timeout(40)
    drv.set_script_timeout(40)
    drv.implicitly_wait(0)
    return drv

def wait_css(drv, css, t=12):
    return WebDriverWait(drv, t).until(EC.presence_of_element_located((By.CSS_SELECTOR, css)))

def js_click(drv, el):
    try:
        drv.execute_script("arguments[0].scrollIntoView({behavior:'instant',block:'center'});", el)
        el.click()
    except Exception:
        try:
            drv.execute_script("arguments[0].click();", el)
        except Exception:
            pass

def get_inner_text(drv, el):
    try:
        return drv.execute_script("return arguments[0].innerText;", el) or ""
    except Exception:
        return el.text or ""


# =============================== Consent banners (old Google logic) ===============================

def dismiss_banners(drv):
    """Best-effort cookie/consent dismissal (Google + generic)."""
    # Google "Accept all" button ids vary by region/build
    selectors = [
        "#L2AGLb",               # Google "I agree"
        "#W0wltc",               # Google "Accept all"
        "button[aria-label*='Accept']", "button[aria-label*='agree']",
        "//button[contains(., 'I agree')]", "//button[contains(., 'Accept all')]",
        "//button[contains(., 'Accept')]", "//button[contains(., 'OK')]",
    ]
    for sel in selectors:
        try:
            if sel.startswith("//"):
                els = drv.find_elements(By.XPATH, sel)
            else:
                els = drv.find_elements(By.CSS_SELECTOR, sel)
            for el in els:
                if el.is_displayed():
                    js_click(drv, el)
                    time.sleep(0.2)
        except Exception:
            pass


# =============================== Team normalization ===============================

ALIASES = {
    "arsenal":{"arsenal"},
    "aston villa":{"aston villa","villa"},
    "brentford":{"brentford"},
    "brighton & hove albion":{"brighton & hove albion","brighton and hove albion","brighton"},
    "bournemouth":{"afc bournemouth","bournemouth"},
    "burnley":{"burnley"},
    "chelsea":{"chelsea"},
    "crystal palace":{"crystal palace","palace"},
    "everton":{"everton"},
    "fulham":{"fulham"},
    "leeds united":{"leeds united","leeds"},
    "liverpool":{"liverpool","liverpool fc"},
    "manchester city":{"manchester city","man city","man. city"},
    "manchester united":{"manchester united","man united","man utd","man. united"},
    "newcastle united":{"newcastle united","newcastle"},
    "nottingham forest":{"nottingham forest","nottm forest","nottingham"},
    "tottenham hotspur":{"tottenham hotspur","tottenham","spurs"},
    "west ham united":{"west ham united","west ham"},
    "wolverhampton wanderers":{"wolverhampton wanderers","wolverhampton","wolves"},
    "sunderland":{"sunderland"},
}
def norm_team(s: str) -> str:
    x = re.sub(r"[^\w& ]+","",(s or "").lower()).strip()
    x = x.replace("fc","").replace("afc ","")
    x = re.sub(r"\s+"," ",x)
    for canon, aliases in ALIASES.items():
        if x in aliases: return canon
    return x


# =============================== Fixtures & MD (timezone-safe) ===============================

@dataclass
class Fixture:
    md: int
    when_utc: Optional[datetime]
    home: str
    away: str

def to_utc(dt)->Optional[datetime]:
    if dt is None: return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

def read_fixtures(path: Path)->List[Fixture]:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    md_col   = next((cols[k] for k in cols if k in {"md","matchday","match_day","round"}), None)
    home_col = next((cols[k] for k in cols if k in {"home","home team","home_team","home club"}), None)
    away_col = next((cols[k] for k in cols if k in {"away","away team","away_team","away club"}), None)
    date_col = next((cols[k] for k in cols if k in {"date","kickoff","kick_off","datetime","utc","time"}), None)
    if not (md_col and home_col and away_col):
        raise SystemExit("CSV must include MD/Home/Away columns.")
    out: List[Fixture] = []
    for _, r in df.iterrows():
        md   = int(r[md_col])
        home = str(r[home_col]).strip()
        away = str(r[away_col]).strip()
        dt   = None
        if date_col and pd.notna(r[date_col]):
            try: dt = dtparse.parse(str(r[date_col]), fuzzy=True)
            except: dt = None
        out.append(Fixture(md=md, when_utc=to_utc(dt), home=home, away=away))
    return out

def choose_md(fixtures: List[Fixture], forced: Optional[int]) -> int:
    if forced: return int(forced)
    now = datetime.now(timezone.utc)
    md_dates: Dict[int, List[datetime]] = {}
    for f in fixtures:
        if f.when_utc:
            md_dates.setdefault(f.md, []).append(f.when_utc)
    upcoming = sorted([md for md, ds in md_dates.items() if any(d >= now for d in ds)])
    return int(upcoming[0] if upcoming else min(f.md for f in fixtures))


# =============================== GOOGLE — keep old working logic ===============================

def google_win_probability(drv, home: str, away: str) -> Tuple[Optional[int], Optional[int], Optional[int], str]:
    """Open Google, click sports header, read Win probability (home/draw/away)."""
    q   = f"{home} vs {away}"
    url = f"https://www.google.com/search?q={quote_plus(q)}&hl=en&gl=us&pws=0&nfpr=1"
    drv.get(url)
    time.sleep(random.uniform(0.6, 1.0))
    dismiss_banners(drv)  # <— this is what often makes the difference

    # Click the immersive header
    try:
        header = WebDriverWait(drv, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'div[data-attrid="TLOsrpMatchHeader"]'))
        )
        js_click(drv, header)
    except TimeoutException:
        return None, None, None, drv.current_url

    # Wait for the Win probability card and parse percentages
    try:
        card = WebDriverWait(drv, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.liveresults-sports-immersive__lr-imso-ss-wp-card"))
        )
        tds = card.find_elements(
            By.CSS_SELECTOR,
            "table.liveresults-sports-immersive__lr-imso-ss-wp-tnp tr.liveresults-sports-immersive__lr-imso-ss-wp-tp td"
        )
        vals = []
        for td in tds[:3]:
            m = re.search(r"(\d+)\s*%", td.text)
            vals.append(int(m.group(1)) if m else None)
        if len(vals) == 3:
            return vals[0], vals[1], vals[2], drv.current_url
    except TimeoutException:
        pass
    return None, None, None, drv.current_url


# =============================== ONEFOOTBALL — improved logic ===============================

FIXTURES_BASE = "https://onefootball.com/en/competition/premier-league-9/fixtures"

def of_links_for_md(drv, md: int) -> List[Tuple[str, str, str]]:
    """Return (home, away, url) for Matchday md as printed on the MD list itself."""
    drv.get(FIXTURES_BASE)
    time.sleep(0.8)
    # Nudge scroll to let lists hydrate
    for y in (200, 800, 1400):
        drv.execute_script(f"window.scrollTo(0,{y});")
        time.sleep(0.2)
    try:
        ul = WebDriverWait(drv, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, f"//h3[contains(., 'Matchday {md}')]/ancestor::div[contains(@class,'SectionHeader_container')]/following-sibling::ul[1]")
            )
        )
    except TimeoutException:
        return []
    out = []
    cards = ul.find_elements(By.CSS_SELECTOR, "a.MatchCard_matchCard__iOv4G")
    for a in cards:
        names = a.find_elements(By.CSS_SELECTOR, ".SimpleMatchCardTeam_simpleMatchCardTeam__name__7Ud8D")
        if len(names) >= 2:
            home = names[0].text.strip()
            away = names[1].text.strip()
        else:
            home = away = ""
        href = a.get_attribute("href") or ""
        if href and not href.startswith("http"):
            href = urljoin(FIXTURES_BASE, href)
        out.append((home, away, href))
    # unique by URL
    seen, uniq = set(), []
    for h, a, u in out:
        if u and u not in seen:
            uniq.append((h, a, u))
            seen.add(u)
    return uniq

def _extract_three_percents_from_widget(drv, widget)->Optional[Tuple[int,int,int]]:
    # 1) Buttons innerText (most builds)
    btns = widget.find_elements(By.CSS_SELECTOR, "ul.MatchPrediction_buttons__mz8Sv button")
    if btns:
        texts = [get_inner_text(drv, b) for b in btns]
        nums = []
        for tx in texts:
            m = re.search(r"(\d+)\s*%", tx.replace("\n"," "))
            nums.append(int(m.group(1)) if m else None)
        if len(nums) == 3 and all(n is not None for n in nums):
            return nums[0], nums[1], nums[2]
    # 2) Any descendant with a % text, left→right
    spans = widget.find_elements(By.XPATH, ".//*[contains(text(), '%')]")
    found = [int(re.search(r"(\d+)", s.text).group(1)) for s in spans if re.search(r"(\d+)", s.text)]
    if len(found) >= 3:
        return found[0], found[1], found[2]
    # 3) CSS bar widths (width: NN%)
    bars = widget.find_elements(By.XPATH, './/*[@style[contains(., "width:")]]')
    widths = []
    for b in bars:
        st = b.get_attribute("style") or ""
        m = re.search(r"width\s*:\s*(\d+)\s*%?", st)
        if m: widths.append(int(m.group(1)))
    if len(widths) >= 3:
        return widths[0], widths[1], widths[2]
    return None

def of_crowd_odds(drv, url: str) -> Tuple[Optional[int], Optional[int], Optional[int], str]:
    drv.get(url)
    time.sleep(0.6)
    # Hydration nudges
    for _ in range(3):
        drv.execute_script("window.scrollBy(0, 400);")
        time.sleep(0.15)
    # Find widget
    try:
        widget = WebDriverWait(drv, 8).until(
            EC.presence_of_element_located((By.XPATH, "//section[contains(@class,'MatchPrediction_wrapper')]"))
        )
    except TimeoutException:
        return None, None, None, url
    # Bring to center, click home to reveal if required
    drv.execute_script("arguments[0].scrollIntoView({behavior:'instant',block:'center'});", widget)
    time.sleep(0.25)
    try:
        btns = widget.find_elements(By.CSS_SELECTOR, "ul.MatchPrediction_buttons__mz8Sv button")
        if btns:
            js_click(drv, btns[0])
            time.sleep(0.6)
    except Exception:
        pass
    # Wait up to 6s for any extraction mode to succeed
    deadline = time.time() + 6.0
    while time.time() < deadline:
        res = _extract_three_percents_from_widget(drv, widget)
        if res:
            return res[0], res[1], res[2], url
        time.sleep(0.25)
    res = _extract_three_percents_from_widget(drv, widget)
    return (res[0], res[1], res[2], url) if res else (None, None, None, url)


# =============================== Main ===============================

@dataclass
class RowOut:
    Matchday: int; Home: str; Away: str
    Google_Home: Optional[int]; Google_Draw: Optional[int]; Google_Away: Optional[int]
    OneFootball_Home: Optional[int]; OneFootball_Draw: Optional[int]; OneFootball_Away: Optional[int]
    Google_URL: str; OneFootball_URL: str

def main():
    ap = argparse.ArgumentParser(description="Scrape Google Win Probability + OneFootball crowd % for a matchday.")
    ap.add_argument("--csv", default="fixtures_2025_26.csv")
    ap.add_argument("--out", default="odds_out.csv")
    ap.add_argument("--md", type=int, default=None)
    args = ap.parse_args()

    fixtures = read_fixtures(Path(args.csv))
    md = choose_md(fixtures, args.md)
    md_fixtures = [f for f in fixtures if f.md == md][:10]
    if not md_fixtures:
        raise SystemExit(f"No fixtures for MD {md}")

    print(f"[MD {md}] {len(md_fixtures)} fixtures…")

    drv = headless_firefox()
    rows: List[RowOut] = []
    try:
        # Build (home,away)->URL from MD-card (reliable naming)
        print("[OneFootball] Collecting MD links…")
        md_cards = of_links_for_md(drv, md)  # [(home, away, url), ...]
        card_map: Dict[Tuple[str,str], str] = {(norm_team(h), norm_team(a)): u for h,a,u in md_cards if h and a and u}

        # Scrape OneFootball first (fewer rate/AB issues once warmed)
        print("[OneFootball] Scraping crowd % per match page…")
        of_perc: Dict[Tuple[str,str], Tuple[Optional[int],Optional[int],Optional[int],str]] = {}
        for (h,a,u) in md_cards:
            key = (norm_team(h), norm_team(a))
            oh, od, oa, ou = of_crowd_odds(drv, u)
            of_perc[key] = (oh, od, oa, ou)
            print(f"  - {h} vs {a}: {oh}/{od}/{oa}")

        # Now Google (old working flow with consent handling)
        print("[Google] Scraping Win probability…")
        for f in md_fixtures:
            g_h, g_d, g_a, g_url = google_win_probability(drv, f.home, f.away)

            key  = (norm_team(f.home), norm_team(f.away))
            of_h = of_d = of_a = None
            of_url = ""
            if key in of_perc:
                of_h, of_d, of_a, of_url = of_perc[key]
            else:
                rkey = (key[1], key[0])  # if card orientation flips
                if rkey in of_perc:
                    tmp_h, tmp_d, tmp_a, of_url = of_perc[rkey]
                    of_h, of_d, of_a = tmp_a, tmp_d, tmp_h  # swap sides

            print(f"  • {f.home} vs {f.away} | Google {g_h}/{g_d}/{g_a} | OneFootball {of_h}/{of_d}/{of_a}")
            rows.append(RowOut(
                Matchday=md, Home=f.home, Away=f.away,
                Google_Home=g_h, Google_Draw=g_d, Google_Away=g_a,
                OneFootball_Home=of_h, OneFootball_Draw=of_d, OneFootball_Away=of_a,
                Google_URL=g_url, OneFootball_URL=of_url
            ))

    finally:
        try: drv.quit()
        except Exception: pass

    out = Path(args.out)
    pd.DataFrame([r.__dict__ for r in rows]).to_csv(out, index=False)
    print(f"[OK] Wrote {len(rows)} rows → {out.resolve()}")

if __name__ == "__main__":
    main()
