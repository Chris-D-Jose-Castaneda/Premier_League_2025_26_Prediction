#!/usr/bin/env python3
# match_statistics.py  — PL Lineups + Stats + Recap (+ OneFootball crowd poll & odds)
# - Input fixtures CSV (md/home/away auto-detected)
# - Resume-safe: writes a row after each fixture; skips already-scraped fixtures
# - Stops on first NOT-FT fixture in an MD
# - Robust poll scraper: interacts with buttons if needed; handles iframes; optional OF fallback

from __future__ import annotations
import re, time, sys, argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from urllib.parse import urljoin

import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Optional VADER
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    HAVE_VADER = True
except Exception:
    HAVE_VADER = False

PL_BASE = "https://www.premierleague.com"

# ---------- small utils ----------
def wait(d, s=15): return WebDriverWait(d, s)
def clean(s: Optional[str]) -> str: return re.sub(r"\s+", " ", (s or "").strip())
def ensure_url(u: str) -> str:
    u = (u or "").strip()
    if u.startswith("/"): return urljoin(PL_BASE, u)
    if not u.startswith("http"): return f"{PL_BASE}{'' if u.startswith('/en/') else '/en/'}{u.lstrip('/')}"
    return u
def parse_match_id(u: str) -> str:
    m = re.search(r"/match/(\d+)", u or ""); return m.group(1) if m else (u or "")

def accept_cookies_if_present(driver):
    try:
        btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        )
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(0.25)
    except Exception:
        pass

# ---------- team canonicalisation ----------
ALIASES = {
    "afc bournemouth":"bournemouth","nottm forest":"nottingham forest","nott m forest":"nottingham forest",
    "west ham":"west ham united","newcastle":"newcastle united","man utd":"manchester united",
    "manchester utd":"manchester united","man city":"manchester city","spurs":"tottenham hotspur",
    "brighton":"brighton and hove albion","wolves":"wolverhampton wanderers","leeds":"leeds united",
}
def canon_team(name: str) -> str:
    s = (name or "").lower().replace("&","and")
    s = re.sub(r"[’'`]", "", s); s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s)

# ---------- formation cleaning (block dates/garbage) ----------
FORM_RE = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")
DATEISH_RE = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})", re.I)
def clean_formation(txt: str) -> str:
    s = clean((txt or "").replace("Formation", ""))
    if DATEISH_RE.search(s): return ""
    m = FORM_RE.match(s)
    if not m:
        m = re.match(r"^\s*(\d{1,2}-\d{1,2}-\d{1,2})", s)
        if m: s = m.group(1)
    return s if FORM_RE.match(s) else ""

# ---------- fixtures CSV ----------
def detect_md_home_away(df: pd.DataFrame,
                        md_col: Optional[str], home_col: Optional[str], away_col: Optional[str]) -> Tuple[str,str,str]:
    md_c = md_col if md_col and md_col in df.columns else \
           next((c for c in df.columns if c.lower() in
                 {"md","matchweek","mw","matchday","match_day","round","week","gameweek","gw"}), None)
    if md_c is None: raise ValueError("Could not detect matchweek column (use --md-col).")
    home_c = home_col if home_col and home_col in df.columns else \
             next((c for c in df.columns if c.lower() in {"home","home_team","homeclub","home side","home-side"}), None)
    away_c = away_col if away_col and away_col in df.columns else \
             next((c for c in df.columns if c.lower() in {"away","away_team","awayclub","away side","away-side"}), None)
    if home_c is None or away_c is None:
        raise ValueError("Could not detect home/away columns (use --home-col/--away-col).")
    return md_c, home_c, away_c

def load_pairs_by_md(fixtures_csv: str, start_md: int,
                     md_col: Optional[str], home_col: Optional[str], away_col: Optional[str]) -> Dict[int, List[Tuple[str,str]]]:
    df = pd.read_csv(fixtures_csv)
    mdc, hc, ac = detect_md_home_away(df, md_col, home_col, away_col)
    df["__md"] = pd.to_numeric(df[mdc], errors="coerce").astype("Int64")
    df = df[df["__md"].notna()]
    plan: Dict[int, List[Tuple[str,str]]] = {}
    for md, sub in df.groupby("__md"):
        md = int(md)
        if md < start_md: continue
        pairs, seen = [], set()
        for _, r in sub.iterrows():
            h, a = str(r[hc]), str(r[ac])
            key = (canon_team(h), canon_team(a))
            if key in seen: continue
            pairs.append((h,a)); seen.add(key)
            if len(pairs) == 10: break
        plan[md] = pairs
    return dict(sorted(plan.items()))

# ---------- fetch links for an MD ----------
def get_links_and_status(driver, season: int, md: int,
                         expected_pairs: List[Tuple[str,str]]) -> Dict[Tuple[str,str], Tuple[str,bool]]:
    url = f"{PL_BASE}/en/matches?competition=8&season={season}&matchweek={md}"
    driver.get(url); accept_cookies_if_present(driver)
    try:
        wait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="matchList"]')))
    except Exception:
        return {}

    card_map: Dict[Tuple[str,str], Tuple[str,bool]] = {}
    cards = driver.find_elements(By.CSS_SELECTOR, 'a[data-testid="matchCard"]') or \
            driver.find_elements(By.CSS_SELECTOR, 'a[href*="/match/"]')
    for card in cards:
        href = card.get_attribute("href"); 
        if not href: continue
        teams = card.find_elements(By.CSS_SELECTOR, '[data-testid="matchCardTeamFullName"]')
        if len(teams) == 2:
            h = canon_team(teams[0].text); a = canon_team(teams[1].text)
        else:
            lines = [t for t in (card.text or "").splitlines() if t.strip()]
            if len(lines) < 2: continue
            h = canon_team(lines[0]); a = canon_team(lines[1])
        is_ft = bool(card.find_elements(By.CSS_SELECTOR, "[data-testid='matchCardFullTime']")) \
                or " FT " in (" " + (card.text or "") + " ")
        card_map[(h,a)] = (ensure_url(href), is_ft)

    out: Dict[Tuple[str,str], Tuple[str,bool]] = {}
    for (home, away) in expected_pairs:
        key = (canon_team(home), canon_team(away))
        if key in card_map: out[key] = card_map[key]
    return out

# ---------- OneFootball poll helpers ----------
PCT_RE = re.compile(r"(\d+)\s*%")
def _get_poll_buttons(ctx) -> List:
    return ctx.find_elements(By.CSS_SELECTOR, "ul[class*='MatchPrediction_buttons'] button")

def _read_percentages_from_buttons(btns) -> Tuple[str,str,str]:
    try:
        ph = PCT_RE.search(btns[0].text or ""); pd_ = PCT_RE.search(btns[1].text or ""); pa = PCT_RE.search(btns[2].text or "")
        return (ph.group(1) if ph else "", pd_.group(1) if pd_ else "", pa.group(1) if pa else "")
    except Exception:
        return ("","","")

def _click_force(driver, el):
    try:
        ActionChains(driver).move_to_element(el).pause(0.05).click(el).perform()
        return True
    except Exception:
        try:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            driver.execute_script("arguments[0].click();", el)
            return True
        except Exception:
            try:
                driver.execute_script("arguments[0].removeAttribute('disabled');", el)
                driver.execute_script("arguments[0].click();", el)
                return True
            except Exception:
                return False

def _scrape_poll_in_context(driver, ctx) -> dict:
    data = {"poll_pct_home":"", "poll_pct_draw":"", "poll_pct_away":"", "poll_votes":""}
    # Locate the "Who will win?" block to scroll into view (more reliable rendering)
    headings = ctx.find_elements(By.XPATH, "//*[contains(@class,'MatchPrediction_title') or normalize-space()='Who will win?']")
    if headings:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", headings[0])
        time.sleep(0.2)

    btns = _get_poll_buttons(ctx)
    if len(btns) == 3:
        h, d, a = _read_percentages_from_buttons(btns)
        if not (h and d and a):
            # Interact to make the widget reveal the % values
            for b in btns:
                _click_force(driver, b)
                time.sleep(0.15)
            btns = _get_poll_buttons(ctx)
            h, d, a = _read_percentages_from_buttons(btns)
        data["poll_pct_home"], data["poll_pct_draw"], data["poll_pct_away"] = h, d, a

    # total votes
    try:
        msg_el = ctx.find_element(By.CSS_SELECTOR, "[class*='MatchPrediction_message']")
        m = re.search(r"Based on\s+([0-9,]+)\s+predictions", msg_el.text or "")
        if m: data["poll_votes"] = m.group(1).replace(",", "")
    except Exception:
        pass
    return data

def scrape_crowd_poll(driver) -> dict:
    """
    1) Try current DOM
    2) Try iframes
    3) If an anchor to onefootball.com/en/match/* exists, open that page in a new tab,
       scrape there, then come back.
    """
    # 1) page itself
    data = _scrape_poll_in_context(driver, driver)
    if any(data.values()):
        return data

    # 2) iframes
    try:
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        for fr in iframes:
            src = (fr.get_attribute("src") or "").lower()
            if "onefootball" not in src and "ofa" not in src and "who will win" not in (fr.get_attribute("title") or "").lower():
                continue
            driver.switch_to.frame(fr)
            data = _scrape_poll_in_context(driver, driver)
            driver.switch_to.default_content()
            if any(data.values()):
                return data
    except Exception:
        try: driver.switch_to.default_content()
        except Exception: pass

    # 3) follow OF link if present
    try:
        of = driver.find_elements(By.CSS_SELECTOR, "a[href*='onefootball.com/en/match/']")
        if of:
            main = driver.current_window_handle
            url = of[0].get_attribute("href")
            driver.execute_script("window.open(arguments[0], '_blank');", url)
            wait(driver, 8).until(EC.number_of_windows_to_be(2))
            driver.switch_to.window([w for w in driver.window_handles if w != main][0])
            wait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            data = _scrape_poll_in_context(driver, driver)
            driver.close()
            driver.switch_to.window(main)
            if any(data.values()):
                return data
    except Exception:
        try:
            driver.switch_to.window(main)
        except Exception:
            pass

    return {"poll_pct_home":"", "poll_pct_draw":"", "poll_pct_away":"", "poll_votes":""}

# ---------- odds (best-effort) ----------
ODDS_DEC = re.compile(r"(?<!\d)(\d+\.\d{2})(?!\d)")
ODDS_FR  = re.compile(r"(\d+)\s*/\s*(\d+)")
def _odds_from_text(txt: str) -> Tuple[str,str,str]:
    if not txt: return ("","","")
    decs = [float(x) for x in ODDS_DEC.findall(txt)]
    if len(decs) >= 3:
        decs = decs[:3]; return (f"{decs[0]:.2f}", f"{decs[1]:.2f}", f"{decs[2]:.2f}")
    frs = ODDS_FR.findall(txt)
    if len(frs) >= 3:
        to_dec = lambda p: f"{(float(p[0])/float(p[1]) + 1):.2f}"
        return (to_dec(frs[0]), to_dec(frs[1]), to_dec(frs[2]))
    return ("","","")

def scrape_any_odds(driver) -> Tuple[str,str,str]:
    try:
        holders = []
        holders += driver.find_elements(By.CSS_SELECTOR, "[class*='odds'], [class*='bet'], [id*='odds'], [id*='bet']")
        holders += driver.find_elements(By.XPATH, "//*[contains(translate(., 'ODDSBET', 'oddsbet'),'odds') or contains(translate(., 'ODDSBET', 'oddsbet'),'bet')]")
        text = "  ".join(h.text for h in holders if h.text and len(h.text) < 6000)
        return _odds_from_text(text)
    except Exception:
        return ("","","")

# ---------- recap / lineups / stats ----------
def scrape_recap(driver, base_url: str, want_sentiment: bool) -> dict:
    url = base_url + "?tab=recap"
    driver.get(url); accept_cookies_if_present(driver)
    try:
        wait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".single-match-page__tabs")))
    except Exception:
        return {"recap_url": url, "recap_text": "", "motm": "", "motm_position": "",
                "poll_pct_home":"", "poll_pct_draw":"", "poll_pct_away":"", "poll_votes":"",
                "odds_home":"", "odds_draw":"", "odds_away":""}

    summary = " ".join(e.text for e in driver.find_elements(By.CSS_SELECTOR, ".match-report__summary *")).strip()
    body    = " ".join(e.text for e in driver.find_elements(By.CSS_SELECTOR, ".match-report__content *")).strip()
    recap   = clean((summary + " " + body).strip())

    motm, motm_pos = "", ""
    try:
        win = driver.find_element(By.CSS_SELECTOR, ".motm.motm--winner")
        first = win.find_element(By.CSS_SELECTOR, ".motm__player-name--first-name").text
        last  = win.find_element(By.CSS_SELECTOR, ".motm__player-name--last-name").text
        motm = clean(f"{first} {last}")
        motm_pos = clean(win.find_element(By.CSS_SELECTOR, ".motm__position").text)
    except Exception:
        pass

    poll = scrape_crowd_poll(driver)
    odds_home, odds_draw, odds_away = scrape_any_odds(driver)

    out = {"recap_url": url, "recap_text": recap, "motm": motm, "motm_position": motm_pos,
           "poll_pct_home": poll["poll_pct_home"], "poll_pct_draw": poll["poll_pct_draw"],
           "poll_pct_away": poll["poll_pct_away"], "poll_votes": poll["poll_votes"],
           "odds_home": odds_home, "odds_draw": odds_draw, "odds_away": odds_away}
    if want_sentiment and recap:
        try:
            sia = SentimentIntensityAnalyzer()
            s = sia.polarity_scores(recap)
            out["recap_sentiment_compound"] = s["compound"]
            out["recap_sentiment_pos"] = s["pos"]
            out["recap_sentiment_neg"] = s["neg"]
            out["recap_sentiment_neu"] = s["neu"]
        except Exception:
            pass
    return out

def scrape_lineups(driver, base_url: str) -> Optional[dict]:
    url = base_url + "?tab=lineups"
    driver.get(url); accept_cookies_if_present(driver)
    try:
        wait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="lineupsFormations"]')))
    except Exception:
        return None
    data = {}
    blocks = driver.find_elements(By.CSS_SELECTOR, '[data-testid="lineupsTeamFormation"]')
    for side, blk in zip(["home","away"], blocks):
        team = clean(blk.find_element(By.CSS_SELECTOR, ".lineups-team-info__team-name").text)
        formation = clean_formation(blk.find_element(By.CSS_SELECTOR, ".lineups-team-info__formation").text)
        # XI
        xi = []
        for a in blk.find_elements(By.CSS_SELECTOR, '[data-testid="teamFormation"] a[data-testid="lineupsPlayer"]'):
            nm = ""
            try: nm = a.find_element(By.CSS_SELECTOR, ".player-headshot img[alt]").get_attribute("alt") or ""
            except Exception: pass
            if not nm: nm = a.text
            nm = re.sub(r"^\s*\d+\s*", "", nm).strip()
            xi.append(nm)
        # subs used (+minute)
        subs_used = []
        try:
            subs_section = driver.find_element(By.CSS_SELECTOR, 'section[data-testid="lineupsSubs"]')
            uls = subs_section.find_elements(By.CSS_SELECTOR, "ul.squad-list")
            ul  = uls[0] if side == "home" else uls[1]
            for li in ul.find_elements(By.CSS_SELECTOR, "[data-testid='squadListItem']"):
                if not li.find_elements(By.CSS_SELECTOR, ".lineups-player-badge--sub-on"): continue
                try: name = li.find_element(By.CSS_SELECTOR, "[data-testid='squadPlayerName']").text
                except Exception:
                    try: name = li.find_element(By.CSS_SELECTOR, ".player-headshot img[alt]").get_attribute("alt")
                    except Exception: name = ""
                minute = ""
                try: minute = li.find_element(By.CSS_SELECTOR, ".lineups-player-badge--sub-on .lineups-player-badge__label").text
                except Exception: pass
                subs_used.append(f"{name} ({minute})" if minute else name)
        except Exception:
            pass
        data[f"{side}_team"] = team
        data[f"{side}_formation"] = formation
        data[f"{side}_xi"] = "; ".join(xi)
        data[f"{side}_subs_used"] = "; ".join(subs_used)
    return data

def scrape_stats(driver, base_url: str) -> Optional[dict]:
    url = base_url + "?tab=stats"
    driver.get(url); accept_cookies_if_present(driver)
    try:
        wait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="matchStatsContainer"]')))
    except Exception:
        return None
    home, away = {}, {}
    containers = driver.find_elements(By.CSS_SELECTOR, '[data-testid="matchStatsContainer"]')
    for cont in containers:
        for bar in cont.find_elements(By.CSS_SELECTOR, ".match-stats__table-row--bar"):
            vals = bar.find_elements(By.CSS_SELECTOR, ".match-stats__stat-percentage")
            if len(vals) >= 2:
                home["possession"] = clean(vals[0].text); away["possession"] = clean(vals[1].text)
        for row in cont.find_elements(By.CSS_SELECTOR, ".match-stats__table-row"):
            if "match-stats__table-row--bar" in row.get_attribute("class"): continue
            try: name = row.find_element(By.CSS_SELECTOR, ".match-stats__stat-name").text
            except Exception: continue
            key = re.sub(r"[^a-z0-9]+","_", name.lower()).strip("_")
            hv = row.find_element(By.CSS_SELECTOR, ".match-stats__table-cell--home").text if row.find_elements(By.CSS_SELECTOR, ".match-stats__table-cell--home") else ""
            av = row.find_element(By.CSS_SELECTOR, ".match-stats__table-cell--away").text if row.find_elements(By.CSS_SELECTOR, ".match-stats__table-cell--away") else ""
            home[key] = clean(hv); away[key] = clean(av)
    out = {f"home_{k}": v for k, v in home.items()}
    out.update({f"away_{k}": v for k, v in away.items()})
    out["stats_url"] = url
    return out

# ---------- writing ----------
SEED_COLS = [
    "season","matchweek","fixture","home_team","away_team",
    "home_formation","home_xi","home_subs_used",
    "away_formation","away_xi","away_subs_used",
    "home_possession","away_possession","home_xg","away_xg","home_total_shots","away_total_shots",
    "home_shots_on_target","away_shots_on_target","home_corners","away_corners","home_saves","away_saves",
    "home_shots_inside_the_box","away_shots_inside_the_box","home_hit_woodwork","away_hit_woodwork",
    "home_big_chances_created","away_big_chances_created","home_total_crosses_completed","away_total_crosses_completed",
    "home_shots_off_target","away_shots_off_target","home_shots_outside_the_box","away_shots_outside_the_box",
    "home_total_passes","away_total_passes","home_long_passes_completed","away_long_passes_completed",
    "home_through_balls","away_through_balls","home_touches","away_touches",
    "home_touches_in_the_opposition_box","away_touches_in_the_opposition_box",
    "home_tackles_won","away_tackles_won","home_blocks","away_blocks","home_interceptions","away_interceptions",
    "home_clearances","away_clearances",
    "home_total_dribbles","away_total_dribbles","home_successful_dribbles","away_successful_dribbles",
    "home_duels_won","away_duels_won","home_aerial_duels_won","away_aerial_duels_won","home_distance_covered","away_distance_covered",
    "home_red_cards","away_red_cards","home_yellow_cards","away_yellow_cards","home_fouls","away_fouls",
    "home_offsides","away_offsides",
    "poll_pct_home","poll_pct_draw","poll_pct_away","poll_votes",
    "odds_home","odds_draw","odds_away",
    "recap_url","stats_url","recap_text","motm","motm_position",
    "recap_sentiment_compound","recap_sentiment_pos","recap_sentiment_neg","recap_sentiment_neu",
    "fixture_url"
]
def load_processed_ids(out_csv: str) -> set[str]:
    p = Path(out_csv)
    if not p.exists(): return set()
    try:
        df = pd.read_csv(out_csv, usecols=["fixture_url"])
        return {parse_match_id(u) for u in df["fixture_url"].dropna().astype(str).tolist()}
    except Exception:
        return set()
def append_row(out_csv: str, row: dict):
    p = Path(out_csv)
    if not p.exists():
        cols = list(dict.fromkeys(SEED_COLS + list(row.keys())))
        pd.DataFrame([row]).reindex(columns=cols).to_csv(out_csv, index=False, encoding="utf-8-sig")
    else:
        cols = pd.read_csv(out_csv, nrows=0).columns.tolist()
        pd.DataFrame([row]).reindex(columns=cols).to_csv(out_csv, mode="a", header=False, index=False, encoding="utf-8-sig")

# ---------- main ----------
STOP_ON_FIRST_NOT_FT = True

def run(fixtures_csv: str, out_csv: str, season: int, start_md: int,
        headless: bool, md_col: Optional[str], home_col: Optional[str], away_col: Optional[str]):

    want_sentiment = False
    if HAVE_VADER:
        try:
            try: nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError: nltk.download("vader_lexicon")
            want_sentiment = True
        except Exception:
            want_sentiment = False

    plan_pairs = load_pairs_by_md(fixtures_csv, start_md, md_col, home_col, away_col)
    if not plan_pairs:
        print("No matchweeks found at/after start_md.", file=sys.stderr); return

    processed = load_processed_ids(out_csv)

    opts = FirefoxOptions()
    if headless: opts.add_argument("-headless")
    opts.set_preference("intl.accept_languages", "en-US,en")
    driver = webdriver.Firefox(options=opts)
    driver.set_page_load_timeout(45)

    try:
        for md in sorted(plan_pairs.keys()):
            expected = plan_pairs[md]
            print(f"[MD {md}] fixtures in CSV: {len(expected)}")

            url_map = get_links_and_status(driver, season, md, expected)
            scraped_this_md = 0
            stop_due_to_future = False

            for (home, away) in expected:
                key = (canon_team(home), canon_team(away))
                if key not in url_map:
                    print(f"  - MISSING on site: {home} vs {away}")
                    continue

                murl, is_ft = url_map[key]
                mid = parse_match_id(murl)

                if mid in processed:
                    print(f"  - [skip existing] {home} vs {away}")
                    continue

                if STOP_ON_FIRST_NOT_FT and not is_ft:
                    print(f"  - [STOP] Encountered NOT-FT fixture: {home} vs {away}")
                    stop_due_to_future = True
                    break

                print(f"  - {home} vs {away}  ->  {murl}")
                row = {"season": season, "matchweek": md, "fixture": f"{home} vs {away}",
                       "home_team": home, "away_team": away, "fixture_url": murl}

                try:
                    lin  = scrape_lineups(driver, murl)
                    stats= scrape_stats(driver,  murl)
                    rec  = scrape_recap(driver,  murl, want_sentiment)
                except Exception as e:
                    print("    [warn] fixture error:", e)
                    continue

                if lin is None and stats is None:
                    print("    [skip] no lineups/stats yet.")
                    continue

                if lin:   row.update(lin)
                if stats: row.update(stats)
                if rec:   row.update(rec)

                append_row(out_csv, row)
                processed.add(mid)
                scraped_this_md += 1
                time.sleep(0.35)

            if stop_due_to_future:
                print(f"[STOP] Halting at first NOT-FT fixture in MD {md}.")
                break

            if scraped_this_md == 0:
                print(f"[CONTINUE] MD {md} had nothing new to scrape (already in CSV). Moving on…")
            else:
                print(f"[OK] MD {md}: scraped {scraped_this_md} fixture(s).")

    finally:
        driver.quit()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PL Lineups+Stats+Recap (+crowd poll & odds).")
    ap.add_argument("--fixtures_csv", default="fixtures_2025_26.csv")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--start_md", type=int, default=1)
    ap.add_argument("--out", default="pl_md1_until_blank.csv")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--md-col", default=None)
    ap.add_argument("--home-col", default=None)
    ap.add_argument("--away-col", default=None)
    args = ap.parse_args()
    run(args.fixtures_csv, args.out, args.season, args.start_md,
        args.headless, args.md_col, args.home_col, args.away_col)