# pl_stats_firefox.py
# Scrapes https://www.premierleague.com/en/stats using Firefox WebDriver (Selenium 4).
# Produces three files in the working directory:
#   1) pl_club_stats.csv                 -> team_name,goals,tacklesWon,blocks,totalPasses
#   2) pl_player_stats_players.csv       -> metric,rank,player,team_name,value
#   3) pl_player_stats_team.csv          -> team_name,p_goals,p_assists,p_totalPasses,p_cleanSheets
# Notes:
# - No PL APIs required. This reads the rendered DOM (supports cookie wall).
# - If headless struggles with the cookie modal, run once with HEADLESS=false, accept, then headless again.

from __future__ import annotations

import os
import csv
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ------------------ Optional .env ------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

HEADLESS        = os.getenv("HEADLESS", "true").lower() in {"1","true","yes","y"}
FIREFOX_BIN     = os.getenv("FIREFOX_BIN")          # optional
GECKODRIVER     = os.getenv("GECKODRIVER_PATH")     # optional absolute path
STATS_URL       = "https://www.premierleague.com/en/stats"
WAIT_SECS       = float(os.getenv("SELENIUM_WAIT_SECS", "25"))
SCROLL_STEPS    = int(os.getenv("SELENIUM_SCROLL_STEPS", "12"))
SCROLL_PAUSE    = float(os.getenv("SELENIUM_SCROLL_PAUSE", "0.75"))

# ------------------ Selenium 4 ---------------------
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ------------------ Normalization ------------------
TEAM_ALIAS: Dict[str, str] = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Man Utd": "Man United",
    "Tottenham Hotspur": "Tottenham",
    "Spurs": "Tottenham",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
    "West Ham United": "West Ham",
    "Liverpool FC": "Liverpool",
    "Liverpool": "Liverpool",
    "Leeds United": "Leeds",
    "Newcastle United": "Newcastle",
    "AFC Bournemouth": "Bournemouth",
    "Bournemouth": "Bournemouth",
    "Aston Villa": "Aston Villa",
    "Chelsea": "Chelsea",
    "Arsenal": "Arsenal",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Brentford": "Brentford",
    "Burnley": "Burnley",
    "Sunderland": "Sunderland",
    "Ipswich Town": "Ipswich",
    "Southampton": "Southampton",
}

# The exact widget headers we want (clubs vs players)
CLUB_METRIC_TITLES   = {"Goals": "goals", "Tackles Won": "tacklesWon", "Blocks": "blocks", "Total Passes": "totalPasses"}
PLAYER_METRIC_TITLES = {"Goals": "p_goals", "Assists": "p_assists", "Total Passes": "p_totalPasses", "Clean Sheets": "p_cleanSheets"}

@dataclass
class ClubRow:
    metric: str     # goals | tacklesWon | blocks | totalPasses
    team: str
    value: float

@dataclass
class PlayerRow:
    metric: str     # p_goals | p_assists | p_totalPasses | p_cleanSheets
    rank: int
    player: str
    team: str
    value: float

# ------------------ WebDriver ----------------------
def build_driver() -> webdriver.Firefox:
    opts = FirefoxOptions()
    opts.headless = HEADLESS
    if FIREFOX_BIN:
        opts.binary_location = FIREFOX_BIN

    # Set preferences directly on options (Selenium 4)
    opts.set_preference("permissions.default.image", 1)        # allow images (we use badge alt & logos)
    opts.set_preference("dom.webnotifications.enabled", False)
    opts.set_preference("privacy.trackingprotection.enabled", False)
    opts.set_preference("intl.accept_languages", "en-US,en;q=0.9")

    service = FirefoxService(executable_path=GECKODRIVER) if GECKODRIVER else FirefoxService()
    driver = webdriver.Firefox(options=opts, service=service)
    driver.set_page_load_timeout(60)
    return driver

def accept_cookies(driver: webdriver.Firefox) -> None:
    """
    Press the cookie 'Accept All' button if present. Try multiple labels.
    """
    labels = [
        "Accept All Cookies", "Accept all cookies", "Accept All", "Accept all",
        "I Accept", "I accept", "Allow all cookies", "Agree"
    ]
    try:
        # Wait briefly for modal
        time.sleep(0.7)
        for text in labels:
            try:
                btn = driver.find_element(By.XPATH, f"//button[normalize-space()='{text}' or contains(., '{text}')]")
                # Scroll into view and click
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                ActionChains(driver).move_to_element(btn).pause(0.2).click(btn).perform()
                time.sleep(0.6)
                return
            except Exception:
                continue
        # Some variants use role=button inside divs
        try:
            btn = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[role='button']"))
            )
            btn.click()
            time.sleep(0.5)
        except Exception:
            pass
    except Exception:
        pass

def scroll_page(driver: webdriver.Firefox, steps: int = SCROLL_STEPS, pause: float = SCROLL_PAUSE) -> None:
    last_h = 0
    for _ in range(steps):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        h = driver.execute_script("return document.body.scrollHeight;")
        if h == last_h:
            break
        last_h = h

# ------------------ Parsing helpers ----------------
def safe_text(el, css: str) -> str:
    try:
        return el.find_element(By.CSS_SELECTOR, css).get_attribute("textContent").strip()
    except Exception:
        return ""

def badge_alt_team(el) -> str:
    """
    Returns team name from badge alt like "Chelsea club badge".
    """
    try:
        img = el.find_element(By.CSS_SELECTOR, "[data-testid='clubBadge'] img")
        alt = (img.get_attribute("alt") or "").replace("club badge", "").strip()
        return alt
    except Exception:
        return ""

def is_player_block(article_el) -> bool:
    """
    Decide if an article leaderboard is a PLAYER list:
    - rows contain an .avatar--player element,
    or header link contains '/top/players'
    """
    try:
        link = article_el.find_element(By.CSS_SELECTOR, "a.stats-leaderboard__header-link").get_attribute("href") or ""
        if "/top/players" in link:
            return True
    except Exception:
        pass
    # Fallback: inspect first li for player avatar
    try:
        li = article_el.find_element(By.CSS_SELECTOR, "ul.stats-leaderboard__leaderboard > li")
        li.find_element(By.CSS_SELECTOR, ".avatar--player")
        return True
    except Exception:
        return False

# ------------------ Scrape -------------------------
def scrape_stats(driver: webdriver.Firefox) -> Tuple[List[ClubRow], List[PlayerRow]]:
    driver.get(STATS_URL)
    accept_cookies(driver)
    WebDriverWait(driver, WAIT_SECS).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "article.stats-leaderboard"))
    )
    scroll_page(driver)

    club_rows: List[ClubRow] = []
    player_rows: List[PlayerRow] = []

    articles = driver.find_elements(By.CSS_SELECTOR, "article.stats-leaderboard")
    for art in articles:
        try:
            header = safe_text(art, ".stats-leaderboard__header")
            header = " ".join(header.split())
            lis = art.find_elements(By.CSS_SELECTOR, "ul.stats-leaderboard__leaderboard > li")
        except Exception:
            continue

        # CLUB leaderboards
        if header in CLUB_METRIC_TITLES and not is_player_block(art):
            key = CLUB_METRIC_TITLES[header]
            for li in lis:
                team = safe_text(li, ".stats-leaderboard__name")
                if not team:
                    team = badge_alt_team(li)
                val_text = safe_text(li, ".stats-leaderboard__stat-value").replace(",", "")
                try:
                    value = float(val_text)
                except Exception:
                    value = 0.0
                team = TEAM_ALIAS.get(team, team)
                if team:
                    club_rows.append(ClubRow(key, team, value))
            continue

        # PLAYER leaderboards
        if header in PLAYER_METRIC_TITLES and is_player_block(art):
            key = PLAYER_METRIC_TITLES[header]
            for li in lis:
                rank_txt = safe_text(li, ".stats-leaderboard__pos")
                try:
                    rank = int(rank_txt)
                except Exception:
                    rank = 0
                player = safe_text(li, ".stats-leaderboard__name")
                # Prefer explicit 'team-name' span; fallback to badge alt
                team_name = safe_text(li, ".stats-leaderboard__team-name")
                if not team_name:
                    team_name = badge_alt_team(li)
                val_text = safe_text(li, ".stats-leaderboard__stat-value").replace(",", "")
                try:
                    value = float(val_text)
                except Exception:
                    value = 0.0
                team = TEAM_ALIAS.get(team_name, team_name)
                player_rows.append(PlayerRow(key, rank, player, team, value))
            continue

        # Ignore anything else (other widgets)

    return club_rows, player_rows

# ------------------ Aggregation & Output ----------------
def aggregate_club(rows: List[ClubRow]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        out.setdefault(r.team, {})
        out[r.team][r.metric] = out[r.team].get(r.metric, 0.0) + float(r.value)
    return out

def aggregate_player_to_team(rows: List[PlayerRow]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        team = r.team.strip()
        if not team:
            continue
        out.setdefault(team, {})
        out[team][r.metric] = out[team].get(r.metric, 0.0) + float(r.value)
    return out

def write_club_csv(table: Dict[str, Dict[str, float]], path="pl_club_stats.csv") -> None:
    cols = ["team_name","goals","tacklesWon","blocks","totalPasses"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for team in sorted(table.keys()):
            m = table[team]
            w.writerow([
                team,
                float(m.get("goals",0.0)),
                float(m.get("tacklesWon",0.0)),
                float(m.get("blocks",0.0)),
                float(m.get("totalPasses",0.0)),
            ])
    print(f"[OUT] {path} ({len(table)} teams)")

def write_player_long_csv(rows: List[PlayerRow], path="pl_player_stats_players.csv") -> None:
    cols = ["metric","rank","player","team_name","value"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r.metric, r.rank, r.player, r.team, r.value])
    print(f"[OUT] {path} ({len(rows)} rows)")

def write_player_team_csv(table: Dict[str, Dict[str, float]], path="pl_player_stats_team.csv") -> None:
    cols = ["team_name","p_goals","p_assists","p_totalPasses","p_cleanSheets"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for team in sorted(table.keys()):
            m = table[team]
            w.writerow([
                team,
                float(m.get("p_goals",0.0)),
                float(m.get("p_assists",0.0)),
                float(m.get("p_totalPasses",0.0)),
                float(m.get("p_cleanSheets",0.0)),
            ])
    print(f"[OUT] {path} ({len(table)} teams)")

# ------------------ Main ----------------------------
def main():
    print(f"[INFO] Launching Firefox (headless={HEADLESS}) …")
    driver = build_driver()
    try:
        driver.get(STATS_URL)
        accept_cookies(driver)
        WebDriverWait(driver, WAIT_SECS).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article.stats-leaderboard"))
        )
        scroll_page(driver)

        club_rows, player_rows = scrape_stats(driver)
        if not club_rows and not player_rows:
            # Helpful debug: take a screenshot to inspect the DOM state
            try:
                driver.save_screenshot("pl_stats_debug.png")
                print("[WARN] No rows parsed. Saved screenshot: pl_stats_debug.png")
            except Exception:
                pass

        # Aggregate + write
        club_table = aggregate_club(club_rows)
        player_team_table = aggregate_player_to_team(player_rows)

        write_club_csv(club_table, "pl_club_stats.csv")
        write_player_long_csv(player_rows, "pl_player_stats_players.csv")
        write_player_team_csv(player_team_table, "pl_player_stats_team.csv")

    finally:
        driver.quit()
        print("[INFO] Firefox closed.")

if __name__ == "__main__":
    main()
