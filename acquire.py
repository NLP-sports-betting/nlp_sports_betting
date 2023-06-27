"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

REPOS = ['https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting',
 'https://github.com/pretrehr/Sports-betting',
 'https://github.com/llSourcell/ChatGPT_Sports_Betting_Bot',
 'https://github.com/georgedouzas/sports-betting',
 'https://github.com/sedemmler/WagerBrain',
 'https://github.com/openbookie/sportbook',
 'https://github.com/Seb943/scrapeOP',
 'https://github.com/cvidan/bet365-scraper',
 'https://github.com/llSourcell/sports_betting_with_reinforcement_learning',
 'https://github.com/jd5688/online-sports-betting',
 'https://github.com/ScrapeWithYuri/Live-Sports-Arbitrage-Bet-Finder',
 'https://github.com/ryankrumenacker/sports-betting-arbitrage-project',
 'https://github.com/charlesmalafosse/sports-betting-customloss',
 'https://github.com/JDaniloC/BetBot',
 'https://github.com/jkrusina/SoccerPredictor',
 'https://github.com/papagorgio23/bettoR',
 'https://github.com/dashee87/betScrapeR',
 'https://github.com/Phisteven/scraping-bets',
 'https://github.com/rockscripts/Sport-Betting-APP-Betfair-Market',
 'https://github.com/jbouzekri/free-bet',
 'https://github.com/MarkipTheMudkip/in-class-project-2',
 'https://github.com/ethbets/ebets',
 'https://github.com/Shymoney/Sports-betting-web-app',
 'https://github.com/Jan17392/asianodds',
 'https://github.com/jrkosinski/crypto-champ',
 'https://github.com/ws2516/manat',
 'https://github.com/OriginSport/bet-center',
 'https://github.com/danderfer/Comp_Sci_Sem_2',
 'https://github.com/addtek/reactnative_sports_betting_app',
 'https://github.com/manuelsc/eSportsETH',
 'https://github.com/mautomic/NitrogenSports-Analysis',
 'https://github.com/acehood0126/ulti-bets-main-app',
 'https://github.com/seanpquig/betting-odds-analyzer',
 'https://github.com/OryJonay/Odds-Gym',
 'https://github.com/angelle-sw/sbp',
 'https://github.com/S1M0N38/aao',
 'https://github.com/day-mon/sports-betting-ai',
 'https://github.com/umitkaanusta/MacTahminBotu',
 'https://github.com/profjordanov/sports-system',
 'https://github.com/thespread/api',
 'https://github.com/noecorp/sports-betting-site',
 'https://github.com/andrew-couch/UFC-Sports-Betting-Model',
 'https://github.com/deepbodra97/ethereum-sports-betting',
 'https://github.com/tomhaydn/BetArbit',
 'https://github.com/k26dr/peerbet',
 'https://github.com/peerplays-network/bookiesports',
 'https://github.com/marcoblume/odds.converter',
 'https://github.com/Ryczko/KKbets-betting',
 'https://github.com/bendominguez0111/nba-models',
 'https://github.com/rsmile310/sports_betting',
 'https://github.com/zaltaie/SportsBettingArbitrage',
 'https://github.com/NaterTots/SportsBetting',
 'https://github.com/pwu97/bettingtools',
 'https://github.com/MULERx/Sports-betting-web-app',
 'https://github.com/bakedziti88/sportsbook-api',
 'https://github.com/wagerlab/model-aggregator',
 'https://github.com/Cloudbet/docs',
 'https://github.com/alishbaimran/Ethereum-Sports-Betting-DApp',
 'https://github.com/jvanderhoof/Sports-Arbitrage-Parser',
 'https://github.com/kheller18/sportsbook-4',
 'https://github.com/garfjohnson/Nba-Sports-Betting-Model',
 'https://github.com/defifarmer265/BetApp',
 'https://github.com/jvanderhoof/Sports-Arbitrage-Website',
 'https://github.com/mipes4/sports-betting-client',
 'https://github.com/daankoning/ArbitrageFinder',
 'https://github.com/jojubart/basketball-betting-bot',
 'https://github.com/andrewtryder/Odds',
 'https://github.com/anthonyjzhang/fouralpha',
 'https://github.com/valenIndovino/apuestas-deportivas',
 'https://github.com/vitaliy-kuzmich/bets',
 'https://github.com/yssefunc/sport_analytics',
 'https://github.com/vnguyen5/MLB-Machine-Learning-Sports-Betting',
 'https://github.com/AkashK23/SportsBettingWebsite',
 'https://github.com/Swati-Subhadarshini/UWFinTech_Project3',
 'https://github.com/incredigroup/cryptobetting_sports',
 'https://github.com/matthewmics/sports-esports-betting-system',
 'https://github.com/EttelasK/NFL_ConfidencePool',
 'https://github.com/vietgamingnetwork/iBetting',
 'https://github.com/focus1691/sports-betting-calculators',
 'https://github.com/npally/sports-betting',
 'https://github.com/peanutshawny/nfl-sports-betting',
 'https://github.com/sportsdataverse/oddsapiR',
 'https://github.com/thebananablender/arbitrage-finder',
 'https://github.com/chogan72/BasketballBettingModel',
 'https://github.com/WilliamMcRoberts/BetBookApp.BlazorServer',
 'https://github.com/jbmenashi/Betski-Frontend',
 'https://github.com/dolbyio-samples/stream-app-web-viewer',
 'https://github.com/S4ltster/Beemovie',
 'https://github.com/geraldpro/richstakers',
 'https://github.com/am-523/Sports-Betting-Dashboard',
 'https://github.com/Nikolamv95/MySportTips',
 'https://github.com/gonzalezlrjesus/API-Betting-Sports',
 'https://github.com/ONESOFT-OS/BetBall',
 'https://github.com/tsinghqs/SportsBettings',
 'https://github.com/gingeleski/soccer-draws-bettor',
 'https://github.com/NJCinnamond/sports-betting-dapp',
 'https://github.com/maxymkuz/Sports-predictor',
 'https://github.com/cloudzombie/bitbettings',
 'https://github.com/efreesen/sports_betting_engine',
 'https://github.com/mipes4/sportsbetting_fe',
 'https://github.com/zporsdata/SportsDataFeedAPI',
 'https://github.com/stuartread7/Unibet-Scraper',
 'https://github.com/callmevojtko/Recommended-Bets-By-Email-MLB',
 'https://github.com/scibrokes/dixon-coles1996',
 'https://github.com/lmiller1990/gamblor-web',
 'https://github.com/Andrewlastrapes/pokerBackend',
 'https://github.com/graphprotocol/sportx-subgraph',
 'https://github.com/24juice22/cam-sportsbook',
 'https://github.com/catoenm/EtherSports',
 'https://github.com/mschoenhart/rbot',
 'https://github.com/blarth/duel-me-esports-front',
 'https://github.com/mredwardyun/betcoin',
 'https://github.com/fuad-ibrahimzade/sports-bettor',
 'https://github.com/MetGreg/sports_betting',
 'https://github.com/mipes4/sports-betting-server',
 'https://github.com/jeremyzhang1/peer-sports-betting',
 'https://github.com/denp1/bfjs',
 'https://github.com/mattyfew/smart_fantasy_sports',
 'https://github.com/TomMago/ScrapeBettingBot',
 'https://github.com/jimtje/sportsbookreview-gql']
 

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        return repo_info.get("language", None)
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_contents = requests.get(get_readme_download_url(contents)).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data2.json", "w"), indent=1)