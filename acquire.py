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

REPOS = ['/kyleskom/NBA-Machine-Learning-Sports-Betting',
 '/pretrehr/Sports-betting',
 '/llSourcell/ChatGPT_Sports_Betting_Bot',
 '/georgedouzas/sports-betting',
 '/sedemmler/WagerBrain',
 '/openbookie/sportbook',
 '/Seb943/scrapeOP',
 '/cvidan/bet365-scraper',
 '/llSourcell/sports_betting_with_reinforcement_learning',
 '/jd5688/online-sports-betting',
 '/ScrapeWithYuri/Live-Sports-Arbitrage-Bet-Finder',
 '/ryankrumenacker/sports-betting-arbitrage-project',
 '/charlesmalafosse/sports-betting-customloss',
 '/JDaniloC/BetBot',
 '/jkrusina/SoccerPredictor',
 '/papagorgio23/bettoR',
 '/dashee87/betScrapeR',
 '/Phisteven/scraping-bets',
 '/rockscripts/Sport-Betting-APP-Betfair-Market',
 '/jbouzekri/free-bet',
 '/ethbets/ebets',
 '/Shymoney/Sports-betting-web-app',
 '/Jan17392/asianodds',
 '/jrkosinski/crypto-champ',
 '/ws2516/manat',
 '/OriginSport/bet-center',
 '/danderfer/Comp_Sci_Sem_2',
 '/addtek/reactnative_sports_betting_app',
 '/manuelsc/eSportsETH',
 '/mautomic/NitrogenSports-Analysis',
 '/acehood0126/ulti-bets-main-app',
 '/seanpquig/betting-odds-analyzer',
 '/OryJonay/Odds-Gym',
 '/angelle-sw/sbp',
 '/S1M0N38/aao',
 '/day-mon/sports-betting-ai',
 '/umitkaanusta/MacTahminBotu',
 '/profjordanov/sports-system',
 '/thespread/api',
 '/noecorp/sports-betting-site',
 '/andrew-couch/UFC-Sports-Betting-Model',
 '/deepbodra97/ethereum-sports-betting',
 '/tomhaydn/BetArbit',
 '/peerplays-network/bookiesports',
 '/marcoblume/odds.converter',
 '/Ryczko/KKbets-betting',
 '/bendominguez0111/nba-models',
 '/zaltaie/SportsBettingArbitrage',
 '/pwu97/bettingtools',
 '/MULERx/Sports-betting-web-app',
 '/bakedziti88/sportsbook-api',
 '/wagerlab/model-aggregator',
 '/Cloudbet/docs',
 '/jvanderhoof/Sports-Arbitrage-Parser',
 '/kheller18/sportsbook-4',
 '/garfjohnson/Nba-Sports-Betting-Model',
 '/defifarmer265/BetApp',
 '/jvanderhoof/Sports-Arbitrage-Website',
 '/mipes4/sports-betting-client',
 '/daankoning/ArbitrageFinder',
 '/jojubart/basketball-betting-bot',
 '/andrewtryder/Odds',
 '/anthonyjzhang/fouralpha',
 '/valenIndovino/apuestas-deportivas',
 '/vitaliy-kuzmich/bets',
 '/yssefunc/sport_analytics',
 '/vnguyen5/MLB-Machine-Learning-Sports-Betting',
 '/AkashK23/SportsBettingWebsite',
 '/Swati-Subhadarshini/UWFinTech_Project3',
 '/incredigroup/cryptobetting_sports',
 '/matthewmics/sports-esports-betting-system',
 '/EttelasK/NFL_ConfidencePool',
 '/vietgamingnetwork/iBetting',
 '/focus1691/sports-betting-calculators',
 '/peanutshawny/nfl-sports-betting',
 '/sportsdataverse/oddsapiR',
 '/thebananablender/arbitrage-finder',
 '/WilliamMcRoberts/BetBookApp.BlazorServer',
 '/jbmenashi/Betski-Frontend',
 '/dolbyio-samples/stream-app-web-viewer',
 '/S4ltster/Beemovie',
 '/geraldpro/richstakers',
 '/am-523/Sports-Betting-Dashboard',
 '/Nikolamv95/MySportTips',
 '/gonzalezlrjesus/API-Betting-Sports',
 '/ONESOFT-OS/BetBall',
 '/tsinghqs/SportsBettings',
 '/gingeleski/soccer-draws-bettor',
 '/NJCinnamond/sports-betting-dapp',
 '/maxymkuz/Sports-predictor',
 '/cloudzombie/bitbettings',
 '/efreesen/sports_betting_engine',
 '/mipes4/sportsbetting_fe',
 '/zporsdata/SportsDataFeedAPI',
 '/stuartread7/Unibet-Scraper',
 '/callmevojtko/Recommended-Bets-By-Email-MLB',
 '/scibrokes/dixon-coles1996',
 '/lmiller1990/gamblor-web',
 '/Andrewlastrapes/pokerBackend',
 '/graphprotocol/sportx-subgraph',
 '/24juice22/cam-sportsbook',
 '/mschoenhart/rbot',
 '/blarth/duel-me-esports-front',
 '/jeremyzhang1/peer-sports-betting',
 '/denp1/bfjs']
 

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
    url = f"https://api.github.com/repos{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        return repo_info.get("language", None)
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos{repo}/contents/"
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

