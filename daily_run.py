import pandas as pd
import numpy as np
import requests
import json
from lxml import etree, html
import joblib
from datetime import *
htmlparser = etree.HTMLParser()

import data_grab.data_grab as data_grab, scripts.data_proc as data_proc, scripts.data_prep as data_prep, scripts.data_explor as data_explor



team_map = {'ANH':'ANA', 
            'ARI':'ARI', 
            'BOS':'BOS', 
            'BUF': 'BUF', 
            'CAR':'CAR', 
            'CGY':'CGY', 
            'CHI':'CHI', 
            'CLS': 'CBJ', 
            'COL':'COL', 
            'DAL':'DAL', 
            'DET':'DET', 
            'EDM':'EDM', 
            'FLA':'FLA', 
            'LA':'LAK', 
            'MIN':'MIN', 
            'MON': 'MTL', 
            'NJ':'NJD', 
            'NSH':'NSH', 
            'NYI':' NYI', 
            'NYR': 'NYR', 
            'OTT':'OTT', 
            'PHI':'PHI', 
            'PIT':'PIT', 
            'SEA':'SEA', 
            'SJ': 'SJS', 
            'STL': 'STL', 
            'TB':'TBL', 
            'TOR':'TOR', 
            'VAN':'VAN', 
            'VGK':'VGK', 
            'WAS':'WSH', 
            'WPG':'WPG'}

def daily_lineups(daily_url, away_teams_xpath, home_teams_xpath, team_map):
    
    daily_results = requests.get(daily_url)
    daily_results_tree = html.fromstring(daily_results.content)
    
    away_teams = daily_results_tree.xpath(away_teams_xpath)
    home_teams = daily_results.xpath(home_teams_xpath)
    
    away_teams = [team_map[x] for x in away_teams if x in team_map]
    home_teams = [team_map[x] for x in home_teams if x in team_map]
    
    return away_teams, home_teams


def todays_data(df_merged, boundary, home_teams, away_teams, today, games_dict_away, games_dict_home):
    
    today_home_df = df_merged[(df_merged['gameDate'] > boundary) & (df_merged['teamAbbrev'].isin(home_teams))]
    
    today_away_df = df_merged[(df_merged['gameDate'] > boundary) & (df_merged['teamAbbrev'].isin(away_teams))]
    
    today_home_df['gameDate'] = today
    today_away_df['gameDate'] = today
    
    today_home_df['homeRoad'] = 'H'
    today_away_df['homeRoad'] = 'R'
    
    today_home_df[['gamesPlayed', 'goals', 'evTimeOnIce', 'evTimeOnIcePerGame',
                   'otTimeOnIce', 'otTimeOnIcePerOtGame', 'goalsBackhand',
                   'goalsDeflected', 'goalsSlap', 'goalsSnap',
                   'goalsTipIn', 'ppTimeOnIce','ppTimeOnIcePerGame',
                   'shTimeOnIce', 'shTimeOnIcePerGame', 'shifts',
                   'shiftsPerGame', 'goalsWrapAround',
                   'goalsWrist','shootingPct', 'shots', 'shotsOnNetBackhand',
                   'shotsOnNetDeflected', 'shotsOnNetSlap', 'shotsOnNetSnap',
                   'shotsOnNetTipIn', 'shotsOnNetWrapAround', 
                   'shotsOnNetWrist','assists', 'evGoals', 'evPoints',
                   'gameWinningGoals', 'otGoals', 'penaltyMinutes', 'plusMinus',
                   'points','pointsPerGame', 'positionCode', 'ppGoals', 
                   'ppPoints', 'shGoals','shPoints', 'timeOnIcePerGame',
                   'blockedShots','blockedShotsPer60', 'emptyNetGoals',
                   'firstGoals', 'giveaways','giveawaysPer60', 'hits',
                   'hitsPer60', 'missedShotCrossbar','missedShotGoalpost',
                   'missedShotOverNet', 'missedShotWideOfNet',
                   'missedShots', 'takeaways', 'takeawaysPer60']] = 0
    
    today_away_df[['gamesPlayed', 'goals', 'evTimeOnIce', 'evTimeOnIcePerGame',
                   'otTimeOnIce', 'otTimeOnIcePerOtGame', 'goalsBackhand',
                   'goalsDeflected', 'goalsSlap', 'goalsSnap',
                   'goalsTipIn', 'ppTimeOnIce','ppTimeOnIcePerGame',
                   'shTimeOnIce', 'shTimeOnIcePerGame', 'shifts',
                   'shiftsPerGame', 'goalsWrapAround',
                   'goalsWrist','shootingPct', 'shots', 'shotsOnNetBackhand',
                   'shotsOnNetDeflected', 'shotsOnNetSlap', 'shotsOnNetSnap',
                   'shotsOnNetTipIn', 'shotsOnNetWrapAround', 
                   'shotsOnNetWrist','assists', 'evGoals', 'evPoints',
                   'gameWinningGoals', 'otGoals', 'penaltyMinutes', 'plusMinus',
                   'points','pointsPerGame', 'positionCode', 'ppGoals', 
                   'ppPoints', 'shGoals','shPoints', 'timeOnIcePerGame',
                   'blockedShots','blockedShotsPer60', 'emptyNetGoals',
                   'firstGoals', 'giveaways','giveawaysPer60', 'hits',
                   'hitsPer60', 'missedShotCrossbar','missedShotGoalpost',
                   'missedShotOverNet', 'missedShotWideOfNet',
                   'missedShots', 'takeaways', 'takeawaysPer60']] = 0
    
    today_away_df['opponentTeamAbbrev'] = today_away_df['teamAbbrev'].map(games_dict_away)
    today_home_df['opponentTeamAbbrev'] = today_home_df['opponentTeamAbbrev'].map(games_dict_home)
    
    today_df = pd.concat([today_away_df, today_home_df])
    
    df_merged = pd.concat([df_merged, today_df])
    
    return df_merged
