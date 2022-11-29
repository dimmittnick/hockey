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



def cal_cols(df_merged):
    
    df_merged['fanPoints'] = data_explor.fan_points(df_merged)
    df_merged['overPerform'] = data_explor.overperform(df_merged, 'fanPoints', 'playerId')
    df_merged['overPerformDummy'] = data_explor.over_perf_dummy(df_merged, 'overPerform')
    df_merged['unerPerformDummy'] = data_explor.under_perf_dummy(df_merged, 'overPerform')
    df_merged['homeRoadPerf'] = data_explor.home_away_perf(df_merged, 'overPerform', ['playerId', 'homeRoad'])
    
    return df_merged



def home_road_skate_split(df_merged):
    
    better_home_skater = list(np.where((df_merged['homeRoad'] == 'H') & (df_merged['homeRoadPerf'] > 0), df_merged['playerId'], None))
    better_away_skater = list(np.where((df_merged['homeRoad'] == 'R') & (df_merged['homeRoadPerf'] > 0), df_merged['playerId'], None))
    better_home_skater = [*set(better_home_skater)]
    better_away_skater = [*set(better_away_skater)]
    
    df_merged['OpHomeDummy'] = np.where(df_merged['playerId'].isin(better_home_skater), 1, 0)
    df_merged['OpAwayDummy'] = np.where(df_merged['playerId'].isin(better_away_skater), 1, 0)
    
    df_merged['OpNowhereDummy'] = np.where((df_merged['OpHomeDummy'] == 0) & (df_merged['OpRoadDummy'] == 0), 1, 0)
    
    

def feature_creation(df_merged, feature_list, goals, shots):
    for feature in feature_list:
        df_merged[f'{feature}Ma7'] = data_proc.moving_average(df_merged, feature, 'playerId', 7)
        df_merged[f'{feature}Ma7'] = df_merged[f'{feature}Ma7'].shift(1)
    
    for feature in feature_list:
        df_merged[f'{feature}Ma3'] = data_proc.moving_average(df_merged, feature, 'playerId', 3)
        df_merged[f'{feature}Ma3'] = df_merged[f'{feature}Ma3'].shift(1)
    
    for feature in feature_list:
        df_merged[f'{feature}LastGame'] = df_merged[feature].shift(1)
    
    for feature in feature_list:
        df_merged[f'{feature}Ma10'] = data_proc.moving_average(df_merged, feature, 'playerId', 10)
        df_merged[f'{feature}Ma10'] = df_merged[f'{feature}Ma10'].shift(1)

    for feature in feature_list:
        df_merged[f'{feature}Ma14'] = data_proc.moving_average(df_merged, feature, 'playerId', 14)
        df_merged[f'{feature}Ma14'] = df_merged[f'{feature}Ma14'].shift(1)
    
    for goal in goals:
        df_merged[f"%{goal}"] = data_proc.percShotType(df_merged, 'playerId', goal, 'goals')

    for shot in shots:
        df_merged[f"%{shot}"] = data_proc.percShotType(df_merged, 'playerId', shot, 'shots')
        
    return df_merged



def cleaning(drop_cols, impute_by_player, impute_by_perf, drop_cols2):
    
    df_merged = data_prep.remove_columns(df_merged, drop_cols)
    
    for col in impute_by_player:
        data_prep.handle_missing(df_merged, 'playerId', col)
        
    for col in impute_by_perf:
        
        try:
            data_prep.handle_missing(df_merged, 'overPerformDummy', col)
        except:
            continue
            
    df_merged = data_prep.remove_columns(df_merged, drop_cols2)
    
    return df_merged
    