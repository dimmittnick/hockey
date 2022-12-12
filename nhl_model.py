import pandas as pd
import numpy as np
import importlib
import os
import requests
import json
from lxml import etree, html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import joblib
from datetime import *
from dateutil import relativedelta
htmlparser = etree.HTMLParser()


nhl_teams = nhl_teams = 'ANA ANH ARI BOS BUF CAR CGY CHI CLS CBJ COL DAL DET EDM FLA LA LAK MIN MON MTL NJ NSH NYI NYR OTT PHI PIT SEA SJ SJS STL TB TBL TOR VAN VGK WAS WPG'.split()

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
            'NYI':'NYI', 
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

end_date = "2022-11-30"
yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
today = datetime.today().strftime("%Y-%m-%d")

def todays_games(url, home_xpath, road_xpath, home_goalies_xpath, road_goalies_xpath, nhl_teams, team_map):
    
    results = requests.get(url)
    tree = html.fromstring(results.content)
    h_teams = tree.xpath(home_xpath)
    r_teams = tree.xpath(road_xpath)
    
    home_teams = [team_map[x] for x in h_teams if x in nhl_teams]
    road_teams = [team_map[x] for x in r_teams if x in nhl_teams]
    
    home_goalies = tree.xpath(home_goalies_xpath)
    road_goalies = tree.xpath(road_goalies_xpath)
    
    games_road = [(x,y) for x,y in zip(road_teams, home_teams)]
    games_home = [(x,y) for x,y in zip(home_teams, road_teams)]

    games_dict_road = dict(games_road)
    games_dict_home = dict(games_home)
    
    
    goalie_dict = {}
    
    for x in range(len(home_goalies)):
        goalie_dict[road_teams[x]] = home_goalies[x]
        goalie_dict[home_teams[x]] = road_goalies[x]
        
    return road_teams, home_teams, games_dict_road, games_dict_home, goalie_dict


def webscrape(url, data_string):
    """
    takes in nhl.com/stats api url and returns data based off dates and page number
    """
    try:
        response = requests.get(url).text
        data = json.loads(response)
        
        return pd.DataFrame(data[data_string])
    
    except:
        return print(f'URL Error: {url}')
    

def url_gen(start_date, end_date, page, data_seg):
    
    skate_url = f'https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22goals%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22assists%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start={page}&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3C=%22{end_date}%2023%3A59%3A59%22%20and%20gameDate%3E=%22{start_date}%22%20and%20gameTypeId=2'
    
    
    skate_misc_url = f'https://api.nhle.com/stats/rest/en/skater/realtime?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22hits%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start={page}&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3C=%22{end_date}%2023%3A59%3A59%22%20and%20gameDate%3E=%22{start_date}%22%20and%20gameTypeId=2'                                
    
    skate_shots_url = f'https://api.nhle.com/stats/rest/en/skater/shottype?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22shots%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22shootingPct%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start={page}&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3C=%22{end_date}%2023%3A59%3A59%22%20and%20gameDate%3E=%22{start_date}%22%20and%20gameTypeId=2'
    
    skate_scor_60_url = f'https://api.nhle.com/stats/rest/en/skater/scoringpergame?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22pointsPerGame%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22goalsPerGame%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start={page}&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3C=%22{end_date}%2023%3A59%3A59%22%20and%20gameDate%3E=%22{start_date}%22%20and%20gameTypeId=2'
    
    skate_toi_url = f'https://api.nhle.com/stats/rest/en/skater/timeonice?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22timeOnIce%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start={page}&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3C=%22{end_date}%2023%3A59%3A59%22%20and%20gameDate%3E=%22{start_date}%22%20and%20gameTypeId=2'
    
    goal_url = f'https://api.nhle.com/stats/rest/en/goalie/summary?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22wins%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22savePct%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start={page}&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3C=%22{end_date}%2023%3A59%3A59%22%20and%20gameDate%3E=%22{start_date}%22%20and%20gameTypeId=2'
    
    team_url = f'https://api.nhle.com/stats/rest/en/team/summary?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22wins%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22teamId%22,%22direction%22:%22ASC%22%7D%5D&start={page}&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3C=%22{end_date}%2023%3A59%3A59%22%20and%20gameDate%3E=%22{start_date}%22%20and%20gameTypeId=2'
    
    if data_seg == 'skater':
        return skate_url
    
    elif data_seg == 'misc':
        return skate_misc_url
    
    elif data_seg == 'shots':
        return skate_shots_url
    
    elif data_seg == 'scor':
        return skate_scor_60_url
    
    elif data_seg == 'toi':
        return skate_toi_url
    
    elif data_seg == 'goal':
        return goal_url
    
    else:
        return team_url


def date_list_gen(start, end):
    '''takes in two dates and returns list of all dates spaced out by months inbetween the two dates
    '''
    
    start_date = datetime.strptime(start, '%Y-%m-%d').date()
    end_date = datetime.strptime(end, '%Y-%m-%d').date()
    current_date = start_date
    date_list = []
    
    while current_date > end_date:
        last_month = current_date - relativedelta.relativedelta(months=1)
        tup = (str(current_date), str(last_month))
        date_list.append(tup)
        current_date = last_month
    
    return date_list



def dg_main(start_date, end_date, data_seg, update_path, start, stop, step, update=False, saveData=False):
    """main function, list comprhension to create list of dataframes then flattens that list, concatentes all of them and returns the df
    """
    if update:
        df_old = pd.read_csv(update_path)
        
    date_list = date_list_gen(start_date, end_date)
    
    df_list = [[webscrape(url_gen(y[1], y[0], x, data_seg), 'data') for y in date_list] for x in range(start, stop, step)]
    
    df_list_flat = [df for sublist in df_list for df in sublist]

    df = pd.concat(df_list_flat)
    
    if update:
        df_update = pd.concat([df, df_old])
    
    if saveData:
        df_update.to_csv(f"/Users/nickdimmitt/hockey/data/df_{data_seg}.csv")
    
    if update:
        
        return df_update
    
    return df

def dropper(df, columns, axis):
    
    df1 = df.drop(columns, axis=axis)
    
    return df1

def merger(df1, df2, on):
    
    df_merged = pd.merge(df1, df2, on=on)
    
    return df_merged

def drop_dups(df, subset=None):
    
    if subset==None:
        return df.drop_duplicates(inplace=True)
    else:
        return df[subset].drop_duplicates(inplace=True)

def rename_cols_filter(df, columns):
    
    df = df[columns]
    df.columns = df.columns.str.rstrip('_x')
    
    return df

def todays_df(df_merged, date_col, cutoff_date, team_col, home_teams, road_teams, today, goalie_dict, columns, games_dict_road, games_dict_home, df_goalie):
    
    today_home_df = df_merged[(df_merged[date_col] > cutoff_date) & (df_merged[team_col].isin(home_teams))]
    today_road_df = df_merged[(df_merged[date_col] > cutoff_date) & (df_merged[team_col].isin(road_teams))]
    
    today_home_df['gameDate'] = today
    today_road_df['gameDate'] = today
    
    today_home_df['homeRoad'] = 'H'
    today_road_df['homeRoad'] = 'R'
    
    today_home_df[columns] = 0
    today_road_df[columns] = 0
    
    today_road_df['opponentTeamAbbrev'] = today_road_df[team_col].map(games_dict_road)
    today_home_df['opponentTeamAbbrev'] = today_home_df[team_col].map(games_dict_home)
    
    today_road_df['goalieFullName'] = today_road_df[team_col].map(goalie_dict)
    today_home_df['goalieFullName'] = today_home_df[team_col].map(goalie_dict)
    
    today_df = pd.concat([today_home_df, today_road_df])
    today_df.drop_duplicates(subset='playerId', inplace=True)
    
    goalies = list(df_goalie['goalieFullName'])
    goalieId = list(df_goalie['goalieId'])
    
    goalie_map = {}
    
    for i in range(len(goalies)):
        goalie_map[goalies[i]] = goalieId[i]
    
    today_df['goalieId'] = today_df['goalieFullName'].map(goalie_map)
    
    today_df['gameDate'] = today
    
    df_merged = pd.concat([df_merged, today_df])
    
    return df_merged
    
def rolling_avg(df, groupby, col, rolling):
    
    for x in col:
        
        if rolling > 1:
            df[f'{x}Ma{rolling}'] = df.groupby(groupby)[x].transform(lambda x: x.rolling(rolling).mean()).shift(1)
    
        else:
            df[f'{x}LastGame'] = df.groupby(groupby)[x].shift(1)
        
    return df
    
def fan_points(df):
    
    fan_points = ((df['assists'] * 8) + (df['goals'] * 12) + (df['blockedShots'] * 1.6) +(df['ppPoints'] * 0.5) + (df['shots'] * 1.6) + (df['shPoints'] * 2))
    
    return fan_points


def overperform(df, column, groupby):
    
    over_perf = df.groupby(groupby)[column].transform(lambda x: x - x.mean())
    
    return over_perf

def over_perf_dummy(df, column):
    
    new_col  = np.where(df[column] > 0, 1, 0)
    
    return new_col

def under_perf_dummy(df, column):
    
    new_col = np.where(df[column] < 0, 1, 0)
    
    return new_col

def better_home_road_splitter(df, hr_col, player_col, hr_perf_col):
    
    better_home = list(np.where((df[hr_col] == 'H') & (df[hr_perf_col] > 0), df[player_col], None))
    better_road = list(np.where((df[hr_col] == 'R') & (df[hr_perf_col] > 0), df[player_col], None))
    
    better_home = [*set(better_home)]
    better_road = [*set(better_road)]
    
    return better_home, better_road     
    
    
def same_perf_dummy(df, column):
    
    new_col = np.where(df[column] == 0, 1, 0)

    return new_col

def home_away_perf(df, column, groupby):
    
    home_perf = df.groupby(groupby)[column].transform(lambda x: x.mean())
    
    return home_perf

def stat_per_60(df, time_column, stat_col):
    
    new_col = (df[stat_col] / df[time_column])*360
    
    return new_col

    
def imputater(df, impute_by_player, impute_by_goalie, impute_by_team, player_col, goalie_col, opp_team_col, drop, impute_by_op, overperform_col, drop2):
    
    for col in impute_by_player:
        df[col] = df.groupby(player_col)[col].transform(lambda x: x.fillna(x.mean()))
    
    for col in impute_by_goalie:
        df[col] = df.groupby(goalie_col)[col].transform(lambda x: x.fillna(x.mean()))
        
    for col in impute_by_team:
        df[col] = df.groupby(opp_team_col)[col].transform(lambda x: x.fillna(x.mean()))
    
    df.drop(drop, axis=1, inplace=True)
    
    for col in impute_by_op:
        df[col] = df.groupby(overperform_col)[col].transform(lambda x: x.fillna(x.mean()))
    
    df.drop(drop2, axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)
    
    return df


def dummy_variables(df, cat_vars):
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encode_cols = pd.DataFrame(encoder.fit_transform(df[cat_vars]))
    encode_cols.index = df.index
    numer_df = df.drop(cat_vars, axis=1)
    encode_df = pd.concat([numer_df, encode_cols], axis=1)

    return encode_df

def scaler(X):

    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    return X


def splitting(df, today, date_col, hr_col, hr_perf_col):
    
    predictable_df = df[df[date_col] == today]
    
    home_df = pd.concat([predictable_df[((predictable_df[hr_perf_col] > 0) & (predictable_df[hr_col] == 'H'))], predictable_df[(predictable_df[hr_perf_col] <= 0) & (predictable_df[hr_col] == 'R')]])
    road_df = pd.concat([predictable_df[((predictable_df[hr_perf_col] > 0) & (predictable_df[hr_col] == 'R'))], predictable_df[(predictable_df[hr_perf_col] <= 0) & (predictable_df[hr_col] == 'H')]])
    
    home_df_d = home_df[home_df['positionCodeCopy'] == 'D']
    home_df_c = home_df[home_df['positionCodeCopy'] == 'C']
    home_df_w = home_df[(home_df['positionCodeCopy'] == 'L') | (home_df['positionCodeCopy'] == 'R')]
    
    road_df_d = road_df[road_df['positionCodeCopy'] == 'D']
    road_df_c = road_df[road_df['positionCodeCopy'] == 'C']
    road_df_w = road_df[(road_df['positionCodeCopy'] == 'L') | (road_df['positionCodeCopy'] == 'R')]
    
    home_df_d_good = home_df_d[home_df_d['avgFanPoints'] >= home_df_d['avgFanPoints'].mean()]
    home_df_d_bad = home_df_d[home_df_d['avgFanPoints'] < home_df_d['avgFanPoints'].mean()]

    home_df_c_good = home_df_c[home_df_c['avgFanPoints'] >= home_df_c['avgFanPoints'].mean()]
    home_df_c_bad = home_df_c[home_df_c['avgFanPoints'] < home_df_c['avgFanPoints'].mean()]

    home_df_w_good = home_df_w[home_df_w['avgFanPoints'] >= home_df_w['avgFanPoints'].mean()]
    home_df_w_bad = home_df_w[home_df_w['avgFanPoints'] < home_df_w['avgFanPoints'].mean()]
    
    road_df_d_good = road_df_d[road_df_d['avgFanPoints'] >= road_df_d['avgFanPoints'].mean()]
    road_df_d_bad = road_df_d[road_df_d['avgFanPoints'] < road_df_d['avgFanPoints'].mean()]

    road_df_c_good = road_df_c[road_df_c['avgFanPoints'] >= road_df_c['avgFanPoints'].mean()]
    road_df_c_bad = road_df_c[road_df_c['avgFanPoints'] < road_df_c['avgFanPoints'].mean()]

    road_df_w_good = road_df_w[road_df_w['avgFanPoints'] >= road_df_w['avgFanPoints'].mean()]
    road_df_w_bad = road_df_w[road_df_w['avgFanPoints'] < road_df_w['avgFanPoints'].mean()]
    
    return home_df_c_good, home_df_c_bad, home_df_d_good, home_df_d_bad, home_df_w_good, home_df_w_bad, road_df_c_good, road_df_c_bad, road_df_d_good, road_df_d_bad, road_df_w_good, road_df_w_bad


def make_pred(X, model_path):
    
    model = joblib.load(model_path)
    
    predictions = model.predict(X)
    
    return predictions
    
    
def main(save_pred=True):
    nhl_teams = nhl_teams = 'ANA ANH ARI BOS BUF CAR CGY CHI CLS CBJ COL DAL DET EDM FLA LA LAK MIN MON MTL NJ NSH NYI NYR OTT PHI PIT SEA SJ SJS STL TB TBL TOR VAN VGK WAS WPG'.split()

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
                'NYI':'NYI', 
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

    end_date = "2022-11-30"
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.today().strftime("%Y-%m-%d")

    home_xpath = '/html/body/div[1]/div/main/div[3]//div//div//div//div//a[2]//div//text()'
    road_xpath = '/html/body/div[1]/div/main/div[3]//div//div//div//div//a[1]//div//text()'
    home_goalies_xpath = '/html/body/div[1]/div/main/div[3]//div//div//div//ul[2]//li[1]//div[1]/a[1]/text()'
    road_goalies_xpath = '/html/body/div[1]/div/main/div[3]//div//div//div//ul[1]//li[1]//div[1]/a[1]/text()'
    
    road_teams, home_teams, games_dict_road, games_dict_home, goalie_dict = todays_games(url="https://www.rotowire.com/hockey/nhl-lineups.php", 
                                                                                         home_xpath = home_xpath, 
                                                                                         road_xpath = road_xpath, 
                                                                                         home_goalies_xpath=home_goalies_xpath, 
                                                                                         road_goalies_xpath=road_goalies_xpath, 
                                                                                         nhl_teams=nhl_teams, 
                                                                                         team_map=team_map)
    
    
    
    
    df_skater = dg_main(yesterday, end_date, 'skater', "~/hockey/data/df_skater.csv", 0, 10000, 100, update=True, saveData=False)
    df_skater = df_skater.drop(['Unnamed: 0.1', 'evGoals', 'evPoints','faceoffWinPct', 'gameWinningGoals', 'gamesPlayed', 'lastName', 'otGoals', 'pointsPerGame', 'timeOnIcePerGame', 'Unnamed: 0'], axis=1)
    
    df_misc = dg_main(yesterday, end_date, 'misc', "~/hockey/data/df_misc.csv", 0, 10000, 100, update=True, saveData=False)
    df_misc = df_misc.drop(['Unnamed: 0.1', 'blockedShotsPer60', 'emptyNetAssists', 'homeRoad', 'emptyNetGoals', 'emptyNetPoints', 'firstGoals', 'gamesPlayed', 'giveaways', 'giveawaysPer60', 'hits', 'hitsPer60', 'missedShotCrossbar', 'missedShotGoalpost', 'missedShotOverNet', 'missedShotWideOfNet', 'missedShots', 'opponentTeamAbbrev', 'otGoals', 'takeaways', 'takeawaysPer60', 'Unnamed: 0'], axis=1)
    
    df_shot = dg_main(yesterday, end_date, 'shots', "~/hockey/data/df_shots.csv", 0, 10000, 100, update=True, saveData=False)
    df_shot = df_shot.drop(['Unnamed: 0.1', 'gamesPlayed', 'goals', 'homeRoad', 'lastName', 'opponentTeamAbbrev', 'teamAbbrev', 'skaterFullName', 'Unnamed: 0'], axis=1)    
    df_toi = dg_main(yesterday, end_date, 'toi', "~/hockey/data/df_toi.csv", 0, 10000, 100, update=True, saveData=False)
    df_toi = df_toi.drop(['Unnamed: 0.1', 'evTimeOnIce', 'evTimeOnIcePerGame', 'gameDate', 'gamesPlayed', 'homeRoad', 'lastName', 'opponentTeamAbbrev','otTimeOnIce', 'otTimeOnIcePerOtGame', 'positionCode', 'shootsCatches','skaterFullName', 'teamAbbrev', 'timeOnIcePerGame', 'Unnamed: 0'], axis=1)
    
    df_goalie = dg_main(yesterday, end_date, 'goalie', "/Users/nickdimmitt/hockey/data/df_goalies.csv", 0, 10000, 100, update=True, saveData=False)
    df_goalie = df_goalie.drop(['Unnamed: 0', 'assists', 'gamesStarted', 'goals', 'goalsAgainstAverage', 'lastName', 'points', 'saves', 'ties', 'timeOnIce', 'wins'], axis=1)
    
    df_team = dg_main(yesterday, end_date, 'team', "~/hockey/data/df_teams.csv", 0, 10000, 100, update=True, saveData=False)
    df_team = df_team.drop(['Unnamed: 0', 'faceoffWinPct', 'gamesPlayed', 'goalsAgainstPerGame', 'goalsForPerGame', 'losses', 'otLosses', 'penaltyKillNetPct', 'pointPct', 'powerPlayNetPct', 'powerPlayPct', 'regulationAndOtWins', 'ties', 'wins', 'winsInRegulation', 'winsInShootout'], axis=1)
    
    df_goalie['goalieId'] = df_goalie['playerId'].copy()
    df_goalie['teamAbbrevMerge'] = df_goalie['opponentTeamAbbrev'].copy()
    df_team['teamAbbrevMerge'] = df_team['opponentTeamAbbrev'].copy()
    df_skater['teamAbbrevMerge'] = df_skater['teamAbbrev'].copy()
    df_goalie = df_goalie[['gameId', 'goalieId','goalieFullName','teamAbbrevMerge','savePct']]
    df_team = df_team[['gameId', 'teamId', 'teamAbbrevMerge', 'goalsAgainst', 'shotsAgainstPerGame']]
    
    df_merged = pd.merge(df_skater, df_misc, on=['gameId', 'playerId'])
    df_merged = pd.merge(df_merged, df_shot, on=['gameId', 'playerId'])
    df_merged = pd.merge(df_merged, df_toi, on=['gameId', 'playerId'])
    df_merged = pd.merge(df_merged, df_goalie, on=['gameId', 'teamAbbrevMerge'])
    df_merged = pd.merge(df_merged, df_team, on=['gameId', 'teamAbbrevMerge'])
    
    df_merged.drop_duplicates(inplace=True)
    
    df_merged = df_merged[['gameId', 'gameDate','playerId', 'opponentTeamAbbrev', 'teamAbbrevMerge', 'homeRoad', 'goalieId', 'goalieFullName', 'goals', 'assists', 'plusMinus',
       'points', 'positionCode_x', 'ppGoals', 'ppPoints', 'shGoals',
       'shPoints', 'shootingPct_x', 'shootsCatches_x', 'shots_x',
       'skaterFullName_x', 'blockedShots',
       'ppTimeOnIce', 'shTimeOnIce', 'shifts', 'timeOnIce',
       'timeOnIcePerShift', 'savePct', 'goalsAgainst', 'shotsAgainstPerGame']]
    df_merged.columns = df_merged.columns.str.rstrip('_x')
    
    
    df_merged = todays_df(df_merged=df_merged, 
                          date_col='gameDate', 
                          cutoff_date="2022-11-01", 
                          team_col='teamAbbrevMerge', 
                          home_teams=home_teams, 
                          road_teams=road_teams, 
                          today=today, 
                          goalie_dict=goalie_dict, 
                          columns=['goals', 'assists',
       'plusMinus', 'points', 'ppGoals', 'ppPoints', 'shGoals',
       'shPoints', 'shootingPct', 'shots', 'blockedShots', 'ppTimeOnIce', 'shTimeOnIce', 'shifts', 'timeOnIce',
       'timeOnIcePerShift', 'savePct', 'goalsAgainst', 'shotsAgainstPerGame'], 
                          games_dict_road=games_dict_road, 
                          games_dict_home=games_dict_home,
                          df_goalie=df_goalie)
    
    
    df_merged['fanPoints'] = fan_points(df_merged)
    df_merged['overPerform'] = overperform(df_merged, 'fanPoints', 'playerId')
    df_merged['homeRoadPerf'] = home_away_perf(df_merged, 'overPerform', ['playerId', 'homeRoad'])
    
    better_home_skater, better_road_skater = better_home_road_splitter(df=df_merged, hr_col='homeRoad', player_col='playerId', hr_perf_col='homeRoadPerf')

    df_merged['OpHomeDummy'] = np.where(df_merged['playerId'].isin(better_home_skater), 1, 0)
    df_merged['OpRoadDummy'] = np.where(df_merged['playerId'].isin(better_road_skater), 1, 0)
    df_merged['OpNowhereDummy'] = np.where((df_merged['OpHomeDummy'] == 0) & (df_merged['OpRoadDummy'] == 0), 1, 0)
    
    
    for x in [1,3,7,16]:
        
        df_merged = rolling_avg(df=df_merged, groupby='goalieId', col=['savePct'], rolling=x)
        df_merged = rolling_avg(df=df_merged, groupby='opponentTeamAbbrev', col=['goalsAgainst', 'shotsAgainstPerGame'], rolling=x)
        df_merged = rolling_avg(df=df_merged, groupby='playerId', col=['assists', 'goals', 'plusMinus', 'points', 'ppPoints', 'fanPoints', 'blockedShots','shootingPct', 'shots', 'timeOnIce', 'ppTimeOnIce', 'shifts', 'timeOnIcePerShift'], rolling=x)
    
    
    df_merged['avgFanPoints'] = df_merged.groupby('playerId')['fanPoints'].transform(lambda x: x.mean())
    df_merged['overPerformDummy'] = over_perf_dummy(df=df_merged, column='overPerform')
    
    impute_by_player = ['assistsMa7', 'goalsMa7', 'plusMinusMa7', 'pointsMa7',
       'ppPointsMa7', 'fanPointsMa7', 'blockedShotsMa7', 'shootingPctMa7',
       'shotsMa7', 'timeOnIceMa7', 'ppTimeOnIceMa7', 'shiftsMa7',
       'timeOnIcePerShiftMa7', 'assistsMa3', 'goalsMa3', 'plusMinusMa3',
       'pointsMa3', 'ppPointsMa3', 'fanPointsMa3', 'blockedShotsMa3',
       'shootingPctMa3', 'shotsMa3', 'timeOnIceMa3', 'ppTimeOnIceMa3',
       'shiftsMa3', 'timeOnIcePerShiftMa3', 'assistsLastGame', 'goalsLastGame',
       'plusMinusLastGame', 'pointsLastGame', 'ppPointsLastGame',
       'fanPointsLastGame', 'blockedShotsLastGame', 'shootingPctLastGame',
       'shotsLastGame', 'timeOnIceLastGame', 'ppTimeOnIceLastGame',
       'shiftsLastGame', 'timeOnIcePerShiftLastGame', 'assistsMa16',
       'goalsMa16', 'plusMinusMa16', 'pointsMa16', 'ppPointsMa16',
       'fanPointsMa16', 'blockedShotsMa16', 'shootingPctMa16', 'shotsMa16',
       'timeOnIceMa16', 'ppTimeOnIceMa16', 'shiftsMa16',
       'timeOnIcePerShiftMa16']

    impute_by_goalie = ['savePctLastGame', 'savePctMa3', 'savePctMa7', 'savePctMa16']

    impute_by_team = ['goalsAgainstLastGame', 'goalsAgainstMa3',
       'goalsAgainstMa7', 'goalsAgainstMa16', 'shotsAgainstPerGameLastGame',
       'shotsAgainstPerGameMa3', 'shotsAgainstPerGameMa7', 'shotsAgainstPerGameMa16']
    
    impute_by_op = ['assistsMa7', 'goalsMa7', 'plusMinusMa7', 'pointsMa7',
       'ppPointsMa7', 'fanPointsMa7', 'blockedShotsMa7', 'shootingPctMa7',
       'shotsMa7', 'timeOnIceMa7', 'ppTimeOnIceMa7', 'shiftsMa7',
       'timeOnIcePerShiftMa7', 'assistsMa3', 'goalsMa3', 'plusMinusMa3',
       'pointsMa3', 'ppPointsMa3', 'fanPointsMa3', 'blockedShotsMa3',
       'shootingPctMa3', 'shotsMa3', 'timeOnIceMa3', 'ppTimeOnIceMa3',
       'shiftsMa3', 'timeOnIcePerShiftMa3', 'assistsLastGame', 'goalsLastGame',
       'plusMinusLastGame', 'pointsLastGame', 'ppPointsLastGame',
       'fanPointsLastGame', 'blockedShotsLastGame', 'shootingPctLastGame',
       'shotsLastGame', 'timeOnIceLastGame', 'ppTimeOnIceLastGame',
       'shiftsLastGame', 'timeOnIcePerShiftLastGame', 'assistsMa16',
       'goalsMa16', 'plusMinusMa16', 'pointsMa16', 'ppPointsMa16',
       'fanPointsMa16', 'blockedShotsMa16', 'shootingPctMa16', 'shotsMa16',
       'timeOnIceMa16', 'ppTimeOnIceMa16', 'shiftsMa16',
       'timeOnIcePerShiftMa16', 'savePctLastGame', 'savePctMa3', 'savePctMa7', 'savePctMa16', 'goalsAgainstLastGame', 'goalsAgainstMa3',
       'goalsAgainstMa7', 'goalsAgainstMa16', 'shotsAgainstPerGameLastGame',
       'shotsAgainstPerGameMa3', 'shotsAgainstPerGameMa7', 'shotsAgainstPerGameMa16']
    
    df_merged = imputater(df=df_merged,
                          impute_by_player=impute_by_player,
                          impute_by_goalie=impute_by_goalie,
                          impute_by_team=impute_by_team,
                          player_col='playerId',
                          goalie_col='goalieId',
                          opp_team_col='opponentTeamAbbrev',
                          drop='shootingPct',
                          impute_by_op=impute_by_op,
                          overperform_col='overPerformDummy',
                          drop2='goalieId')
    
    
    df_merged = dummy_variables(df=df_merged, cat_vars=['homeRoad', 'positionCode', 'opponentTeamAbbrev', 'teamAbbrevMerge'])
    
    df_merged.drop(['playerId', 'goals', 'assists',
       'plusMinus', 'points', 'ppGoals', 'ppPoints', 'shGoals',
       'shPoints', 'shootsCatches', 'shots', 'blockedShots',
       'ppTimeOnIce', 'shTimeOnIce', 'shifts', 'timeOnIce',
       'timeOnIcePerShift'], axis=1, inplace=True)
    
    #home_df_c_good, home_df_c_bad, home_df_d_good, home_df_d_bad, home_df_w_good, home_df_w_bad, road_df_c_good, road_df_c_bad, road_df_d_good, road_df_d_bad, road_df_w_good, road_df_w_bad = splitting(df=df_merged, today=today, date_col='gameDate', hr_col='homeRoadCopy', hr_perf_col='homeRoadPerf')
    
    
    features = ['savePctLastGame', 'savePctMa3', 'savePctMa7',
       'savePctMa16', 'goalsAgainstLastGame', 'goalsAgainstMa3',
       'goalsAgainstMa7', 'goalsAgainstMa16', 'shotsAgainstPerGameLastGame',
       'shotsAgainstPerGameMa3', 'shotsAgainstPerGameMa7', 'shotsAgainstPerGameMa16', 'homeRoadPerf', 'OpHomeDummy', 'OpRoadDummy',
       'OpNowhereDummy','assistsMa7', 'goalsMa7', 'plusMinusMa7', 'pointsMa7',
       'ppPointsMa7', 'fanPointsMa7', 'blockedShotsMa7', 'shootingPctMa7',
       'shotsMa7', 'timeOnIceMa7', 'ppTimeOnIceMa7', 'shiftsMa7',
       'timeOnIcePerShiftMa7', 'assistsMa3', 'goalsMa3', 'plusMinusMa3',
       'pointsMa3', 'ppPointsMa3', 'fanPointsMa3', 'blockedShotsMa3',
       'shootingPctMa3', 'shotsMa3', 'timeOnIceMa3', 'ppTimeOnIceMa3',
       'shiftsMa3', 'timeOnIcePerShiftMa3', 'assistsLastGame', 'goalsLastGame',
       'plusMinusLastGame', 'pointsLastGame', 'ppPointsLastGame',
       'fanPointsLastGame', 'blockedShotsLastGame', 'shootingPctLastGame',
       'shotsLastGame', 'timeOnIceLastGame', 'ppTimeOnIceLastGame',
       'shiftsLastGame', 'timeOnIcePerShiftLastGame', 'assistsMa16',
       'goalsMa16', 'plusMinusMa16', 'pointsMa16', 'ppPointsMa16',
       'fanPointsMa16', 'blockedShotsMa16', 'shootingPctMa16', 'shotsMa16',
       'timeOnIceMa16', 'ppTimeOnIceMa16', 'shiftsMa16',
       'timeOnIcePerShiftMa16', 'avgFanPoints', 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73]
    
    target = 'fanPoints'
    
    predictable_df = df_merged[df_merged['gameDate'] == today]
    
    X = predictable_df[features].values
    X = scaler(X)

    predictable_df['predictions'] = make_pred(X, '/Users/nickdimmitt/hockey/old_work/training/ohmodel2020_scaled.pkl')    
    final_pred = [['skaterFullName', 'positionCodeCopy', 'teamAbbrevMerge', 'predictions']].sort_values(by='predictions', ascending=False)
    
    path='/Users/nickdimmitt/hockey/predictions/pred'
    
    if save_pred:
        final_pred.to_csv(f'{path}_{today}.csv')
        
if __name__=='__main__':
    main()