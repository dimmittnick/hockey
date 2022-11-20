import pandas as pd
import numpy as np
import json
import requests
from datetime import *
from dateutil import *

## this needs to be updated after data is read in everytime and added to database
## most recent date: 11/01/22


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



def main(start_date, end_date, data_seg, update_path, start, stop, step, update=False, saveData=False):
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
        df_update.to_csv(f"data/df_{data_seg}.csv")
    
    if update:
        
        return df_update
    
    return df
