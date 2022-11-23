import pandas as pd
import numpy as np
import sys


##############################################################################
# READING IN DATA FROM EITHER ONLINE LOCATION OR LOCAL
##############################################################################

def read_data(path_to_data, skater=False, goalie=False, team=False):
    
    if skater:
        df_skate = pd.read_csv(path_to_data + "df_skaters.csv")
        
        return df_skate
    
    if goalie:
        df_goalie = pd.read_csv(path_to_data + "df_goalies.csv")
        return df_goalie
    
    if team:
        df_team = pd.read_csv(path_to_data + "df_teams.csv")
        
        return df_team
    
    
###############################################################################
# VARIABLE CREATION
###############################################################################

def add_fantasy(df, column_name):
    
    df[column_name] = ((df['assists'] * 8) + (df['goals'] * 12) + (df['blocked'] * 1.6) +(df['ppPoints'] * 0.5) + (df['shots'] * 1.6) + (df['shPoints'] * 2))
    
    return df

def moving_average(df, column, groupby, window):
    
    ma = df.groupby(groupby)[column].transform(lambda x: x.rolling(window).mean())
    
    return ma

def percShotType(df, groupby, col1, col2):
    
    df['all1'] = df.groupby(groupby)[col1].transform(sum)
    df['all2'] = df.groupby(groupby)[col2].transform(sum)
    
    return np.where(df['all2']>0, df['all1']/df['all2'], 0)