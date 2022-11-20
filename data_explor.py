import pandas as pd
import numpy as np


##############################################################################
# VARIABLE CREATION
##############################################################################

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
    
def same_perf_dummy(df, column):
    
    new_col = np.where(df[column] == 0, 1, 0)

    return new_col

def home_away_perf(df, column, groupby):
    
    home_perf = df.groupby(groupby)[column].transform(lambda x: x.mean())
    
    return home_perf

def stat_per_60(df, time_column, stat_col):
    
    new_col = (df[stat_col] / df[time_column])*360
    
    return new_col

def moving_average(df, column, groupby, window):
    
    ma = df.groupby(groupby)[column].transform(lambda x: x.rolling(window).mean())
    
    return ma

