import pandas as pd
import numpy as np
import sys


##############################################################################
# READING IN DATA FROM EITHER ONLINE LOCATION OR LOCAL
##############################################################################

def read_data(path):
    
    df = pd.read_csv(path)
    
    return df

def save_data(df, path, name):
    df.to_csv(path + name)
    

##############################################################################
# SANITY CHECK DATA
##############################################################################

def data_review(df):
    
    print(df.shape)
    print(df.isna().sum())
    print(df.duplicated().value_counts())
    return df.head()

##############################################################################
# DATA CLEAN
##############################################################################

def drop_duplicates(df):
    
    df = df.drop_duplicates()
    
    return df

def make_index(df, columns):
    
    df = df.set_index(columns)
    
    return df

def remove_index(df):
    
    df = df.reset_index()
    
    return df

def remove_columns(df, columns):
    
    try:
        df = df.drop(labels=columns, axis=1)
        return df
    except:
        return df

def rename_columns(df, rename_dict):
    
    df1 = df.rename(columns=rename_dict)
    
    return df1


def remove_dup_col(df):
    
    df.columns = df.columns.str.rstrip("_x")
    df.columns = df.columns.str.rstrip("_y")
    df1 = df.loc[:,~df.columns.duplicated()].copy()
    
    return df1

def handle_missing(df, groupby, column, how='mean'):
    
    if how == 'mean':
        df[column] = df.groupby(groupby)[column].transform(lambda x: x.fillna(x.mean()))
    
    elif how == 'median':
        df[column] = df.groupby(groupby)[column].transform(lambda x: x.fillna(x.median()))
    
    elif how == 'mode':
        df[column] = df.groupby(groupby)[column].transform(lambda x: x.fillna(x.mode()))
    
    else:
        df[column] = df.groupby(groupby)[column].transform(lambda x: x.fillna(0))
            
    return df

##############################################################################
# DATA JOIN
##############################################################################

def df_join_index(df1, df2):
    
    df = pd.merge(df1, df2, how='inner', left_index=True, right_index=True)
    
    return df


##############################################################################
# MAIN THAT UPDATES, CLEANS, MERGES DATA
##############################################################################

def main(df1, df2, df3, df4, shared_index, drop_columns, merged_drop_col, saveData=False):
    
    df1 = drop_duplicates(df1)
    df2 = drop_duplicates(df2)
    df3 = drop_duplicates(df3)
    df4 = drop_duplicates(df4)
    
    
    df1 = make_index(df1, shared_index)
    df2 = make_index(df2, shared_index)
    df3 = make_index(df3, shared_index)
    df4 = make_index(df4, shared_index)

    
    
    df1 = remove_columns(df1, drop_columns)
    df2 = remove_columns(df2, drop_columns)
    df3 = remove_columns(df3, drop_columns)
    df4 = remove_columns(df4, drop_columns)

    
    merged_df = df_join_index(df4, df_join_index(df3, df_join_index(df1, df2)))
    merged_df = remove_index(merged_df)
    merged_df = remove_dup_col(merged_df)
    merged_df = drop_duplicates(merged_df)
    merged_df = remove_columns(merged_df, merged_drop_col)
    
    if saveData:
        merged_df.to_csv('data/merged_df.csv')
        
    return merged_df
