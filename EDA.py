# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:46:06 2023

@author: rezas
"""

import pandas as pd

def EDA_process(file_path):
    #read data
    df = pd.read_csv(file_path)
    
    #describe data
    print(df.shape)
    print(df.info())
    print(df.describe(include='all'))
    print(df.head())
    print(df['Item_Type'].unique())
    print(df['Item_Fat_Content'].unique())
    
    #replace LF and low fat with Low Fat / also reg with Regular
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})
    print(df['Item_Fat_Content'].unique())
    
    
    #check nulls
    print(df.isnull().sum())
    
    #handle nulls (mean for item_weight | mode for outlet_size)
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
    
    # filling the missing values in "Outlet_Size" column with Mode
    #Here we take Outlet_Size column & Outlet_Type column since they are correlated
    mode_of_Outlet_size = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
    print(mode_of_Outlet_size)
    miss_values = df['Outlet_Size'].isnull()
    df.loc[miss_values, 'Outlet_Size'] = df.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])
    
    
    print(df.isnull().sum())
    

    return df