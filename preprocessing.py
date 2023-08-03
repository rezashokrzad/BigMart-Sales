# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:26:56 2023

@author: rezas
"""
from sklearn.preprocessing import LabelEncoder, StandardScaler # lead to 52% r2 while get_dummies 57%
from sklearn.model_selection import train_test_split

import pandas as pd
def preprocess(df):
    # encoder = LabelEncoder()
    # df['Item_Identifier'] = encoder.fit_transform(df['Item_Identifier'])
    # df['Item_Fat_Content'] = encoder.fit_transform(df['Item_Fat_Content'])
    # df['Item_Type'] = encoder.fit_transform(df['Item_Type'])
    # df['Outlet_Identifier'] = encoder.fit_transform(df['Outlet_Identifier'])
    # df['Outlet_Size'] = encoder.fit_transform(df['Outlet_Size'])
    # df['Outlet_Location_Type'] = encoder.fit_transform(df['Outlet_Location_Type'])
    # df['Outlet_Type'] = encoder.fit_transform(df['Outlet_Type'])
    
    
    # print(df.head())
    df_encoded = pd.get_dummies(df, columns=['Item_Identifier', 'Item_Fat_Content', 'Item_Type',
                                             'Outlet_Identifier', 'Outlet_Size',
                                             'Outlet_Location_Type', 'Outlet_Type'])
    df = df_encoded.copy()
    #creating dataset
    X = df.drop(columns='Item_Outlet_Sales', axis=1)
    y = df['Item_Outlet_Sales']
    
    
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    SS = StandardScaler()
    X_train = SS.fit_transform(X_train)
    X_test = SS.transform(X_test)
    
    return X_train, X_test, y_train, y_test
    
