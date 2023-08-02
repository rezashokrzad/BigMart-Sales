# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:32:47 2023

@author: rezas
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(data, bins, kde, ax, title):
    sns.histplot(data, bins=bins, kde=kde, ax=ax)
    ax.set_title(title)

def plot_count_plot(df, column, title):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df)
    plt.title(title)
    plt.show()
    
    
    
def plot_all_visualizations(df):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    plot_histogram(df['Item_Weight'], bins=20, kde=True, ax=axes[0, 0], title='Item Weight Distribution')
    plot_histogram(df['Item_Visibility'], bins=20, kde=True, ax=axes[0, 1], title='Item Visibility Distribution')
    plot_histogram(df['Item_MRP'], bins=20, kde=True, ax=axes[1, 0], title='Item MRP Distribution')
    plot_histogram(df['Item_Outlet_Sales'], bins=20, kde=True, ax=axes[1, 1], title='Item Outlet Sales Distribution')

    plt.tight_layout()
    plt.show()

    plot_count_plot(df, 'Outlet_Establishment_Year', 'Outlet Establishment Year')
    plot_count_plot(df, 'Item_Fat_Content', 'Item Fat Content')
    plot_count_plot(df, 'Item_Type', 'Item Type')
    plot_count_plot(df, 'Outlet_Size', 'Outlet Size')