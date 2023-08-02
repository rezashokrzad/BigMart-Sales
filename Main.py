# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:28:04 2023

@author: rezas
"""

from EDA import EDA_process
from Visualization import plot_all_visualizations


def main():
    file_path = './train.csv'
    df = EDA_process(file_path)
    plot_all_visualizations(df)
    
if __name__ == "__main__":
    main()

