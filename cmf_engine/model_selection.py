# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:17:50 2023

@author: BGondaliya
"""

import datetime
from datetime import date
import pandas as pd
from dateutil.relativedelta import relativedelta


class model_selection:
    def __init__(self,refresh_date,train_basins):
        self.ref_date = refresh_date
        self.refresh_date = str(pd.to_datetime(refresh_date).replace(day=1)) #str(date.today(
        temp = pd.to_datetime(refresh_date) + relativedelta(months=-12)
        self.cutoff_date = temp.strftime('%YQ{}'.format( (temp.month - 1) // 3 + 1))
        self.train_basins = train_basins


    def best_model(self, X):
        data = X[X['refresh_date']== self.ref_date]
        data = data[(data['quarter'] > self.cutoff_date) & (data['quarter'] < self.ref_date)]

        data['abs_rev_error'] = abs(data['revenue_error'])
        data['abs_cm_error'] = abs(data['contribution_margin_error'])
        
        grouped_df = data.groupby(['SL BASIN (CODE)', 'model','refresh_date']).agg({'abs_rev_error': 'mean', 'abs_cm_error': 'mean'})
        grouped_df = grouped_df.dropna()
        
        grouped = grouped_df.groupby('SL BASIN (CODE)')
        min_indices = grouped['abs_rev_error'].idxmin()
        best_rev_df = grouped_df.loc[min_indices]
        
        grouped = grouped_df.groupby('SL BASIN (CODE)')

        min_indices = grouped['abs_cm_error'].idxmin()
        best_cm_df = grouped_df.loc[min_indices]
        
        # to extract dataframe for best model
        data = X[X['refresh_date']== self.ref_date]
#         data = data[data['refresh_date']==self.refresh_date]
        ans = pd.DataFrame(columns=['quarter', 'SL BASIN (CODE)', 'rev', 'actual_revenue',
               'revenue_error', 'YM', 'refresh_date', 'level', 'rev_Model', 'cm',
               'actual_contribution_margin', 'contribution_margin_error', 'cm_Model'])   
        for basin in self.train_basins:
            print(basin)

            rev_model = best_rev_df.loc[best_rev_df.index.get_level_values('SL BASIN (CODE)') == basin].index.get_level_values('model')
            cm_model = best_cm_df.loc[best_cm_df.index.get_level_values('SL BASIN (CODE)') == basin].index.get_level_values('model')

            df_rev = data[(data['SL BASIN (CODE)']==basin) &(data['model']==rev_model[0])][['quarter', 'SL BASIN (CODE)', 'rev',
                    'actual_revenue','revenue_error', 'YM', 'refresh_date',
                   'level']]
            df_rev['rev_Model'] = str(rev_model[0])
            df_rev = df_rev.drop_duplicates()

            df_cm = data[(data['SL BASIN (CODE)']==basin) &(data['model']==cm_model[0])][['quarter', 'SL BASIN (CODE)', 'cm',
                   'actual_contribution_margin', 
                   'contribution_margin_error', ]]
            df_cm['cm_Model'] = str(cm_model[0])
            df_cm = df_cm.drop_duplicates()

            ans = ans.append(pd.merge(df_rev,df_cm, how='left', on = ['quarter','SL BASIN (CODE)']))

        return ans