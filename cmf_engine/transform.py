# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:08:02 2023

@author: BGondaliya
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:32:58 2023

@author: BGondaliya

In this Module there are three class defined:

1. TrainingTransformer: To transform data and generate features for training set.
2. InferenceRevTransformer: To generate data for test/inference set of Revenue Prediction.
3. InferenceConMarTransformer: To generate data for test/inference set of Contribution margin.

"""
from datetime import date

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.base import BaseEstimator, TransformerMixin


class TrainingTransformer(TransformerMixin, BaseEstimator):

    def __init__(self,start_date,interval,basins):

        temp_date = pd.to_datetime(start_date,format="%Y-%m-%d")
        self.start_date = temp_date.strftime('%YQ{}'.format( (temp_date.month - 1) // 3 + 1))
        self.interval = interval
        temp_date_2 = pd.to_datetime(date.today() + relativedelta(months=+self.interval*3),format="%Y-%m-%d")
        self.end_date = temp_date_2.strftime('%YQ{}'.format( (temp_date_2.month - 1) // 3 + 1))
        self.basins = basins


    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X, oil_df):
#         X["YEARMONTH"] = pd.to_datetime(X["YEARMONTH"], format="%Y-%m")
        X = X.merge(oil_df)
        X = X.sort_values(["quarter"])
        


        # Aggregate
        X_agg = X.reset_index().drop(["index","SL DIVISION (CODE)"],axis=1)
        X_agg = X_agg.round(2)
        col_mapping = {"TOTAL_WEIGHTED_FCST":"c4c_rev","TOTAL_RIG_COUNT_BASIN_woChina":"gac_rev",
                       "TOTAL_RIG_COUNT_BASIN_all":"gaca_rev","CM":"cm","REV":"rev"}

        X_agg = X_agg.rename(col_mapping,axis=1)

        # Indexing
        X_agg.set_index(X_agg.quarter,inplace=True)
        X_agg = X_agg.drop("quarter",axis=1)
        
        # Errorenous Data removal
        # X_agg = X_agg[X_agg["SL BASIN (CODE)"]!='GHQ']


        gac_cols = ["gac_rev","gaca_rev"]
        c4c_cols = ["c4c_rev"]
        hfm_cols = ["rev"]
        oil_cols = ["WTI_OIL_PRICE_USD","BC_OIL_PRICE_USD"]
        
        for basin in self.basins:
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gac_activity(t-1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gac_rev"].shift(1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gac_activity(t-2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gac_rev"].shift(2))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gac_activity(t-3)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gac_rev"].shift(3))

            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gac_activity(t+1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gac_rev"].shift(-1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gac_activity(t+2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gac_rev"].shift(-2))

            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gaca_activity(t-1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gaca_rev"].shift(1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gaca_activity(t-2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gaca_rev"].shift(2))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gaca_activity(t-3)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gaca_rev"].shift(3))

            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gaca_activity(t+1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gaca_rev"].shift(-1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gaca_activity(t+2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gaca_rev"].shift(-2))

            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'c4c_activity(t-1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"c4c_rev"].shift(1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'c4c_activity(t-2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"c4c_rev"].shift(2))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'c4c_activity(t-3)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"c4c_rev"].shift(3))

            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'c4c_activity(t+1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"c4c_rev"].shift(-1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'c4c_activity(t+2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"c4c_rev"].shift(-2))

            ##
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'wti_activity(t-1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"WTI_OIL_PRICE_USD"].shift(1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'wti_activity(t-2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"WTI_OIL_PRICE_USD"].shift(2))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'wti_activity(t-3)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"WTI_OIL_PRICE_USD"].shift(3))

            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'wti_activity(t+1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"WTI_OIL_PRICE_USD"].shift(-1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'wti_activity(t+2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"WTI_OIL_PRICE_USD"].shift(-2))
            
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'btc_activity(t-1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"BC_OIL_PRICE_USD"].shift(1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'btc_activity(t-2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"BC_OIL_PRICE_USD"].shift(2))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'btc_activity(t-3)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"BC_OIL_PRICE_USD"].shift(3))

            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'btc_activity(t+1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"BC_OIL_PRICE_USD"].shift(-1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'btc_activity(t+2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"BC_OIL_PRICE_USD"].shift(-2))

            
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'rev_activity(t-1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"rev"].shift(1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'rev_activity(t-2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"rev"].shift(2))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'rev_activity(t-3)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"rev"].shift(3))


#         X_agg["quarter_flag"] = 0
#         X_agg.loc[X_agg.index.month%3==0,'quarter_flag']=1

        X_agg = X_agg[(X_agg.index>=self.start_date)&(X_agg.index<self.end_date)]
        training_df = X_agg[X_agg["rev"]>0]

        return (training_df)
    
    @staticmethod
    def prophet_transformer(training_df,oil_df):
        pr_train_df = training_df.reset_index()
        pr_train_df = pr_train_df[['quarter', 'SL BASIN (CODE)', 'c4c_rev', 'gac_rev','gaca_rev', 'cm', 'rev','quarter_flag']].drop_duplicates()
        pr_train_df = pr_train_df.merge(oil_df,on=["quarter"],how="left")
        return pr_train_df


class InferenceRevTransformer(TransformerMixin, BaseEstimator):

    def __init__(self,interval,pickup_date,basins):
        self.interval = interval
        self.pickup_date = pd.to_datetime(pickup_date,format="%Y-%m-%d")
        
        temp_date = pd.to_datetime(self.pickup_date + relativedelta(months=-12),format="%Y-%m-%d")
        self.buffer_start_date = temp_date.strftime('%YQ{}'.format( (temp_date.month - 1) // 3 + 1))
        
        temp_date2 = pd.to_datetime(self.pickup_date + relativedelta(months=-3),format="%Y-%m-%d")
        self.start_date = temp_date2.strftime('%YQ{}'.format( (temp_date2.month - 1) // 3 + 1))
        
        temp_date3 = pd.to_datetime(self.pickup_date + relativedelta(months=+(self.interval*3+9)),format="%Y-%m-%d")
        self.buffer_end_date = temp_date3.strftime('%YQ{}'.format( (temp_date3.month - 1) // 3 + 1))
        
        temp_date4 = pd.to_datetime(self.pickup_date + relativedelta(months=+(self.interval*3-3)),format="%Y-%m-%d")
        self.end_date = temp_date4.strftime('%YQ{}'.format( (temp_date4.month - 1) // 3 + 1))
        
        self.basins = basins
        self.pickup_date = pickup_date.strftime('%YQ{}'.format( (pickup_date.month - 1) // 3 + 1))

    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X, oil_df):
#         X["YEARMONTH"] = pd.to_datetime(X["YEARMONTH"], format="%Y-%m")
        X = X.merge(oil_df)
        # X = X[(X["quarter"]>self.buffer_start_date)&(X["quarter"]<self.buffer_end_date)]
        
        # X.loc[X["quarter"]>=self.pickup_date,["CM","REV"]] = np.nan
        X = X.sort_values(["quarter"])

        # Aggregate
        X_agg = X.reset_index().drop(["index","SL DIVISION (CODE)"],axis=1)
        X_agg = X_agg.round(2)
        col_mapping = {"TOTAL_WEIGHTED_FCST":"c4c_rev","TOTAL_RIG_COUNT_BASIN_woChina":"gac_rev",
                       "TOTAL_RIG_COUNT_BASIN_all":"gaca_rev","CM":"cm","REV":"rev"}

        X_agg = X_agg.rename(col_mapping,axis=1)

        # Indexing
        X_agg.set_index(X_agg.quarter,inplace=True)
        X_agg = X_agg.drop("quarter",axis=1)
                
        # Errorenous Data removal
        # X_agg = X_agg[X_agg["SL BASIN (CODE)"]!='GHQ']

        gac_cols = ["gac_rev","gaca_rev"]
        c4c_cols = ["c4c_rev"]
        hfm_cols = ["rev"]
        oil_cols = ["WTI_OIL_PRICE_USD","BC_OIL_PRICE_USD"]
        
        for basin in self.basins:
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gac_activity(t-1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gac_rev"].shift(1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gac_activity(t-2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gac_rev"].shift(2))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gac_activity(t-3)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gac_rev"].shift(3))

            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gac_activity(t+1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gac_rev"].shift(-1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gac_activity(t+2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gac_rev"].shift(-2))

            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gaca_activity(t-1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gaca_rev"].shift(1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gaca_activity(t-2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gaca_rev"].shift(2))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gaca_activity(t-3)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gaca_rev"].shift(3))

            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gaca_activity(t+1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gaca_rev"].shift(-1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'gaca_activity(t+2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"gaca_rev"].shift(-2))

            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'c4c_activity(t-1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"c4c_rev"].shift(1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'c4c_activity(t-2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"c4c_rev"].shift(2))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'c4c_activity(t-3)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"c4c_rev"].shift(3))

            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'c4c_activity(t+1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"c4c_rev"].shift(-1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'c4c_activity(t+2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"c4c_rev"].shift(-2))

            ##
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'wti_activity(t-1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"WTI_OIL_PRICE_USD"].shift(1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'wti_activity(t-2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"WTI_OIL_PRICE_USD"].shift(2))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'wti_activity(t-3)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"WTI_OIL_PRICE_USD"].shift(3))

            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'wti_activity(t+1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"WTI_OIL_PRICE_USD"].shift(-1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'wti_activity(t+2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"WTI_OIL_PRICE_USD"].shift(-2))
            
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'btc_activity(t-1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"BC_OIL_PRICE_USD"].shift(1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'btc_activity(t-2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"BC_OIL_PRICE_USD"].shift(2))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'btc_activity(t-3)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"BC_OIL_PRICE_USD"].shift(3))

            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'btc_activity(t+1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"BC_OIL_PRICE_USD"].shift(-1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'btc_activity(t+2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"BC_OIL_PRICE_USD"].shift(-2))

            
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'rev_activity(t-1)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"rev"].shift(1))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'rev_activity(t-2)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"rev"].shift(2))
            X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,'rev_activity(t-3)'] = list(X_agg.loc[X_agg["SL BASIN (CODE)"]==basin,"rev"].shift(3))


#         X_agg["quarter_flag"] = 0
#         X_agg.loc[X_agg.index.month%3==0,'quarter_flag']=1
        X_agg = X_agg[(X_agg.index>self.buffer_start_date)&(X_agg.index<self.buffer_end_date)]
        
        X_agg.loc[X_agg.index>=self.pickup_date,["CM","REV"]] = np.nan

        inference_df = X_agg[(X_agg.index>self.start_date)&(X_agg.index<=self.end_date)]
        inference_df = inference_df.fillna(0).dropna()

        return inference_df
    
    @staticmethod
    def prophet_transformer(inference_df,oil_df):
        pr_infer_df = inference_df.reset_index()
        pr_infer_df = pr_infer_df[['quarter', 'SL BASIN (CODE)', 'c4c_rev', 'gac_rev', 'gaca_rev', 'cm', 'rev','quarter_flag']].drop_duplicates()
        pr_infer_df = pr_infer_df.merge(oil_df,on=["quarter"],how="left")
        return pr_infer_df



class InferenceConMarTrTransformer(TransformerMixin, BaseEstimator):

    def __init__(self,rev_data_dict,forecast_start_date):
        self.rev_data_dict = rev_data_dict
        self.forecast_start_date = forecast_start_date
        
    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X):
        adj_infer_df = pd.DataFrame(columns=['quarter', "SL BASIN (CODE)" ,'rev',
        'rev_activity(t-1)_', 'rev_activity(t-2)_', 'rev_activity(t-3)_'])

        og_infer_df = X.reset_index()
        
        # [['YEARMONTH', "SL BASIN (CODE)" ,'rev','rev_activity(t-1)', 'rev_activity(t-2)', 'rev_activity(t-3)','quarter_flag']]

        cols = ['quarter',"SL BASIN (CODE)",'rev_activity(t-1)', 'rev_activity(t-2)', 'rev_activity(t-3)',]
        
        for col in cols:
            og_infer_df[col] = og_infer_df[col].replace({0:np.nan})
            
        
        og_infer_df = og_infer_df.drop(["rev"],axis=1)
        X = X[X.index<self.forecast_start_date]
        X['quarter']=X.index

        for basin in self.rev_data_dict.keys():
            chip = self.rev_data_dict[basin].reset_index().copy()
            chip["SL BASIN (CODE)"] = basin


            div_infer_df = X[X['SL BASIN (CODE)']==basin]
            chip = chip.append(div_infer_df)
            chip = chip.sort_values(['quarter'])

            chip['rev_activity(t-1)_'] = chip["rev"].shift(1)
            chip['rev_activity(t-2)_'] = chip["rev"].shift(2)
            chip['rev_activity(t-3)_'] = chip["rev"].shift(3)

            chip = chip[chip['quarter']>=self.forecast_start_date]
            adj_infer_df = adj_infer_df.append(chip[['quarter', "SL BASIN (CODE)" ,'rev',
                'rev_activity(t-1)_', 'rev_activity(t-2)_', 'rev_activity(t-3)_']])

        print('self.forecast_start_date:',self.forecast_start_date)
        cmf_infer_df = adj_infer_df.merge(og_infer_df,on=["quarter","SL BASIN (CODE)"],how="inner")
        cmf_infer_df["rev_activity(t-1)"] = cmf_infer_df["rev_activity(t-1)_"].combine_first(cmf_infer_df["rev_activity(t-1)"])
        cmf_infer_df["rev_activity(t-2)"] = cmf_infer_df["rev_activity(t-2)_"].combine_first(cmf_infer_df["rev_activity(t-2)"])
        cmf_infer_df["rev_activity(t-3)"] = cmf_infer_df["rev_activity(t-3)_"].combine_first(cmf_infer_df["rev_activity(t-3)"])
        
        cmf_infer_df = cmf_infer_df.round(2)
        """
        cmf_infer_df = cmf_infer_df[['quarter', "SL BASIN (CODE)" ,'rev',
                'rev_activity(t-1)', 'rev_activity(t-2)', 'rev_activity(t-3)',
               'quarter_flag']].round(2)
        """
    
        cmf_infer_df.index = cmf_infer_df["quarter"]
        cmf_infer_df = cmf_infer_df.drop(["quarter"],axis=1)
        return cmf_infer_df

     
    
class InferenceConMarPrTransformer(TransformerMixin, BaseEstimator):

    def __init__(self,pr_df,basins,forecast_start_date):
        self.pr_df = pr_df
        self.basins = basins
        self.forecast_start_date = forecast_start_date
        
        
    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X):
        adj_infer_df = pd.DataFrame(columns=['quarter', "SL BASIN (CODE)" ,'rev',
        'rev_activity(t-1)_', 'rev_activity(t-2)_', 'rev_activity(t-3)_'])

        og_infer_df = X.reset_index()
        
        # [['YEARMONTH', "SL BASIN (CODE)" ,'rev','rev_activity(t-1)', 'rev_activity(t-2)', 'rev_activity(t-3)','quarter_flag']]

        cols = ['quarter',"SL BASIN (CODE)",'rev_activity(t-1)', 'rev_activity(t-2)', 'rev_activity(t-3)',]
        
        for col in cols:
            og_infer_df[col] = og_infer_df[col].replace({0:np.nan})
            
        
        og_infer_df = og_infer_df.drop(["rev"],axis=1)
        X = X[X.index<self.forecast_start_date]
        X['quarter']=X.index 

        for basin in self.basins:
            chip = self.pr_df[self.pr_df["SL BASIN (CODE)"]==basin].reset_index().copy()
            chip["SL BASIN (CODE)"] = basin
            
            div_infer_df = X[X['SL BASIN (CODE)']==basin]
            chip = chip.append(div_infer_df)
            chip = chip.sort_values(['quarter'])

            
            chip['rev_activity(t-1)_'] = chip["rev"].shift(1)
            chip['rev_activity(t-2)_'] = chip["rev"].shift(2)
            chip['rev_activity(t-3)_'] = chip["rev"].shift(3)
               
            chip = chip[chip['quarter']>=self.forecast_start_date]
            adj_infer_df = adj_infer_df.append(chip[['quarter', "SL BASIN (CODE)" ,'rev',
                'rev_activity(t-1)_', 'rev_activity(t-2)_', 'rev_activity(t-3)_']])


        cmf_infer_df = adj_infer_df.merge(og_infer_df,on=["quarter","SL BASIN (CODE)"],how="right")
        cmf_infer_df["rev_activity(t-1)"] = cmf_infer_df["rev_activity(t-1)_"].combine_first(cmf_infer_df["rev_activity(t-1)"])
        cmf_infer_df["rev_activity(t-2)"] = cmf_infer_df["rev_activity(t-2)_"].combine_first(cmf_infer_df["rev_activity(t-2)"])
        cmf_infer_df["rev_activity(t-3)"] = cmf_infer_df["rev_activity(t-3)_"].combine_first(cmf_infer_df["rev_activity(t-3)"])
        
        cmf_infer_df = cmf_infer_df.round(2)
        """
        cmf_infer_df = cmf_infer_df[['quarter', "SL BASIN (CODE)" ,'rev',
                'rev_activity(t-1)', 'rev_activity(t-2)', 'rev_activity(t-3)',
               'quarter_flag']].round(2)
        """
    
        cmf_infer_df.index = cmf_infer_df["quarter"]
        cmf_infer_df = cmf_infer_df.drop(["quarter"],axis=1)
        cmf_infer_df = cmf_infer_df.dropna()
        return cmf_infer_df
    