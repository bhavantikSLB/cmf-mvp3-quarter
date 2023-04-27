# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:34:21 2023

@author: BGondaliya
"""

import pandas as pd
import numpy as np
import dateutil.parser as dparser

class postprocess:
    def __init__(self,refresh_date, cxn_ODS):
        self.refresh_date = refresh_date
        self.cxn_ODS = cxn_ODS
        
    def post_process(self, train_df, data_dict, model, multi_model_df, ground_truth):
        data_dict["model"] = model
        columns_out = ["quarter","SL BASIN (CODE)","cm","rev","model"]
        

        multi_model_df = multi_model_df.append(data_dict[columns_out])

        multi_model_df = multi_model_df.reset_index().drop(["index"],axis=1)
       # multi_model_df['quarter'] = pd.to_datetime(multi_model_df['quarter'])
        multi_model_df = multi_model_df.merge(ground_truth,on=["quarter","SL BASIN (CODE)"],how="left")
        multi_model_df["contribution_margin_error"] = np.round(((multi_model_df["cm"]-multi_model_df["actual_contribution_margin"])/multi_model_df["actual_contribution_margin"]),2)*100
        multi_model_df["revenue_error"] = np.round(((multi_model_df["rev"]-multi_model_df["actual_revenue"])/multi_model_df["actual_revenue"]),2)*100
        multi_model_df["YM"] = list(map(lambda x: dparser.parse(str(x),fuzzy=True).strftime('%b\'%y'), multi_model_df["quarter"]))
        multi_model_df["refresh_date"] = self.refresh_date
        multi_model_df["level"] = multi_model_df["SL BASIN (CODE)"]
        multi_model_df = multi_model_df.round(2) 


        multi_model_df[multi_model_df["SL BASIN (CODE)"]=="Worldwide"].sort_values(["SL BASIN (CODE)","quarter","model"]).dropna()
        multi_model_df = multi_model_df.drop_duplicates()
        multi_model_df.to_sql(con=self.cxn_ODS, name='cmf3_1_mvp3_Q1',if_exists = 'append',index=None)
       # print(" Written for Model - {}".format(, model))

