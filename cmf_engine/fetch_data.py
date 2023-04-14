# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 18:28:47 2023

@author: BGondaliya
"""



### To extract data

import pandas as pd


class TeleportData:
    def __init__(self,cnx,cnx_ods):
            self.cnx = cnx
            self.cnx_ods = cnx_ods


    def get_data(self):

            # C4C 
            c4c_query = """
            SELECT 
                    `YEARMONTH`,
                    `SL BASIN (CODE)`,
                    `SL DIVISION (CODE)`,
                    sum(`TOTAL_WEIGHTED_FCST`) as `TOTAL_WEIGHTED_FCST`

            FROM 
                    FACT_WEIGHTED_FORECAST_CMF
            WHERE `YEAR`>2016
            GROUP BY  `YEARMONTH`,`SL BASIN (CODE)`,`SL DIVISION (CODE)`;

            """

            # GAC
            gac_query = """
            SELECT
                    `YEARMONTH`,
                    `SL BASIN (CODE)`,
                    `RIG OPERATING ENVIRONMENT`,
                    `RIG_COUNT`,
                    `TOTAL_RIG_COUNT_BASIN`,
                    `SCENARIO`

            FROM
                    VW_RIG_COUNT
            """

            # HFM
            hfm_query = """
            SELECT
                    `YEARMONTH`,
                    SL_BASIN__CODE_ as `SL BASIN (CODE)`,
                    SL_DIVISION__CODE_ as `SL DIVISION (CODE)`,
                    `TOTAL`,
                    `SCENARIO`

            FROM
                    VW_HFM_CM

            """

            c4c_df = pd.read_sql(c4c_query,self.cnx)
            gac_df = pd.read_sql(gac_query,self.cnx)
            hfm_df = pd.read_sql(hfm_query,self.cnx)


            raw_df = TeleportData.combine_data (c4c_df, gac_df, hfm_df)

            return (raw_df)

    @staticmethod
    def combine_data(c4c_df, gac_df, hfm_df):

        #c4c_df = c4c_df.groupby(["YEARMONTH","SL BASIN (CODE)"])["TOTAL_WEIGHTED_FCST"].sum().reset_index()

        # C4C
        c4c_df['quarter'] = pd.PeriodIndex(c4c_df['YEARMONTH'], freq='Q').strftime('Q%q %Y')
        c4c_df = c4c_df.drop(columns='YEARMONTH')

        c4c_df = c4c_df.groupby(['SL BASIN (CODE)', 'SL DIVISION (CODE)','quarter'])['TOTAL_WEIGHTED_FCST'].sum().reset_index()
        c4c_global = c4c_df.groupby(["quarter","SL DIVISION (CODE)"])["TOTAL_WEIGHTED_FCST"].sum().reset_index()
        c4c_global["SL BASIN (CODE)"] = "Worldwide"

        c4c_df = c4c_df.append(c4c_global[c4c_df.columns])
        c4c_df = c4c_df.reset_index().drop(["index"],axis=1)
        c4c_df["TOTAL_WEIGHTED_FCST"] = c4c_df["TOTAL_WEIGHTED_FCST"]/1000
        
        
        # GAC
        gac_df['quarter'] = pd.PeriodIndex(gac_df['YEARMONTH'], freq='Q').strftime('Q%q %Y')
        gac_df = gac_df.drop(columns='YEARMONTH')

        gac_df = gac_df.groupby(["quarter","SL BASIN (CODE)","SCENARIO"])["TOTAL_RIG_COUNT_BASIN"].mean().reset_index()
        gac_df_wochina = gac_df.loc[gac_df["SCENARIO"]=="RIG COUNT WITHOUT CHINA",["quarter","SL BASIN (CODE)","TOTAL_RIG_COUNT_BASIN"]]
        gac_df_wochina.columns = ["quarter","SL BASIN (CODE)","TOTAL_RIG_COUNT_BASIN_woChina"]

        gac_df_wochina_global = gac_df_wochina.groupby(["quarter"])["TOTAL_RIG_COUNT_BASIN_woChina"].sum().reset_index()
        gac_df_wochina_global["SL BASIN (CODE)"] = "Worldwide"
        gac_df_wochina = gac_df_wochina.append(gac_df_wochina_global[gac_df_wochina.columns])
        gac_df_wochina = gac_df_wochina.reset_index().drop(["index"],axis=1)


        gac_df_all = gac_df.loc[gac_df["SCENARIO"]=="RIG COUNT",["quarter","SL BASIN (CODE)","TOTAL_RIG_COUNT_BASIN"]]
        gac_df_all.columns = ["quarter","SL BASIN (CODE)","TOTAL_RIG_COUNT_BASIN_all"]

        gac_df_all_global = gac_df_all.groupby(["quarter"])["TOTAL_RIG_COUNT_BASIN_all"].sum().reset_index()
        gac_df_all_global["SL BASIN (CODE)"] = "Worldwide"
        gac_df_all = gac_df_all.append(gac_df_all_global[gac_df_all.columns])
        gac_df_all = gac_df_all.reset_index().drop(["index"],axis=1)
        
        
        # HFM

        hfm_df['quarter'] = pd.PeriodIndex(hfm_df['YEARMONTH'], freq='Q').strftime('Q%q %Y')
        hfm_df = hfm_df.drop(columns='YEARMONTH')

        hfm_df = hfm_df.groupby(["quarter","SL BASIN (CODE)","SL DIVISION (CODE)","SCENARIO"])["TOTAL"].sum().reset_index()         

        hfm_df_cm = hfm_df.loc[hfm_df["SCENARIO"]=="CONTRIBUTION MARGIN(KUSD)",["quarter","SL BASIN (CODE)","SL DIVISION (CODE)","TOTAL"]]
        hfm_df_cm.columns = ["quarter","SL BASIN (CODE)","SL DIVISION (CODE)","CM"]


        hfm_df_cm_global = hfm_df_cm.groupby(["quarter","SL DIVISION (CODE)"])["CM"].sum().reset_index()
        hfm_df_cm_global["SL BASIN (CODE)"] = "Worldwide"
        hfm_df_cm = hfm_df_cm.append(hfm_df_cm_global[hfm_df_cm.columns])
        hfm_df_cm = hfm_df_cm.reset_index().drop(["index"],axis=1)

        hfm_df_rev = hfm_df.loc[hfm_df["SCENARIO"]=="TOTAL REVENUE(KUSD)",["quarter","SL BASIN (CODE)","SL DIVISION (CODE)","TOTAL"]]
        hfm_df_rev.columns = ["quarter","SL BASIN (CODE)","SL DIVISION (CODE)","REV"]

        hfm_df_rev_global = hfm_df_rev.groupby(["quarter","SL DIVISION (CODE)"])["REV"].sum().reset_index()
        hfm_df_rev_global["SL BASIN (CODE)"] = "Worldwide"
        hfm_df_rev = hfm_df_rev.append(hfm_df_rev_global[hfm_df_rev.columns])
        hfm_df_rev = hfm_df_rev.reset_index().drop(["index"],axis=1)

        raw_df = c4c_df.merge(gac_df_wochina,on=["quarter","SL BASIN (CODE)"],how="inner").merge(gac_df_all,on=["quarter","SL BASIN (CODE)"],how="inner")
        raw_df = raw_df.merge(hfm_df_cm,on=["quarter","SL BASIN (CODE)","SL DIVISION (CODE)"],how="outer").merge(hfm_df_rev,on=["quarter","SL BASIN (CODE)","SL DIVISION (CODE)"],how="outer")


        return raw_df


    @staticmethod
    def write_data(cxn_ods,table_name,df,if_exists):
        df.to_sql(con=cxn_ods, name=table_name,if_exists = if_exists, index=None)

    @staticmethod
    def get_eia_data(cxn_ods):
        oil_query = """ SELECT * FROM ODS.eia_oil_prices """
        oil_df = pd.read_sql(oil_query,cxn_ods)
        oil_df['quarter'] = pd.PeriodIndex(oil_df['YEARMONTH'], freq='Q').strftime('Q%q %Y')
        oil_df = oil_df.drop(columns='YEARMONTH')
        
        oil_df = oil_df.groupby(["quarter"])["BC_OIL_PRICE_USD","BC_OIL_PRICE_USD"].mean().reset_index() 
        return oil_df

