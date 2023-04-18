# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 18:55:42 2023

@author: BGondaliya
"""




"inference module"
import gzip
import os
import pickle
import pickletools
import datetime

import pandas as pd
from prophet.serialize import model_from_json, model_to_json
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
# Neural Prophet 0.4.2
from neuralprophet import NeuralProphet,utils


class RfModel:
    def __init__(self,ind_features,dep_var,basins,cutoff_date,model_path):
        self.ind_features = ind_features
        self.dep_var = dep_var
        self.basins = basins
        self.cutoff_date = pd.to_datetime(cutoff_date, format="%Y-%m-%d")
        self.cutoff_date = cutoff_date.strftime('%Y Q{}'.format( (cutoff_date.month - 1) // 3 + 1))
        self.model_path = os.path.join(model_path,str(self.cutoff_date))


    def train(self,X):
        X = X.dropna()
        for basin in tqdm(self.basins):
            
            

            
            in_train = X[X.index<self.cutoff_date]

            # Weightage Mechanism
            most_recent_date = in_train.index.max()
            most_recent_date = pd.Timestamp(datetime.datetime.strptime(most_recent_date, '%YQ%m'))


            in_train.index = pd.DatetimeIndex([pd.Timestamp(datetime.datetime.strptime(d, '%YQ%m')) for d in in_train.index])

            days_before_recent_date = (most_recent_date - in_train.index).days
            print('days_before_recent_date:',days_before_recent_date)
            in_train["days_elapsed"] = days_before_recent_date
            in_train["weight"]=0.5
            in_train.loc[in_train["days_elapsed"]<90,'weight'] =1
            in_train.loc[(in_train["days_elapsed"]>=90)*(in_train["days_elapsed"]<270),'weight'] = 0.9
            in_train.loc[(in_train["days_elapsed"]>=270)*(in_train["days_elapsed"]<360),'weight'] = 0.8
            in_train.loc[(in_train["days_elapsed"]>=360)*(in_train["days_elapsed"]<540),'weight'] = 0.7
            
            weights = in_train.loc[in_train["SL BASIN (CODE)"]==basin,"weight"]
            X_train = in_train.loc[in_train["SL BASIN (CODE)"]==basin,self.ind_features]
            yt_train = in_train.loc[in_train["SL BASIN (CODE)"]==basin,self.dep_var]
            

            # Train 
            regressor = RandomForestRegressor(n_estimators=100,verbose=1,
                                              random_state=42
                                    
                                             )

            regressor.fit(X=X_train,y=yt_train)#,sample_weight = weights)


            # Saving model
            RfModel.save_model(self.model_path,regressor,basin)

            del regressor



    def infer(self,X):

        model_dict = RfModel.load_model(self.model_path,self.basins)
        data_dict = {}

        for basin in model_dict.keys():
            regressor = model_dict[basin]
            print("Predicting Basin --> {}".format(basin))
            infer_df = X.loc[X["SL BASIN (CODE)"]==basin,self.ind_features]
            #print(infer_df,list(regressor.predict(infer_df)),len(regressor.predict(infer_df)))
            infer_df.loc[:,self.dep_var] = regressor.predict(infer_df)
            data_dict[basin]  = infer_df


        return(data_dict)



    @staticmethod
    def save_model(path,artifact,model_name):
        directory = os.path.join(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        
        filepath = os.path.join(directory,model_name+'_regressor.pkl.gz')
        with gzip.open(filepath, "wb") as f:
            pickled = pickle.dumps(artifact)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)
        
        f.close()



    @staticmethod
    def load_model(path,basins):
        model_dict = {}
        for basin in basins:

            filepath = os.path.join(path,basin+'_regressor.pkl.gz')
            with gzip.open(filepath, "rb") as f:
                p = pickle.Unpickler(f)
                regressor = p.load()

            model_dict[basin] = regressor
            f.close()

        return model_dict
    
    
class ProphetModel:
    def __init__(self,ind_features,ind_features_basin,dep_var,basin,cutoff_date,model_path):
        self.ind_features = ind_features
        self.ind_features_basin = ind_features_basin
        self.dep_var = dep_var
        self.basin = basin
        self.cutoff_date = pd.to_datetime(cutoff_date, format="%Y-%m-%d")
        self.cutoff_date = cutoff_date.strftime('%Y Q{}'.format( (cutoff_date.month - 1) // 3 + 1))
        self.model_path = os.path.join(model_path,str(self.cutoff_date))


    def train(self,X):
        pr_train = X.rename({'quarter':'ds',self.dep_var[0]:'y'},axis='columns')
        pr_train = pr_train.dropna()
        
        for basin in self.basin:
            X_train = pr_train[pr_train["SL BASIN (CODE)"]==basin]
            
            if basin=="Worldwide":
                future_regressors = self.ind_features
            else:
                future_regressors = self.ind_features_basin
                
                
            prophet_model = Prophet(interval_width=0.95,weekly_seasonality=True,daily_seasonality=True)
            for reg in future_regressors:
                prophet_model.add_regressor(reg)


            prophet_model.fit(X_train[["ds","y"]+future_regressors])
            ProphetModel.save_model(self.model_path,prophet_model,basin)
            del prophet_model


    def infer(self,X):
        
        
        infer_df = X.copy()
        prophet_model_dict  = ProphetModel.load_model(self.model_path,self.basin)
        pr_infer = infer_df.rename({'quarter':'ds'},axis='columns')
        infer_df[self.dep_var[0]] = 0
        
        for basin in self.basin:
            
            X_infer = pr_infer[pr_infer["SL BASIN (CODE)"]==basin]
            
            if basin=="Worldwide":
                future_regressors = self.ind_features
            else:
                future_regressors = self.ind_features_basin
                
            
            X_infer = X_infer[["ds"]+future_regressors]
                
            forecast = prophet_model_dict[basin].predict(X_infer.fillna(0))
            infer_df.loc[infer_df["SL BASIN (CODE)"]==basin, self.dep_var[0]] = list(forecast["yhat"])
            
            del forecast
        
        return infer_df



    @staticmethod
    def save_model(path,artifact,model_name):
        directory = os.path.join(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        
        filepath = os.path.join(directory,model_name+'_prophet.json')
        with open(filepath, "w") as f:
            f.write(model_to_json(artifact))
        
        f.close()



    @staticmethod
    def load_model(path,basins):
        pr_model_dict={}
        
        for basin in basins:
            
            filepath = os.path.join(path,basin+'_prophet.json')
            with open(filepath, "r") as f:
                pr_model_dict[basin] = model_from_json(f.read())

            f.close()
            
            
        return pr_model_dict
    
    
class NeuralProphetModel:
    def __init__(self,ind_features,ind_features_basin,dep_var,lagged_features,basin,cutoff_date,model_path):
        self.ind_features = ind_features
        self.ind_features_basin = ind_features_basin
        self.dep_var = dep_var
        self.lagged_features = lagged_features
        self.basin = basin
        self.cutoff_date = pd.to_datetime(cutoff_date, format="%Y-%m-%d")
        self.cutoff_date = cutoff_date.strftime('%Y Q{}'.format( (cutoff_date.month - 1) // 3 + 1))
        self.model_path = os.path.join(model_path,str(self.cutoff_date))

    def train(self,X):
        utils.set_random_seed(seed=0)


        npr_train = X.rename({'quarter':'ds',self.dep_var:'y'},axis='columns')
        npr_train = npr_train.dropna()


        
        for basin in self.basin:
            
            print(basin,"--------------> Regressors")
            X_train = npr_train[npr_train["SL BASIN (CODE)"]==basin].copy()
            
            if basin=="Worldwide":
                future_regressors = self.ind_features
                print(future_regressors,"Future Regressors")
            else:
                future_regressors = self.ind_features_basin
                print(future_regressors,"Future Regressors")

                
            neural_prophet_model = NeuralProphet( growth="discontinuous", weekly_seasonality=False,daily_seasonality=False,yearly_seasonality=False,
                      n_lags = 4, n_forecasts=4, num_hidden_layers=8, d_hidden = 48,drop_missing=True)
            
            if len(self.lagged_features)>0:
                neural_prophet_model = neural_prophet_model.add_lagged_regressor(names=self.lagged_features)

            for reg in future_regressors:
                neural_prophet_model = neural_prophet_model.add_future_regressor(name = reg)


            neural_prophet_model.fit(X_train[["ds","y"]+future_regressors+self.lagged_features])

            NeuralProphetModel.save_model(self.model_path,neural_prophet_model,basin)
            
            del neural_prophet_model

        return npr_train


    def infer(self,X,X_dash):
        
        infer_df = X.copy()

        nprophet_model_dict  = NeuralProphetModel.load_model(self.model_path,self.basin)
        
        npr_infer = infer_df.rename({'quarter':'ds'},axis='columns')
        npr_train = X_dash.copy()

        infer_df[self.dep_var] = 0
        
        for basin in self.basin:
            print(basin,"--------------> Regressors")

            nprophet_model = nprophet_model_dict[basin]

            X_infer = npr_infer[npr_infer["SL BASIN (CODE)"]==basin].copy()
            X_train = npr_train[npr_train["SL BASIN (CODE)"]==basin].copy()
            
            if basin=="Worldwide":
                future_regressors = self.ind_features
                print(future_regressors,"Future Regressors")
            else:
                future_regressors = self.ind_features_basin
                print(future_regressors,"Future Regressors")


            X_infer_dash = nprophet_model.make_future_dataframe(X_train[["ds","y"]+future_regressors+self.lagged_features],
                                   regressors_df=X_infer[future_regressors],
                                   periods = 6)
                
                            
            forecast = nprophet_model.predict(X_infer_dash)
            out = nprophet_model.get_latest_forecast(forecast)


            infer_df.loc[infer_df["SL BASIN (CODE)"]==basin, self.dep_var] = list(out["origin-0"])
            
            del forecast, nprophet_model, X_infer, X_train, X_infer_dash
        
        return infer_df




    @staticmethod
    def save_model(path,artifact,model_name):
        directory = os.path.join(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        
        filepath = os.path.join(directory,model_name+'_neuralprophet.pkl.gz')
        with gzip.open(filepath, "wb") as f:
            pickled = pickle.dumps(artifact)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)
        
        f.close()



    @staticmethod
    def load_model(path,basins):
        model_dict = {}
        for basin in basins:

            filepath = os.path.join(path,basin+'_neuralprophet.pkl.gz')
            with gzip.open(filepath, "rb") as f:
                p = pickle.Unpickler(f)
                regressor = p.load()

            model_dict[basin] = regressor
            f.close()

        return model_dict
 
    