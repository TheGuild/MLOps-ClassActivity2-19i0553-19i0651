#imports
import os
import mlflow
from concurrent.futures import ThreadPoolExecutor
from pmdarima import auto_arima
import csv
import ctypes
import string
import datefinder
import matplotlib
import numpy as np
from dateutil.parser import parse
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima import ADFTest
import requests
import re
import json
import pickle

class ProductCatalouge:
    def __init__(self):
        self.Mobiles = []
        self.Laptops = []
        self.HomeAppliances = []
        self.FutureMobiles = []
        self.FutureLaptops = []
        self.FutureHomeAppliances = []
        self.LoadDataforAnalysis()
        self.RunForecastModels()

    def LoadDataforAnalysis(self):
        #load data from csv file of each product one by one
        path = "cleaned_Laptops.csv"
        data = pd.read_csv(path)
        data.set_index('Date', inplace=True)
        data.index = pd.to_datetime(data.index)
        result = data.resample('1M').count()
        Temp = result['Prices'].values.tolist()
        print("Laptops",Temp)
        self.Laptops = Temp[-12:]
        #for mobiles
        path = "cleaned_Mobiles.csv"
        data = pd.read_csv(path)
        data.set_index('Date', inplace=True)
        data.index = pd.to_datetime(data.index)
        #total sales counts of each month
        result = data.resample('1M').count()
        Temp = result['Prices'].values.tolist()
        print("Mobiles",Temp)
        self.Mobiles = Temp[-12:]
        #for home appliances
        path = "cleaned_Home Appliances.csv"
        data = pd.read_csv(path)
        data.set_index('Date', inplace=True)
        data.index = pd.to_datetime(data.index)
        result = data.resample('1M').count()
        Temp = result['Prices'].values.tolist()
        print("Home Appliances",Temp)
        self.HomeAppliances = Temp[-12:]
        return 1

    def RunForecastModels(self):
        with mlflow.start_run(run_name="Forecast Models"):
            self.FutureMobiles = self.getforecast("Mobiles")
            self.FutureLaptops = self.getforecast("Laptops")
            self.FutureHomeAppliances = self.getforecast("Home Appliances")

            # Save the Laptop forecast model to a file
            with open('auto_arima_model.pkl', 'wb') as pkl_file:
                pickle.dump(self.FutureLaptops, pkl_file)

            # Log the forecasts as metrics
            mlflow.log_metric("Mobiles Forecast", self.FutureMobiles[-1])
            mlflow.log_metric("Laptops Forecast", self.FutureLaptops[-1])
            mlflow.log_metric("Home Appliances Forecast", self.FutureHomeAppliances[-1])
            mlflow.log_artifact()
        return 1


    def getSalesInsights(self, productname):
        pricelist = []
        responsedata={}
        labels = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if productname == "Mobiles":
            pricelist = self.Mobiles
            pricelist = pricelist[-9:]
        elif productname == "Laptops":
            pricelist = self.Laptops
            pricelist = pricelist[-9:]
        elif productname == "Home Appliances":
            pricelist = self.HomeAppliances
            pricelist = pricelist[-9:]
        datasets = {"label": productname, "data": pricelist}
        responsedata = {"labels": labels, "datasets": datasets}
        return responsedata

    def getforecast(self, productname):
        pricelist = []
        if productname != None:
            if productname == "Mobiles":
                pricelist = self.Mobiles
                pricelist = pricelist[-12:]
            elif productname == "Laptops":
                pricelist = self.Laptops
                pricelist = pricelist[-12:]
            elif productname == "Home Appliances":
                pricelist = self.HomeAppliances
                pricelist = pricelist[-12:]
            data = pd.DataFrame(pricelist, columns=['Monthly Prices'])
            # predict next month sale based on previous 12 months data using auto arima model
            model = auto_arima(data, start_p=1, start_q=1,
                               test='adf',
                               max_p=3, max_q=3, m=12,
                               start_P=0, seasonal=False,
                               d=1, D=1, trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)
            model.fit(data)
            future_forecast = model.predict(n_periods=9)
            #print(future_forecast)
            prediction = pd.DataFrame(future_forecast, columns=['Prediction'])
            pricelist.pop(0)
            pricelist.append(int(prediction["Prediction"].iloc[0]))
            pricelist=pricelist[-9:]
            #append first value of future forecast to pricelist
            print(pricelist)
            return pricelist

    def getSalesforecast(self, productname):
        pricelist = []
        if productname != None:
            if productname == "Mobiles":
                pricelist = self.FutureMobiles
            elif productname == "Laptops":
                pricelist = self.FutureLaptops
            elif productname == "Home Appliances":
                pricelist = self.FutureHomeAppliances
        responsedata={}
        labels = ["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan"]
        datasets = {"label": productname, "data": pricelist}
        responsedata = {"labels": labels, "datasets": datasets}
        return responsedata


catalogue = ProductCatalouge()
sales_forecast = catalogue.getSalesforecast("Laptops")
print(sales_forecast)
