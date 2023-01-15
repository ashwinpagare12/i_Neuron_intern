import numpy as np
import pandas as pd
import pickle
import json
import os
import config

class adult():
    def __init__(self,age,workclass,fnlwgt,education,education_num,occupation,race,sex,capital_gain,capital_loss,hours_per_week,country):
        self.age             =  age
        self.workclass       =  workclass
        self.fnlwgt          =  fnlwgt
        self.education       =  education
        self.education_num   =  education_num
        self.occupation      =  'occupation_' + occupation
        self.race            =  race
        self.sex             =  sex
        self.capital_gain    =  capital_gain
        self.capital_loss    =  capital_loss
        self.hours_per_week  =  hours_per_week
        self.country         =  'country_' + country

    def load_model(self):
        with open(config.MODEL_FILE_PATH, "rb") as f:
            self.adaboost_model=pickle.load(f)
        with open(config.SCALING_FILE_PATH,'rb') as f:
            self.scalling_model = pickle.load(f)
        with open(config.JSON_FILE_PATH,"r") as f:
            self.json_data = json.load(f)
    
    def predict_salary(self):
        self.load_model()

        occupation_index = self.json_data["columns"].index(self.occupation)
        country_index = self.json_data["columns"].index(self.country)
        test_array=np.zeros(len(self.json_data["columns"]))
        
        test_array[0] =  self.age
        test_array[1] =  self.json_data["workclass"][self.workclass]
        test_array[2] =  self.fnlwgt
        test_array[3] =  self.json_data["education"][self.education]
        test_array[4] =  self.education_num
        test_array[5] =  self.json_data["race"][self.race]
        test_array[6] =  self.json_data["sex"][self.sex]
        test_array[7] =  self.capital_gain
        test_array[8] =  self.capital_loss
        test_array[9] =  self.hours_per_week
        test_array[occupation_index] =1
        test_array[country_index] =1
        print("test_array is: ",test_array)
        scale_test_array=self.scalling_model.transform([test_array])
        prediction=self.adaboost_model.predict(scale_test_array)[0]
        return prediction

if __name__ == "__main__":
    age          =   25   
    workclass    =   'Local-gov'             
    fnlwgt       =   65098                 
    education    =   'Assoc-acdm'                        
    education_num=   15                             
    race         =   'Asian-Pac-Islander'                       
    sex          =   'Male'                       
    capital_gain =   2020 
    capital_loss =   812
    hours_per_week=  55
    occupation   =   'Transport-moving'
    country      =   'India'               

    obj=adult(age,workclass,fnlwgt,education,education_num,occupation,race,sex,capital_gain,capital_loss,hours_per_week,country)
    obj.predict_salary()

    # salary = ?


