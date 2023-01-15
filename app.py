from flask import Flask,jsonify,request,render_template
from project_data.utils import adult
import config

app=Flask(__name__)
@app.route("/")
def home():
    print("Welcome to Adult Cencus Analysis")
    return render_template("home.html")

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        data = request.form
        print(f"Data is: {data}")

        age          =   eval(data['age'])   
        workclass    =   data['workclass']             
        fnlwgt       =   eval(data['fnlwgt'])                
        education    =   data['education']                        
        education_num=   eval(data['education_num'])                            
        race         =   data['race']                       
        sex          =   data['sex']                       
        capital_gain =   eval(data['capital_gain'])
        capital_loss =   eval(data['capital_loss'])
        hours_per_week=  eval(data['hours_per_week'])
        occupation   =   data['occupation']
        country      =   data['country'] 

        obj = adult(age,workclass,fnlwgt,education,education_num,occupation,race,sex,capital_gain,capital_loss,hours_per_week,country)
        result = obj.predict_salary()              
        print("Result is: ",result)
        return render_template("after.html", data=result)
        # return render_template("after.html", data=result) 
        #
if __name__=="__main__":
    app.run(host="0.0.0.0",port=config.PORT_NUMBER,debug=True)       
