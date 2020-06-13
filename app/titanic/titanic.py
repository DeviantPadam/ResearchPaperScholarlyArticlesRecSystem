#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:37:02 2020

@author: deviantpadam
"""

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

from flask import render_template,request
import random
from app.titanic import titanic_app
import pandas as pd
import pickle


model = None

model = pickle.load(open('model/titanicModel.pkl','rb'))


@titanic_app.route('/titanic',methods=['GET','POST'])
def titanic():
    fact_num = random.randint(1,10)
    title = "Titanic Survival Checker"
    return render_template('titanic.html',fact_num=fact_num,title = title)


@titanic_app.route('/result',methods=['GET','POST'])
def result():
    
    global model
    fact_num = random.randint(1,10)

    if request.method == 'POST':
        columns = ['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
        col_dict = {i:[] for i in columns}
        
        col_dict['Pclass'].append(int(request.form['pclass']))
        col_dict['Name'].append('random, '+request.form['title']+' '+request.form['name'])
        
        if request.form['title'] in ['Mr.','Master']:
            col_dict['Sex'].append('male')
        elif request.form['title'] in ['Mrs.','Miss']:
            col_dict['Sex'].append('female')
        else:
            col_dict['Sex'].append(request.form['sex'])
            
        col_dict['Age'].append(int(request.form['age']))
        
        if request.form['alone']=='1':
            col_dict['SibSp'].append(0)
            col_dict['Parch'].append(0)
        else:
            col_dict['SibSp'].append(int(request.form['family'])//2)
            col_dict['Parch'].append(int(request.form['family'])//2)
            
        col_dict['Ticket'].append('randomTicket')
        col_dict['Fare'].append(int(request.form['fare']))
        
        if int(request.form['fare'])>4350:
            col_dict['Cabin'].append('C97')
        else:
            col_dict['Cabin'].append(float('NaN'))
            
        col_dict['Embarked'].append(request.form['embark'])
        
        prob=model.predict_proba(pd.DataFrame(col_dict))

        result = str((prob[0][1])*100)+'%'
        
        if prob[0][1]>0.5:
        	title="Congrats You survived!!"
        else:
        	title="Hehe Dead"
        return render_template('result.html',result=result,fact_num=fact_num,title=title)


        
        
   
