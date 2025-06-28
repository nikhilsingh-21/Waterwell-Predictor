from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import numpy as np
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import pickle
from math import sqrt
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


global uname

analyzer = SentimentIntensityAnalyzer()
rmse = []
mae = []

def calculateMetrics(algorithm, predict, test_labels):
    mse_value = sqrt(mean_squared_error(test_labels, predict))
    score = mean_absolute_error(np.asarray(test_labels), np.asarray(predict))
    rmse.append(mse_value)
    mae.append(score)

water_bearing = pd.read_csv("Dataset/District_Statewise_Well.csv")
water_bearing.fillna(0, inplace = True)
water_bearing_Y = water_bearing['Stage of Ground Water Extraction (%)'].ravel()
water_bearing.drop(['S.no.', 'Stage of Ground Water Extraction (%)'], axis = 1,inplace=True)
le = LabelEncoder()
water_bearing['Name of State'] = pd.Series(le.fit_transform(water_bearing['Name of State'].astype(str)))#encode all str columns to numeric
le1 = LabelEncoder()
water_bearing['Name of District'] = pd.Series(le1.fit_transform(water_bearing['Name of District'].astype(str)))#encode all str columns to numeric
water_bearing_X = water_bearing.values
water_bearing_Y = water_bearing_Y.reshape(-1, 1)
scaler2 = MinMaxScaler(feature_range = (0, 1))
scaler3 = MinMaxScaler(feature_range = (0, 1))
water_bearing_X = scaler2.fit_transform(water_bearing_X)#normalize train features
water_bearing_Y = scaler3.fit_transform(water_bearing_Y)
X_train, X_test, y_train, y_test = train_test_split(water_bearing_X, water_bearing_Y, test_size = 0.2)
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train.ravel())
predict = dt.predict(X_test)
calculateMetrics("Decision Tree Calssifier", predict, y_test)

rf = RandomForestRegressor()
rf.fit(X_train, y_train.ravel())
predict = rf.predict(X_test)
calculateMetrics("Random Forest Classifier", predict, y_test)

current_water = pd.read_csv("Dataset/GWLevel_20240521213900.csv", usecols=['Reference_Point_Elevation','Ground_Surface_Elevation',
                                                                           'Distance_from_RP_to_WS','Groundwater_Elevation','Ground_Surface_to_Water Surface'])

current_water.fillna(0, inplace = True)
current_Y = current_water['Ground_Surface_to_Water Surface'].ravel()
current_water.drop(['Ground_Surface_to_Water Surface'], axis = 1,inplace=True)

current_X = current_water.values
current_Y = current_Y.reshape(-1, 1)

scaler = MinMaxScaler(feature_range = (0, 1))
scaler1 = MinMaxScaler(feature_range = (0, 1))
current_X = scaler.fit_transform(current_X)#normalize train features
current_Y = scaler1.fit_transform(current_Y)

X_train, X_test, y_train, y_test = train_test_split(current_X, current_Y, test_size = 0.1)

dt1 = DecisionTreeRegressor()
dt1.fit(X_train, y_train.ravel())
predict = dt1.predict(X_test)
calculateMetrics("Decision Tree Regressor", predict, y_test)

rf1 = RandomForestRegressor()
rf1.fit(X_train, y_train.ravel())
predict = rf1.predict(X_test)
calculateMetrics("Random Forest Regressor", predict, y_test)

def TrainModels(request):
    if request.method == 'GET':
        global rmse, mae
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">RMSE</th>'
        output += '<th><font size="" color="black">MAE</th>'
        output+='</tr>'
        algorithms = ['Decision Tree Classifier', 'Random Forest Classifier', 'Decision Tree Regressor', 'Random Forest Regressor']
        for i in range(len(rmse)):
            output += '<tr><td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(rmse[i])+'</td>'
            output += '<td><font size="" color="black">'+str(mae[i])+'</td></tr>'
        output+= "</table></br></br></br></br>"
        context= {'data':output}
        return render(request, 'ViewResult.html', context)

def Clustering(request):
    if request.method == 'GET':
        global water_bearing_X
        pca = PCA(2) 
        X = pca.fit_transform(water_bearing_X)
        kmeans = KMeans(n_clusters=2).fit(X)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        unique_cluster = np.unique(labels) 
        plt.figure(figsize=(7, 7))
        class_labels = ['Less Water', 'High Water']
        for cls in unique_cluster:
            plt.scatter(X[labels == cls, 0], X[labels == cls, 1], label=class_labels[cls]) 
        plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=169,linewidths=3,color='k',zorder=10) 
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        context= {'data':"Kmeans Clustering with Low & High Water", 'img': img_b64}
        return render(request, 'ViewResult.html', context)

def Visualization(request):
    if request.method == 'GET':
        figure, axis = plt.subplots(nrows=2, ncols=2,figsize=(14,12))
        df = pd.read_csv("Dataset/District_Statewise_Well.csv")
        df.groupby(['Name of District', 'Current Annual Ground Water Extraction For Domestic & Industrial Use']).mean()
        df = df.sort_values('Current Annual Ground Water Extraction For Domestic & Industrial Use', ascending=False)
        df = df.head(10)
        df = df.rename(columns={'Current Annual Ground Water Extraction For Domestic & Industrial Use': 'Ground Water Extraction'})
        df.plot.pie(y='Ground Water Extraction', labels = df['Name of District'], autopct='%1.1f%%', ax=axis[0,0])
        axis[0,0].get_legend().remove()
        axis[0,0].set_title('Top 10 Districts Annual Groundwater for Domestic & Industrial Use')
        df = pd.read_csv("Dataset/District_Statewise_Well.csv")
        bins = [0, 7000, 14000, 21000, 28000, 35000]
        df['binned'] = pd.cut(df['Current Annual Ground Water Extraction For Domestic & Industrial Use'], bins)
        df = df.groupby('binned').mean().reset_index()
        df = df.sort_values('Current Annual Ground Water Extraction For Domestic & Industrial Use', ascending=False)
        df = df.rename(columns={'Current Annual Ground Water Extraction For Domestic & Industrial Use': 'Annual Usage Extraction'})
        df.plot.pie(y='Annual Usage Extraction', labels = df['binned'], autopct='%1.1f%%', ax=axis[0,1])
        axis[0,1].set_title('Distribution Annual Groundwater for Domestic & Industrial Use')
        axis[0,1].get_legend().remove()
        df = pd.read_csv("Dataset/District_Statewise_Well.csv")
        df = df.groupby('Name of State').mean().reset_index()
        df = df.sort_values('Current Annual Ground Water Extraction For Domestic & Industrial Use', ascending=False)
        df = df.head(20)
        df = df[::-1]
        df = df.rename(columns={'Current Annual Ground Water Extraction For Domestic & Industrial Use': 'Ground Water Extraction'})
        df.plot('Name of State','Ground Water Extraction',kind="bar", ax=axis[1,0])
        axis[1,0].set_title('State Wise Annual Groundwater Extraction Domestic & Industrial Use')
        df = pd.read_csv("Dataset/District_Statewise_Well.csv")
        df = df.sort_values('Total Annual Ground Water Recharge', ascending=True)
        axis[1,1].plot(df['Total Annual Ground Water Recharge'], df['Recharge from rainfall During Monsoon Season'])
        axis[1,1].set_xlabel('Total Annual Ground Water Recharge')
        axis[1,1].set_ylabel('Recharge from rainfall During Monsoon Season')
        axis[1,1].set_title("Recharge from rainfall from Monsoon VS Total Annual GroudWater")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        context= {'data':"Visualization Dashboard", 'img': img_b64}
        return render(request, 'ViewResult.html', context)

def WaterBearing(request):
    if request.method == 'GET':
        df = pd.read_csv("Dataset/District_Statewise_Well.csv")
        states = np.unique(df['Name of State'].ravel())
        district = np.unique(df['Name of District'].ravel())
        output = '<tr><td><font size="" color="black">Name&nbsp;of&nbsp;State</b></td><td><select name="t1">'
        for i in range(len(states)):
            output += '<option value="'+states[i]+'">'+states[i]+'</option>'
        output += '</select></td></tr>'
        output += '<tr><td><font size="" color="black">Name&nbsp;of&nbsp;District</b></td><td><select name="t2">'
        for i in range(len(district)):
            output += '<option value="'+district[i]+'">'+district[i]+'</option>'
        output += '</select></td></tr>'
        context= {'data1': output}
        return render(request, 'WaterBearing.html', context)

def WaterBearingAction(request):
    if request.method == 'POST':
        global uname, analyzer, scaler2, scaler3, dt, le, le1
        v1 = request.POST.get('t1', False)
        v2 = request.POST.get('t2', False)
        v3 = request.POST.get('t3', False)
        v4 = request.POST.get('t4', False)
        v5 = request.POST.get('t5', False)
        v6 = request.POST.get('t6', False)
        v7 = request.POST.get('t7', False)
        v8 = request.POST.get('t8', False)
        v9 = request.POST.get('t9', False)
        v10 = request.POST.get('t10', False)
        v11 = request.POST.get('t11', False)
        v12 = request.POST.get('t12', False)
        v13 = request.POST.get('t13', False)
        v14 = request.POST.get('t14', False)
        
        data = [[v1, v2, float(v3), float(v4), float(v5), float(v6), float(v7), float(v8), float(v9), float(v10), float(v11), float(v12), float(v13), float(v14)]]
        data = pd.DataFrame(data, columns=['Name of State','Name of District','3','4','5','6','7','8','9','10','11','12','13','14'])
        data['Name of State'] = pd.Series(le.transform(data['Name of State'].astype(str)))#encode all str columns to numeric
        data['Name of District'] = pd.Series(le1.transform(data['Name of District'].astype(str)))#encode all str columns to numeric
        data = data.values
        data = scaler2.transform(data)
        predict = dt.predict(data)
        predict = predict.reshape(-1, 1)
        predict = scaler3.inverse_transform(predict)
        predict = predict.ravel()[0]
        context= {'data':"Water Well Level Predicted As : "+str(predict)}
        return render(request, 'ViewResult.html', context)

def CurrentWater(request):
    if request.method == 'GET':
       return render(request, 'CurrentWater.html', {})

def CurrentWaterAction(request):
    if request.method == 'POST':
        global uname, analyzer, scaler, scaler1, dt1
        v1 = request.POST.get('t1', False)
        v2 = request.POST.get('t2', False)
        v3 = request.POST.get('t3', False)
        v4 = request.POST.get('t4', False)
        v5 = request.POST.get('t5', False)
        data = [[float(v1), float(v2), float(v3), float(v4)]]
        data = np.asarray(data)
        data = scaler.transform(data)
        predict = dt1.predict(data)
        predict = predict.reshape(-1, 1)
        predict = scaler1.inverse_transform(predict)
        predict = predict.ravel()[0]
        context= {'data':"Water Level Predicted As : "+str(predict)}
        return render(request, 'CurrentWater.html', context)
        

def Feedback(request):
    if request.method == 'GET':
       return render(request, 'Feedback.html', {})  

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def FeedbackAction(request):
    if request.method == 'POST':
        global uname, analyzer
        feedback = request.POST.get('t1', False)
        sentiment_dict = analyzer.polarity_scores(feedback)
        label = ""
        if sentiment_dict['compound'] >= 0.05:
            label = "Positive"
        elif sentiment_dict['compound'] <= - 0.05:
            label = "Negative"
        else:
            label = "Neutral"
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'nikhil@21', database = 'waterwell',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO feedback(username,feedback,nlp_feedback_analysis) VALUES('"+uname+"','"+feedback+"','"+label+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        context= {'data':'Your feedback accepted<br/>NLP Analysis predicted from feedback = '+label}
        return render(request, 'Feedback.html', context)    

def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)        
        status = 'none'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'nikhil@21', database = 'waterwell',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from signup where username = '"+username+"'")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == email:
                    status = 'Given Username already exists'
                    break
        if status == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'nikhil@21', database = 'waterwell',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO signup(username,password,contact_no,email_id,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = 'Signup Process Completed'
        context= {'data':status}
        return render(request, 'Register.html', context)

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        option = 0
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'nikhil@21', database = 'waterwell',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    uname = username
                    option = 1
                    break
        if option == 1:
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'UserLogin.html', context)

