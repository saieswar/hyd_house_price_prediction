
import pickle
import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from flask import Flask,request,jsonify,render_template

## import model , standard scalar & encoder pickle

linear_reg = pickle.load(open('models/house_model.pkl', 'rb'))
encoder = pickle.load(open('models/endocder.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))


application = Flask(__name__)
app = application



@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get_options",methods=['GET'])
def get_options():
    df = pd.read_csv("notebooks/Hyderbad_House_price.csv")
    unique_titles = sorted(df["title"].dropna().unique().tolist())
    unique_locations = sorted(df["location"].dropna().unique().tolist())
    unique_statuses = sorted(df["building_status"].dropna().unique().tolist())
    return jsonify({
        "titles": unique_titles,
        "locations": unique_locations,
        "building_statuses": unique_statuses
    })
    

@app.route("/predict_data", methods=['POST'])
def predict_data():
        
    title = request.form['title']
    location = request.form['location']
    rate_persqft = float(request.form['rate_persqft'])
    area_insqft = float(request.form['area_insqft'])
    building_status = request.form['building_status']
    print(title,location,rate_persqft,area_insqft,building_status)
    categorical_cols = ['title', 'location', 'building_status']
    numerical_cols = ['rate_persqft', 'area_insqft']
    user_input_pd = pd.DataFrame([[title,location,building_status,rate_persqft,area_insqft]], columns=categorical_cols+numerical_cols)
    encoded_df = pd.DataFrame(encoder.transform(user_input_pd[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
    numerical_df = user_input_pd[numerical_cols]
    scaler_df = pd.DataFrame(standard_scaler.transform(numerical_df),columns=numerical_cols)
    final_df = pd.concat([scaler_df,encoded_df], axis=1)
    predict_price = linear_reg.predict(final_df)[0]
    
    return render_template('index.html', prediction=round(predict_price),title=title, 
                       location=location, 
                       area_insqft=area_insqft, 
                       building_status=building_status)

if __name__=="__main__":
    application.run(host="0.0.0.0")