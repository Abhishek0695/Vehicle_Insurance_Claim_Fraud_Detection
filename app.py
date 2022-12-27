
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer


st.header("Predicting Fraudulent Vehicle Insurnace Claims")
st.subheader("Select the input features on the left for prediction model")
st.sidebar.header('User Input Features')

def user_input_features():
    month = st.sidebar.selectbox('Month of Accident', ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))
    WeekOfMonth = st.sidebar.selectbox('WeekOfMonth', (1,2,3,4,5))
    DayOfWeek = st.sidebar.selectbox('DayOfWeek', ('Wednesday', 'Friday', 'Saturday', 'Monday', 'Tuesday', 'Sunday', 'Thursday'))
    Make = st.sidebar.selectbox('Make', ('Honda','Toyota','Ford','Mazda','Chevrolet','Pontiac','Accura','Dodge','Mercury','Jaguar','Nisson','VW','Saab','Saturn','Porche','BMW','Mecedes','Ferrari','Lexus'))
    AccidentArea = st.sidebar.selectbox('AccidentArea', ('Urban', 'Rural'))
    DayOfWeekClaimed = st.sidebar.selectbox('DayOfWeekClaimed', ('Wednesday', 'Friday', 'Saturday', 'Monday', 'Tuesday', 'Sunday', 'Thursday'))
    MonthClaimed = st.sidebar.selectbox('MonthClaimed', ('Feb','Jan','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))
    WeekOfMonthClaimed = st.sidebar.selectbox('WeekOfMonthClaimed', (1,2,3,4,5))
    Sex = st.sidebar.selectbox('Sex', ('Female', 'Male'))
    MaritalStatus = st.sidebar.selectbox('MaritalStatus', ('Single', 'Married', 'Widow', 'Divorced'))
    Fault = st.sidebar.selectbox('Fault', ('Policy Holder', 'Third Party'))
    PolicyType = st.sidebar.selectbox('PolicyType', ('Sport - Liability','Sport - Collision','Sedan - Liability','Utility - All Perils','Sedan - All Perils','Sedan - Collision','Utility - Collision','Utility - Liability','Sport - All Perils'))
    VehicleCategory = st.sidebar.selectbox('VehicleCategory', ('Sport', 'Utility', 'Sedan'))
    VehiclePrice = st.sidebar.selectbox('VehiclePrice', ('more than 69000','20000 to 29000','30000 to 39000','less than 20000','40000 to 59000','60000 to 69000'))
    Deductible = st.sidebar.selectbox('Deductible', (300, 400, 500, 700))
    DriverRating = st.sidebar.selectbox('DriverRating', (1, 2, 3, 4))
    Days_Policy_Accident = st.sidebar.selectbox('Days_Policy_Accident', ('more than 30', '15 to 30', 'none', '1 to 7', '8 to 15'))
    Days_Policy_Claim = st.sidebar.selectbox('Days_Policy_Claim', ('more than 30', '15 to 30', '8 to 15', 'none'))
    PastNumberOfClaims = st.sidebar.selectbox('PastNumberOfClaims', ('none', '1', '2 to 4', 'more than 4'))
    AgeOfVehicle = st.sidebar.selectbox('AgeOfVehicle', ('3 years','6 years','7 years','more than 7','5 years','new','4 years','2 years'))
    AgeOfPolicyHolder = st.sidebar.selectbox('AgeOfPolicyHolder',('26 to 30','1 to 35','41 to 50','51 to 65','21 to 25','36 to 40','16 to 17','over 65','18 to 20'))
    PoliceReportFiled = st.sidebar.selectbox('PoliceReportFiled', ('No', 'Yes'))
    WitnessPresent = st.sidebar.selectbox('WitnessPresent', ('No', 'Yes'))
    AgentType = st.sidebar.selectbox('AgentType', ('External', 'Internal'))
    NumberOfSuppliments = st.sidebar.selectbox('NumberOfSuppliments', ('none', 'more than 5', '3 to 5', '1 to 2'))
    AddressChange_Claim = st.sidebar.selectbox('AddressChange_Claim', ('1 year', 'no change', '4 to 8 years', '2 to 3 years', 'under 6 months'))
    NumberOfCars = st.sidebar.selectbox('NumberOfCars', ('3 to 4', '1 vehicle', '2 vehicles', '5 to 8', 'more than 8'))
    BasePolicy = st.sidebar.selectbox('BasePolicy', ('Liability', 'Collision', 'All Perils'))

    data = {'Month': month,
            'WeekOfMonth': WeekOfMonth,
            'DayOfWeek': DayOfWeek,
            'Make': Make,
            'AccidentArea': AccidentArea,
            'DayOfWeekClaimed': DayOfWeekClaimed,

            'MonthClaimed': MonthClaimed,
            'WeekOfMonthClaimed': WeekOfMonthClaimed,
            'Sex': Sex,
            'MaritalStatus': MaritalStatus,
            'Fault': Fault,

            'PolicyType': PolicyType,
            'VehicleCategory': VehicleCategory,
            'VehiclePrice': VehiclePrice,
            'Deductible': Deductible,
            'DriverRating': DriverRating,

            'Days_Policy_Accident': Days_Policy_Accident,
            'Days_Policy_Claim' : Days_Policy_Claim,
            'PastNumberOfClaims': PastNumberOfClaims,
            'AgeOfVehicle': AgeOfVehicle,
            'AgeOfPolicyHolder' : AgeOfPolicyHolder,
            'PoliceReportFiled': PoliceReportFiled,
            'WitnessPresent': WitnessPresent,

            'AgentType': AgentType,
            'NumberOfSuppliments': NumberOfSuppliments,
            'AddressChange_Claim': AddressChange_Claim,
            'NumberOfCars': NumberOfCars,
            'BasePolicy': BasePolicy
            }
    features = pd.DataFrame(data,index=[0])
    return features

input_df = user_input_features()

df = pd.read_csv('Vehicle_Insurance_Fraud_Detection.csv')
df.drop(['Year','Age','PolicyNumber','RepNumber'],axis=1,inplace=True)


# from sklearn.model_selection import train_test_split

# train_set, test_set = train_test_split(df, test_size=0.3)

# train_y = train_set['FraudFound_P']
# # test_y = test_set['FraudFound_P']

df = df.drop(['FraudFound_P'], axis=1)
df = pd.concat([input_df, df], axis=0)
# st.write(df.shape)
# test_inputs = test_set.drop(['FraudFound_P'], axis=1)

# Identify the numerical columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.to_list()

# Identify the categorical columns
categorical_columns = df.select_dtypes('object').columns.to_list()

numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)],
        remainder='passthrough')

output_df = preprocessor.fit_transform(df)
# test_x = preprocessor.transform(test_inputs)
# input_df = preprocessor.transform(input_df)
output_df = output_df[:1]
# st.write(output_df.shape)

st.subheader('This is the user input in the form of a Dataframe')
st.write(input_df)


load_clf = pickle.load(open('randomforestmodel.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(output_df)
prob = round(load_clf.predict_proba(output_df)[0][list(load_clf.classes_).index(prediction)],2)

st.subheader('Click on the prediction button to make prediction')

if st.button('Predict'):
    if prediction == 0:
        st.write("**Vehicle Insurance Claim is Legit**")
    else:
        st.write("Vehicle Insurance Claim is a Fraud as per the model")

    st.write("The Prediction probability is ", prob)


