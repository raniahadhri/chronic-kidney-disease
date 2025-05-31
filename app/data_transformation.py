from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler,normalize,MinMaxScaler
import joblib



def data_spliting(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    df_cat = df[cat_cols]
    df_num = df[num_cols]
    return df_num, df_cat


def encoding(df_cat, label_encoders):
    df_cat = df_cat.copy()  # Avoid SettingWithCopyWarning
    for col in df_cat.columns:
        if col in label_encoders:
            le = label_encoders[col]
            df_cat[col] = le.transform(df_cat[col])
    return df_cat


def scaling(df_cat,df_num, scaler):
    df_encoded = pd.concat([df_cat, df_num], axis=1)
    df_scaled = scaler.transform(df_encoded)
    df_scaled = pd.DataFrame(df_scaled, columns=df_encoded.columns)
    return df_scaled

def data_transformation(df, label_encoders, scaler):
    df_num, df_cat = data_spliting(df)
    df_cat = encoding(df_cat.copy(), label_encoders)
    df_scaled = scaling(df_cat, df_num, scaler)
    return df_scaled

def decode_prediction(prediction):
    categories = ['No_Disease', 'Low_Risk', 'Moderate_Risk', 'Severe_Disease', 'High_Risk']
    return categories[int(prediction)]
