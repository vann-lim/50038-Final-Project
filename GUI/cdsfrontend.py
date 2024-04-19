import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import timedelta
import talib
import numpy as np
import pickle 
from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
#from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from custom_module import CustomLayer


st.title('Predicting Rise/Fall of Bitcoin Prices')
# Add bullet points in the first column

st.markdown("### What this does:")
st.write('This is a simple web app to predict the rise/fall of Bitcoin prices based on the following data:')

col1, col2 = st.columns(2)
with col1:
    st.markdown("- Tweet\n"
                "- News\n"
                "- Past Bitcoin Price")

with col2:
    st.markdown("- Past Bitcoin Volume\n"
                "- Past S&P 500 Price\n"
                "- Past US Interest rates")

st.markdown("### How to use:")


st.markdown("###### Step 1")
date_input = st.date_input("Select a **date** from the calendar")
start_date= date_input - timedelta(days=40)
formatted_start_date=str(start_date).replace("/","-")
# formatted_date_input = str(date_input).replace("/","-")
# if date_input!= None:
#     st.write('You selected:', formatted_date_input)

# Reformat Date
formatted_end_date = str(date_input).replace("/","-")


# st.markdown("\n\n\n\n\n")
# st.markdown("###### Step 2")
# tweet_input = st.text_input('Enter a **tweet** from this date')

# st.markdown("\n\n\n\n\n")
# st.markdown("###### Step 3")
# news_input = st.text_input('Enter a **crypto-related news** from this date')

st.markdown("\n\n\n\n\n")
st.markdown("###### Step 2")
st.write('Press the button below to get the predict ion')


# -----------------------------------------------------------------
# Calling the model
def predict(btc):
    # ...
    
    with open('poly_svm_model_k5_BTC_only.pkl', 'rb') as f:
        trained_model = pickle.load(f)


    selected_feats=['Open', 'High', 'Close', 'Adj Close', 'Volume', 'RSI_14', 'macd', 'macd_signal', 'BB_middle', 'BB_lower', 'Close-Open', 'High-Low', 'OBV', 'BB_height']
    X_test = btc[selected_feats]
    
    prediction = trained_model.predict(X_test.values)
    return prediction[0]


# def model_tweet(tweet_input):
#     # model = AutoModelForSequenceClassification.from_pretrained('model_tweet_textonly.h5', num_labels=2)
#     model = tf.keras.models.load_model('model_tweet.h5', compile=False)
#     # model = load_model('model_tweet_textonly.h5', compile=False)

#     # Load the tokenizer
#     with open('tokenizer_tweet.pickle.h5', 'rb') as handle:  
#         tokenizer = pickle.load(handle)
    
#     sequences = tokenizer.texts_to_sequences(tweet_input)
#     max_len = max([len(seq) for seq in sequences])
#     new_data_tokenized = pad_sequences(sequences, maxlen=max_len)
    
#     predictions = model.predict(new_data_tokenized)
#     binary_predictions = (predictions > 0.5).astype("int32")
    
#     return binary_predictions

# def model_news(news_input):
#     # ...
#     return prediction




# -----------------------------------------------------------------


# Styled button using HTML and CSS
button_style = """
    <style>
    .styled-button {
        display: flex;
        justify-content: center;
        align-items: center;
        background-image: linear-gradient(to right, #ff7e5f, #feb47b);
        border-radius: 5px;
        color: white;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        font-size: 16px;
    }
    </style>
"""

# Add the styled button
st.markdown(button_style, unsafe_allow_html=True)

# Add functionality to the button
if st.markdown("<button class='styled-button' type='submit'>Let's go!</button>", unsafe_allow_html=True):
    # Obtain S&P 500, intrest rate, and BTC Price data
    # Define the ticker symbol for S&P 500
    # sp500_symbol = "^GSPC"  # ticker symbol for S&P 500
    # interest_symbol = "FEDFUNDS" # ticker symbol for US interest rates
    bitcoin_symbol = "BTC-USD"  # ticker symbol for Bitcoing

    # # Get S&P 500 data
    # sp500_data = yf.download(sp500_symbol, start=formatted_date_input, end=formatted_date_input)
    # sp500_close = sp500_data['Close'][0]
    # # st.dataframe(sp500_data)

    # # Get bitcoin data
    df = yf.download(bitcoin_symbol, start=formatted_start_date, end=formatted_end_date)#df
        #just to make sure sorted by date

    # Compute 14-day RSI
    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)

    # Compute MACD
    df['macd'], df['macd_signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Compute Bollinger Bands
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['Close-Open']=df['Close']-df['Open']
    df['High-Low']=df['High']-df['Low']
    #remove all rows with NAN
    df.dropna(inplace=True)
    #reset index
    df.reset_index(drop=True, inplace=True)
    # Calculate OBV
    df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0)).cumsum()

    df.drop(df.index[0], inplace=True)

    df['BB_height']=df['BB_upper']-df['BB_lower']

    scaler = MinMaxScaler()
    df['RSI_14'] = scaler.fit_transform(df[['RSI_14']])
    df['macd'] = scaler.fit_transform(df[['macd']])
    df['macd_signal'] = scaler.fit_transform(df[['macd_signal']])
    df['BB_height'] = scaler.fit_transform(df[['BB_height']])
    df['BB_upper'] = scaler.fit_transform(df[['BB_upper']])
    df['BB_middle'] = scaler.fit_transform(df[['BB_middle']])
    df['BB_lower'] = scaler.fit_transform(df[['BB_lower']])
    df['Close-Open'] = scaler.fit_transform(df[['Close-Open']])
    df['High-Low'] = scaler.fit_transform(df[['High-Low']])
    df['OBV'] = scaler.fit_transform(df[['OBV']])
    df['Open'] = scaler.fit_transform(df[['Open']])
    df['High'] = scaler.fit_transform(df[['High']])
    df['Low'] = scaler.fit_transform(df[['Low']])
    df['Close'] = scaler.fit_transform(df[['Close']])
    df['Adj Close'] = scaler.fit_transform(df[['Adj Close']])
    df['Volume'] = scaler.fit_transform(df[['Volume']])
        

    # # Get US interest rate data
    # interest_data = yf.download(interest_symbol, start=formatted_date_input, end=formatted_date_input)
    # interest_close = interest_data['Close'][0]
    

    prediction = predict(df.iloc[[-1]])
    print("done:",formatted_end_date)
    st.write('Prediction:', prediction)

