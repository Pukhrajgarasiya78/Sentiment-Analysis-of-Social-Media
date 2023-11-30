import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import streamlit as st
import streamlit.components.v1 as components
from textblob import TextBlob
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    with open('twitter.pkl', 'rb') as file:
        data = pickle.load(file)
except Exception as e:
    st.error(f"Error loading the pickle file: {e}")
    data = None

st.title('Sentiment Analysis Visualization')

username = st.text_input('Enter Username:', 'text')

if st.button('Visualize'):
    if data is not None:
        if username in data['name'].values:
            user_data = data[data['name'] == username]

            pos_value = user_data['pos'].values[0]
            neu_value = user_data['neu'].values[0]
            neg_value = user_data['neg'].values[0]

            

            st.write(f'Sentiment Analysis for {username}')
            st.write(f'Positive: {pos_value}')
            st.write(f'Neutral: {neu_value}')
            st.write(f'Negative: {neg_value}')

            plt.figure(figsize=(6, 6))
            plt.bar(['Positive', 'Neutral', 'Negative'], [pos_value, neu_value, neg_value])
            plt.xlabel('Sentiment Category')
            plt.ylabel('Values')
            plt.title(f'Sentiment Analysis for {username}')
            st.pyplot(plt)
        else:
            st.write("Username not found in the dataset.")
    else:
        st.warning("No data loaded. Please check the pickle file.")
        
def convert_to_df(sentiment):
    sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
    return sentiment_df

def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)
            
        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)
            
    result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
    return result

def main():
    st.title("Analyze By Words")
    st.subheader("Hello Visitors")
    
    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Home":
        st.subheader("Home")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')
            
        #layout
        col1,col2 = st.columns(2)
        if submit_button:
        
            with col1:
                st.info("Results")
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)
                
                #Emoji
                if sentiment.polarity > 0:
                    st.markdown("Sentiment:: Positive :smiley:")
                elif sentiment.polarity < 0:
                    st.markdown("Sentiment:: Negative :angry:")
                else:
                    st.markdown("Sentiment:: Neutral 😐")
                #DataFrame
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)    
                    
                #Visualization
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric')
                st.altair_chart(c,use_container_width=True)
                
            with col2:
                st.info("Token Sentiment")
                
                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)
                
            
    else:
        st.subheader("About")

if __name__ == '__main__':
    main()