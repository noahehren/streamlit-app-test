import streamlit as st
from ingest_final import ingest_predicted_reviews, data_file
import pandas as pd
from datetime import datetime
import numpy as np

#################################################################
################## Page Settings ################################
#################################################################
st.set_page_config(page_title="My Streamlit App", layout="wide")
st.markdown('''
<style>
    #MainMenu
    {
        display: none;
    }
    .css-18e3th9, .css-1d391kg
    {
        padding: 1rem 2rem 2rem 2rem;
    }
</style>
''', unsafe_allow_html=True)

#################################################################
################## Page Header ##################################
#################################################################
st.header("Predicting Sentiment of Reddit Movie Reviews")
st.write("Our application uses Artificial Intelligence to predict user sentiment of user comments posted on r/movies subreddit")
st.markdown('---')

################## Sidebar Menu #################################
page_selected = st.sidebar.radio("Menu", ["Home", "Model", "Ingest", "About"])

################################################################
################## Home Page ###################################
################################################################
if page_selected == "Home":
    
    ######### Load labeled data from datastore #################
    df = pd.read_csv(data_file)
    
    ##### Compute month-day, hour columns from timestamp #######
    df['mmmdd'] = df['created'].apply(lambda x: datetime.fromtimestamp(x).strftime('%b %d'))
    df['hour'] = df['created'].apply(lambda x: datetime.fromtimestamp(x).hour)
    df['length'] = df['title'].apply(lambda x: len(str(x))) + df['comment'].apply(lambda x: len(str(x))) 
    df['color'] = df['sentiment'].apply(lambda x: 'orange' if x == 'negative' else 'skyblue')
    
    ############# Filters ######################################
    
    ######### Date range slider ################################
    start, end = st.sidebar.select_slider(
                    "Select Date Range", 
                    df.mmmdd.drop_duplicates().sort_values(), 
                    value=(df.mmmdd.min(), df.mmmdd.max()))
    
    ######### Review Type Filter ################################
    type = st.sidebar.selectbox("Type", ['All', 'Discussion', 'Recommendation', 'Article', 'News'])
    
    ######### Apply filters ####################################
    df_filter = df.loc[(df.mmmdd >= start) & (df.mmmdd <= end), :]
    if type != "All":
        df_filter = df.loc[df.type == type, :]
    
    ######### Main Story Plot ###################################
    col1, col2 = st.columns((2,1))
    with col1: 
        ax = pd.crosstab(df_filter.mmmdd, df_filter.sentiment).plot(
                kind="bar", 
                figsize=(6,2), 
                xlabel = "Date",
                color={'positive':'skyblue', 'negative': 'orange'})
        st.pyplot(ax.figure)
    with col2:
        st.write('This plot shows the daily count of positive and negative movie reviews in the r/movies subreddit.')
    st.markdown('---')

    ######### Length vs Hour Plot ###################################
    col1, col2 = st.columns((2,1))
    with col1: 
        ax = df_filter.plot.scatter(x='length', y='hour', c=df_filter['color'], figsize=(6,2))
        st.pyplot(ax.figure)
    with col2:
        st.write('This scatter plot shows the length of movie review againt hour of day')
    st.markdown('---')

    ######### Length vs Hour Plot ###################################
    col1, col2 = st.columns((2,1))
    with col1: 
        ax = df_filter.groupby(['type']).size().reset_index(name='count').plot.bar(x='type', y='count', figsize=(6,2))
        st.pyplot(ax.figure)
    with col2:
        st.write('This bar plot shows the number of different movie review types')
    st.markdown('---')

    ######## Sample Reviews and Sentiment Predictions ###############
    st.subheader("Sample Reviews and Sentiment Predictions")
    df_sample = df_filter.sample(5)
    for index, row in df_sample.iterrows():
        col1, col2 = st.columns((1,5))
        with col1:
            if row['sentiment'] == "positive":
                st.success("Positive") 
            else: 
                st.error("Negative")    
        with col2:
            with st.expander(row['title']):
                st.write(row['comment'])

################################################################
############### Model Training and Evaluation ##################
################################################################
elif page_selected == "Model":
    st.subheader("Training and Model Evalutaion")
    col1, col2 = st.columns(2)
    with col1:
            st.image("wordcloud_pos.png")
            st.caption("Word Cloud for Positive Sentiment")
    with col2:
            st.image("wordcloud_neg.png")
            st.caption("Word Cloud for Negative Sentiment")

    # from wordcloud import WordCloud
    # df = pd.read_csv('IMDB_movie_reviews_labeled.csv')  
    # text = " "
    # for index, row in df.iloc[:100,:].iterrows(): #going to to go every row in the dataframe and collect the text
    #     text += str(row['review']) + " "
    # wc = WordCloud()
    # #preprocessed = WordCloud.process_text(text) #splits long text into words and eliminates the stop words
    # wc.generate(text).to_file('wordcloud.png') #check documentation on wordcloud website
    

################################################################
#################### Ingest Data Tool ##########################
################################################################
elif page_selected == "Ingest":
    st.subheader("Ingest Tool")
    num_of_posts = st.selectbox("Select number of Posts", [1, 5, 10, 20, 50, 100])
    if st.button("Submit"):
        with st.spinner('Wait for it...'): 
            ingest_predicted_reviews(num_of_posts)
        st.success('Done!')
        st.balloons()

################################################################
############### About My Company and Team ######################
################################################################
else:
    st.subheader("About Us")
    col1, col2 = st.columns((1,40))
    with col2:
        st.write("write about us text here")
    st.subheader('Team')
    ### Member 1
    col1, col2 = st.columns((1,40))
    with col2:
        st.markdown('**Firstname Lastname**')
    col11, col12, col13 = st.columns((1,6,34))
    with col12:
        st.image('member1.jpg') 
    with col13:
        st.write('write bio')
    ### Member 2    
    col1, col2 = st.columns((1,40))
    with col2:
        st.markdown('**Firstname Lastname**')
    col11, col12, col13 = st.columns((1,6,34))
    with col12:
        st.image('member1.jpg') 
    with col13:
        st.write('write bio')