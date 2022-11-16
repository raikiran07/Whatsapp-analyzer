import streamlit as st
import preprocess
import helper
import matplotlib.pyplot as plt
# Utils
import joblib
pipe_lr = joblib.load(open("./models/emotion_classifier_pipe_lr_03_june_2021.pkl", "rb"))

#code for the sidebar
st.sidebar.title("Whatsapp Analyzer")

#code for uploading the file
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")

    df = preprocess.preprocess(data)
    # st.dataframe(df)
    st.title("Top Statistics")

    #fetch unique users
    user_list = df['user'].unique().tolist()

    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):
        num_of_msg,words,num_media_msg,num_of_links = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header("Total Messages")
            st.success(num_of_msg)
        with col2:
            st.header("Total Words")
            st.success(words)

        with col3:
            st.header("Media Shared")
            st.success(num_media_msg)

        with col4:
            st.header("Link Shared")
            st.success(num_of_links)

        #Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #Daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(daily_timeline['only-date'],daily_timeline['message'],color='purple')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #Day-wise timeline
        st.title("Day wise Timeline")
        col1, col2 = st.columns(2)
        day_timeLine = helper.day_wise_analyse(selected_user, df)
        fig, ax = plt.subplots()

        ax.bar(day_timeLine['day'], day_timeLine['message'], color='grey')
        plt.xlabel("Days",color='red',fontsize='20')
        plt.ylabel("No of messages",color='red',fontsize='20')
        plt.xticks([i for i in range(1,32,2)])

        st.pyplot(fig)

        #Hourly timeline
        st.title("Hourly Timelin")
        hour_df = helper.hour_timeLine(selected_user,df)
        fig, ax = plt.subplots()
        ax.bar(hour_df['hours'],hour_df['No_of_Messages'],color='purple')
        plt.xlabel('Hours',color='black',fontsize='18')
        plt.ylabel('No of Messages',color='black',fontsize='18')
        plt.xticks([i for i in range(1,24,2)])
        st.pyplot(fig)







        #finding the busiest users in the group(Group level)

        if selected_user == 'Overall':
            st.title('Most Active Users')
            x,new_df = helper.most_active_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)





        #word cloud
        st.title("WorldCloud")
        df_wc = helper.create_worldcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)


        #Most common 20 words
        most_common_df = helper.most_common_words(selected_user,df)

        # st.dataframe(most_common_df)
        fig,ax = plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title("Most Common Words")
        st.pyplot(fig)


        #emoji analysis
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_analyzer(selected_user,df)
        # st.dataframe(emoji_df)
        col1,col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)

        #emotional analysis
        # Fxn
        def predict_emotions(docx):
            results = pipe_lr.predict([docx])
            return results[0]

        # emotional probability analysis
        def get_prediction_proba(docx):
            results = pipe_lr.predict_proba([docx])
            return results

        st.title("Emotional Analysis of the messages")
        emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
                               "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}
        message = helper.emotional_analysis(selected_user,df)
        emotion = predict_emotions(message)
        predict_prob = get_prediction_proba(message)
        emoji_icon = emotions_emoji_dict[emotion]

        st.success("{}:{}".format(emotion,emoji_icon))
        # emoji_icon = emotions_emoji_dict[prediction]
        # st.write("{}:{}".format(prediction, emoji_icon))









