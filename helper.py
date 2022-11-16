from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import emoji






extract = URLExtract()
def fetch_stats(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())

    #fetch number of media messages
    num_of_med = df[df['message'] == '<Media omitted>\n'].shape[0]

    #fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))
    return num_messages, len(words), num_of_med,len(links)

def most_active_users(df):
    df = df[df['user'] != 'group_notification']
    x = df['user'].value_counts().head()
    nums_df = df['user'].value_counts().reset_index()
    df = round((df['user'].value_counts()/df.shape[0])*100,2).reset_index().rename(columns={'index':'Name','user':'Percentage%'})
    df['Messages'] = nums_df['user']
    return x,df

#create word cloud
def create_worldcloud(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]



    # removint the group notification user
    temp = df[df['user'] != 'group_notification']
    # removing the midea ommitted message
    temp = temp[temp['message'] != '<Media omitted>\n']



    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

#Most common 20 words
def most_common_words(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # remvovint the groupt notification user
    temp = df[df['user'] != 'group_notification']
    # removing the midea ommitted message
    temp = temp[temp['message'] != '<Media omitted>\n']

    # importing the hinglish stopword
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)


    most_common_df = pd.DataFrame(Counter(words).most_common(20))

    return most_common_df


def emoji_analyzer(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        #emoji.UNICODE_EMOJI['en'] is replaced by emoji.EMOJI_DATA
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

#Monthly timeline
def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    # groupby using the month
    time_line = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(time_line.shape[0]):
        time.append(time_line['month'][i] + "-" + str(time_line['year'][i]))
    time_line['time'] = time

    return time_line

def daily_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('only-date').count()['message'].reset_index()
    return daily_timeline

def day_wise_analyse(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df.groupby('day').count()['message'].reset_index()
    return df

#hour timeline
def hour_timeLine(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = (df['hour']).value_counts().reset_index().rename(columns={'index':'hours','hour':'No_of_Messages'})
    return df



def emotional_analysis(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    message_list = []
    for message in df['message']:
        message_list.append(message)
    message_text = "".join(message_list)

    return message_text

