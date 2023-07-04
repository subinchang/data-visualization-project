import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from collections import OrderedDict
import random
import decimal
from itertools import islice
from wordcloud import WordCloud 
from PIL import Image

import plotly
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from gensim.models import KeyedVectors

# 페이지 기본 설정
st.set_page_config(
    layout="wide",
)
st.title("🎧 멜론 플레이리스트 시각화 🎧")

# 데이터 불러오기
@st.cache_data
def load_data():
    with open("data/df_genre.pickle", 'rb') as f:
        df_genre = pickle.load(f)
    with open("data/df_meta.pickle", 'rb') as f:
        df_meta = pickle.load(f)
    with open("data/df_pl.pickle", 'rb') as f:
        df_pl = pickle.load(f)
    return df_genre, df_meta, df_pl
df_genre, df_meta, df_pl = load_data()

#####################################################################
# 1. 전체 플레이리스트 순위
st.markdown('#### :green[1. Most Liked Playlists]')
show_cols = ["plylst_title", "like_cnt", "updt_date"]
df_pl[show_cols].set_index(['plylst_title'])
st.dataframe(df_pl[show_cols].set_index(['plylst_title']).sort_values(by=['like_cnt'], axis=0, ascending=False), width=700)

#####################################################################
# 2. 플레이리스트의 곡 수, 태그 수 통계
st.title('')
st.markdown('#### :green[2. Playlist 곡 수, 태그 수 통계]')
pl = df_pl[['id', 'songs', 'tags']].copy()
pl['song_cnt'] = pl.songs.apply(lambda x: len(x))
pl['tag_cnt'] = pl.tags.apply(lambda x: len(x))

col1, col2 = st.columns(2)
with col1:
    df = px.data.tips()
    fig = px.box(pl, y="song_cnt", color_discrete_sequence=['green'])
    fig.update_layout(title='Song Count Statistics', height=500, width=500)
    st.write(fig)
with col2:
    df = px.data.tips()
    fig = px.box(pl, y="tag_cnt", color_discrete_sequence=['green'])
    fig.update_layout(title='Tag Count Statistics', height=500, width=500)
    st.write(fig)

######################################################################
# 3. 전체 장르 트리맵
st.title('')
st.markdown('#### :green[3. Genre Treemap]')
def genre_seperate(x):
    if x[-2:] == "00":
        value = "big"
    else:
        value = "small"
    return value
df_genre["big_small"] = df_genre["Code"].map(genre_seperate)

for i in df_genre.index:
    x = (df_genre[df_genre['Code']== (df_genre.loc[i,'Code'][:4]+'00')]['Genre'])
    df_genre.loc[i,'big'] = x.iloc[0]
df = df_genre.copy()

big_genres = df[df['big_small'] == 'big']
small_genres = df[df['big_small'] == 'small']

fig = px.treemap(df_genre, path=['big', 'Genre'], 
                 color_discrete_sequence=px.colors.qualitative.Prism)
fig.update_layout(width=1000, height=800)
st.write(fig)

######################################################################
# 4. 대분류 장르 비율, 최대 대분류 내 소분류 장르 비율
st.title('')
st.markdown('#### :green[4. Genre Details]')
gnr_code_b = df_genre[df_genre['Code'].str[-2:] == '00'].copy()
gnr_code_b.rename(columns = {'Code':'gnr_code', 'Genre':'gnr_name'}, inplace=True)

song_gnr_map = df_meta[['id', 'song_gn_gnr_basket', 'song_gn_dtl_gnr_basket']].copy()
song_gnr_map['song_gn_gnr_basket'] = song_gnr_map['song_gn_gnr_basket'].apply(lambda x: '|'.join(x))
song_gnr_map.rename(columns = {'id' : 'song_id', 'song_gn_gnr_basket':'gnr_code'}, inplace = True)

gnr_song = pd.merge(gnr_code_b, song_gnr_map, how='outer',on='gnr_code')
gnr_song_count = gnr_song.groupby(['gnr_code', 'gnr_name']).count().reset_index()

col1, col2 = st.columns([3, 2])
with col1:
    colors = ['darkgreen' if i in [2, 3, 5, 10] else 'darkseagreen' for i in range(len(gnr_song_count))]
    # darkgreen(진한 색)이 국내
    colors[8] = 'orange'
    fig = go.Figure(data=[go.Bar(x=gnr_song_count['gnr_name'], y=gnr_song_count['song_id'], marker_color=colors)])
    fig.update_layout(title='대분류 장르별 곡 수', yaxis_title='song_cnt')
    fig.update_xaxes(tickangle=45)
    st.write(fig)
with col2:
    song_dtl = song_gnr_map[song_gnr_map['gnr_code'] == 'GN0900'].copy()
    dtl_gnr = song_dtl['song_gn_dtl_gnr_basket'].explode()
    freq = dtl_gnr.value_counts()
    freq = freq.rename({'GN0908': "'10-'", 'GN0902': '얼터너티브팝', 'GN0903':'올디스', 'GN0904':'월드팝',
                        'GN0905':"'60-'70", 'GN0906':"'80-'90", 'GN0907':"'00'"})
    colors = ['rgb(95,70,144)', 'rgb(29,105,150)', 'rgb(56,166,165)', 'rgb(15,133,84)',
              'rgb(115,175,72)', 'rgb(237,173,8)', 'rgb(225,124,5)']
    fig = go.Figure(data=[go.Pie(labels=freq[1:8].index, values=freq[1:8].values, marker=dict(colors=colors))])
    fig.update_layout(title='POP 대분류의 소분류 장르 비율')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.write(fig)

######################################################################
# 5. 발행연도 태그수에 따른 좋아요수의 관계
st.title('')
st.markdown('#### :green[5. Playlist Year & Tag Count & Like Count]')
st.markdown('- 플레이리스트의 업데이트 날짜, 태그 수, 좋아요 수의 관계')
tag_len = pd.Series([len(x) for x in df_pl['tags']])

df_pl['tag_len'] = tag_len
df_pl['tag_len'] = df_pl['tag_len'].astype(float)
df_pl['updt_date'] = pd.to_datetime(df_pl['updt_date'])
df_pl['year_month'] = df_pl['updt_date'].dt.strftime("%Y-%m")

df_pl2008 = df_pl[df_pl['updt_date'] > pd.to_datetime('2008-01-01')]

fig = px.scatter(df_pl2008,
                 x="year_month",
                 y="like_cnt",
                 color = 'tag_len',
                 hover_data=["tag_len", "like_cnt"],
                 size = df_pl2008['like_cnt']*2,
                 size_max = 50,
                 opacity = 0.7,
                 color_continuous_scale=px.colors.sequential.Viridis)
fig.update_traces(marker=dict(line=dict(width=0)))
fig.update_layout(width=1100, height=600)
st.write(fig)

######################################################################
# 6. 노래 발매연도별 비중
st.markdown('#### :green[6. Song Count by Year]')
song_issue_date = df_meta[['id', 'issue_date']].copy()
song_issue_date.loc[:, 'issue_date'] = song_issue_date['issue_date'].astype(str)
song_issue_date['issue_year'] = song_issue_date['issue_date'].str[0:4]
song_issue_date['id'] = song_issue_date['id'].astype(str)
song_issue_date_filter = song_issue_date[song_issue_date.issue_year >= '2000']
issue_year_song_cnt = song_issue_date_filter.groupby('issue_year').id.nunique().reset_index(name = 'song_cnt')

fig = go.Figure()

fig.add_trace(go.Bar(x=issue_year_song_cnt['issue_year'], y=issue_year_song_cnt['song_cnt'], name='Song count', marker_color='darkseagreen'))
fig.update_layout(title='2000년 이후 연도별 곡 count',
                  xaxis_title='Year',
                  yaxis_title='Song Count',
                  width=1100, height=450)
st.write(fig)

######################################################################
# 7. TOP 150 플레이리스트 가수, 곡 산점도
st.title('')
st.markdown('#### :green[7. TOP 150 Playlist Details]')
artist_group = pd.read_csv('data/artist_group.csv',index_col = 0)

size = np.log10(artist_group['like_avg'])
size = (artist_group['like_avg'] - artist_group['like_avg'].min()) / (artist_group['like_avg'].max() - artist_group['like_avg']) * 500 + 10

tab1, tab2 = st.tabs(["🎤 Artists", "🎶 Songs"])
with tab1:
    fig = px.scatter(artist_group,
                    x="many",
                    y="included",
                    color=artist_group.index,
                    hover_data={'included': True, 'like_avg': True},
                    hover_name=artist_group.index,
                    size = size,
                    size_max=200,
                    opacity = 0.57,
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_layout(width=1000, height=650)             
                    # hovermode = "x unified")
    st.write(fig)
def extract_year_month(string):
    year_month = string[:6]
    return f"{year_month[:4]}-{year_month[4:]}"

song_date = df_meta['issue_date']
song_date = song_date.astype('string')
song_date = song_date.apply(extract_year_month)

a = 150 # 상위 몇개로 추출할 변수

ten_song = []
for x,i in enumerate(df_pl['songs']):
    ten_song.extend(i)
topsong = pd.Series(ten_song).value_counts()[:a]

graph_group = pd.DataFrame()
song_id_name = df_meta['song_name']

graph_group['included'] = topsong
graph_group['like_cnt_avg'] = 0
graph_group['song_name'] = song_id_name[graph_group.index]
s = pd.to_datetime(song_date[graph_group.index])
s = s.dt.strftime("%Y-%m")
graph_group['release_date'] = s

for i in graph_group.index:
    sum = 0
    temp = df_pl[pd.Series([i in x for x in df_pl['songs']])]
    # graph_group.loc[i,'like_cnt_avg'] = temp['like_cnt'].mean().round()
    graph_group.loc[i,'like_cnt_avg'] = decimal.Decimal(temp['like_cnt'].mean()).quantize(decimal.Decimal('0.01'))

size = np.log10(graph_group['included'])
size = (graph_group['included'] - graph_group['included'].min()) / (graph_group['included'].max() - graph_group['included'].min()) * 100 + 10
with tab2:
    fig = px.scatter(graph_group,
                    x="release_date",
                    y="like_cnt_avg",
                    color="song_name",
                    hover_data=["song_name", "like_cnt_avg",'included'],
                    hover_name = "song_name",
                    size = size,
                    size_max=45,
                    opacity = 0.57)
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_layout(width=1000, height=650)
    st.write(fig)