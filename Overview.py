import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import pickle
from pyvis.network import Network

# 페이지 기본 설정

st.set_page_config(
    layout="wide",
)
st.title("🎧 멜론DJ 플레이리스트 추천 Overview 🎧")


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

#########################################################
# 1. 데이터셋 소개
st.markdown('#### :green[1. Dataset Overview]')
df_meta_ord = df_meta[['song_name', 'song_gn_gnr_basket', 'song_gn_dtl_gnr_basket','artist_id_basket', 'artist_name_basket','album_id', 'album_name', 'issue_date']]
df_pl_ord = df_pl[['plylst_title','tags','songs', 'like_cnt','updt_date']]

tab1, tab2, tab3 = st.tabs(['Genre', 'Song', 'Playlist'])
with tab1:
    col1, col2 = st.columns([1, 4])
    with col1:
        st.dataframe(df_genre.head(20), height=250)
    with col2:
        st.markdown('- 254 x 2\n'
                    '- 00으로 끝나면 대분류, 그 외는 소분류이다.\n'
                    '- 같은 대분류가 2번 나타나는 경우 코드의 숫자가 작은 것이 국내, 큰 것이 해외를 의미한다.\n')
with tab2:
    st.dataframe(df_meta_ord.head(15), height=250)
    st.markdown('- 707989 x 9')
with tab3:
    st.dataframe(df_pl_ord.head(15), height=250)
    st.markdown('- 115071 x 6')

#########################################################
# 2. 네트워크 그래프
st.markdown('#### :green[2. Artist & Genre & Songs]')
genre = "data/genre_gn_all.json"
meta = "data/song_meta.json"
play_list = "data/train.json"

# genre 데이터 부터 확인
with open(genre) as f:
    js = json.loads(f.read())
df_genre = pd.DataFrame.from_dict(js, orient='index')
df_genre = df_genre.reset_index()
df_genre.columns = ["Code", "Genre"]

# 메타 데이터
df_meta = pd.read_json(meta)
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

def getlistout(x):
    return x[0]

df_meta['artist_name'] = df_meta['artist_name_basket'].apply(getlistout)
new_df_genre = df_genre.set_index('Code')

def return_genre(x):
    return new_df_genre.loc[x,'Genre']

def return_big(x):
    return x[:4]+'00'

# 입력 받기
artists = ['아이유', '박효신']
artist = st.selectbox("Select an artist", artists)

# 필터링 조건 설정
filter_cond = df_meta["artist_name"] == artist
df_filtered = df_meta[filter_cond].reset_index(drop=True)

# pyvis 네트워크 그래프 생성
net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white",notebook=True)
net.barnes_hut(gravity=-8000, central_gravity=0.6, spring_length=200, spring_strength=0.001, damping=0.15, overlap=0)

net.add_node(artist, title=artist, label=artist, color="lightgreen",size = 120)
for _,row in df_filtered.iterrows():
    for i in row['song_gn_gnr_basket']:
        if i==None: break
        net.add_node(artist+i, title=return_genre(i), label=return_genre(i), color="orange",size = 80)
        net.add_edge(artist, artist+i)
    for i in row['song_gn_dtl_gnr_basket']:
        if i==None: break
        net.add_node(artist+i, title=return_genre(i), label=return_genre(i), color="yellow",size = 55)
        net.add_edge(artist+return_big(i), artist+i)
for i in range(len(df_filtered)):
    song = df_filtered.loc[i,'song_name']
    net.add_node(song, title=song, label=song, color="skyblue",size = 30)
    for j in df_filtered.loc[i,'song_gn_dtl_gnr_basket']:
        if j==None: break
        net.add_edge(artist+j, song)

net.toggle_hide_edges_on_drag(False)
net.show_buttons(filter_=['physics'])
net.show("iu_network.html")

with open("iu_network.html", "r") as f:
    html_content = f.read()
components.html(html_content, height=600)