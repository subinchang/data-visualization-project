import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from collections import OrderedDict
import random
from itertools import islice
from wordcloud import WordCloud 
from PIL import Image

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
st.title("🎵 김멜론의 플레이리스트 🎵")

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


###########################################################
# 랜덤 태그 가수, 장르, 워드클라우드, 유사 태그
# big, small 만들기
def genre_seperate(x):
    if x[-2:] == "00":
        value = "big"
    else:
        value = "small"
    return value

df_genre["big_small"] = df_genre["Code"].map(genre_seperate)

# 플레이리스트 데이터 할당 함수
def playlist_choose(df_pl: pd.DataFrame, df_meta: pd.DataFrame, x: list):
    """
    X 안에는 플레이리스트의 id가 들어온다.
    tags -> 플레이 리스트들의 태그 정보들의 list
    pl_title -> 플레이 리스트들의 제목들의 list
    song_id -> 플레이 리스트들의 노래의 id list
    genre_id -> 플레이 리스트들의 여러 장르들의 nested list(한곡에 여러가지 list가 있을수 있어서)
    artist_names -> 플레이 리스트들의 여러 아티스트 들의 nested list(한곡에 여러가지 list가 있을수 있어서)
    song_names ->  플레이 리스트들의의 노래 이름 list
    artist_count -> 플레이스트 안에 아트스트의 비율(ordered dict)
    """
    # 플레이리스트에서만 뽑아낸 데이터
    tags = []
    pl_title = []
    song_id = []
    for i in x:
        tags.extend(df_pl.loc[df_pl["id"]==i, "tags"].to_list()[0])
        pl_title.append(df_pl.loc[df_pl["id"]==i, "plylst_title"].to_list()[0])
        song_id.extend((df_pl.loc[df_pl["id"]==i, "songs"]).to_list()[0])

    # 위의 정보를 통해 메타 데이터 뽑아내기
    genre_id = [df_meta.loc[df_meta.id==song, "song_gn_dtl_gnr_basket"].to_list()[0] for song in song_id] #이중 리스트(대장르, 소장르 포함이라)
    artist_names = [df_meta.loc[df_meta.id==song, "artist_name_basket"].to_list()[0] for song in song_id] #이중 리스트
    song_names = [df_meta.loc[df_meta.id==song, "song_name"].values[0] for song in song_id]# 이거 그냥 리스트트
    album_list = [df_meta.loc[df_meta.id==song, "album_id"].values[0] for song in song_id]# 이거 그냥 리스트트

    def count_frequency(my_list):
        count = {}
        for item in my_list:
            count[item] = count.get(item, 0) + 1
        count = OrderedDict(sorted(count.items(), key=lambda t:t[1], reverse=True))
            
        return count

    artist_count = count_frequency(sum(artist_names, [])) #sum(artict_names, []) 이차원 리스트 풀기기
        
    return tags, pl_title, song_id, genre_id, artist_names, song_names, artist_count, album_list

# artist 비율 차트 만들기
def art_chart(od: "dict", x: "int"):
    """
    상위 x개만 뽑아서 파이 차트를 그려준다.
    """
    artist = []
    artist_cnt = []

    sliced = islice(od.items(), x) 
    sliced_o = OrderedDict(sliced)

    for k, v in sliced_o.items():
        artist.append(k)
        artist_cnt.append(v)
    
    df = pd.DataFrame({"artist": artist,
                       "artist_cnt": artist_cnt})

    fig = px.pie(df, values="artist_cnt", names="artist", color_discrete_sequence=px.colors.sequential.Greens[::-1], hole=0.3)
    st.plotly_chart(fig, theme=None, use_container_width=True)

# genre chart
def genre_chart(genre_id, df_genre):
    def count_frequency(my_list):
        count = {}
        for item in my_list:
            count[item] = count.get(item, 0) + 1
        count = OrderedDict(sorted(count.items(), key=lambda t:t[1], reverse=True))
            
        return count
    
    genre = sum(genre_id, []) # nested list를 풀어줌
    small_genre_code = df_genre.loc[df_genre["big_small"]=="small", "Code"].values # df_genre로 부터 소장르의 코드만 가져옴
    pl_small_genre = [code for code in genre if (code in small_genre_code)] # 소장르의 코드만 추출해냄
    pl_big_genre = [code[:4]+"00" for code in pl_small_genre] # 소장르의 코드들의 뒷자리 2개를 00으로 바꾸어 대장르의 코드로 변환환

    genre_name = []
    genre_cnt = []
    for k, i in count_frequency(pl_big_genre).items():
        genre_name.append(df_genre.loc[df_genre["Code"] == k, "Genre"].values[0])
        genre_cnt.append(i)
    df = pd.DataFrame({"genre": genre_name,
                       "count": genre_cnt})
    fig = px.pie(df, values="count", names="genre", color_discrete_sequence=px.colors.sequential.Greens[::-1], hole=0.3)
    # fig.update_layout(paper_bgcolor="white")
    st.plotly_chart(fig, theme=None, use_container_width=True)

# WordCloud
def show_word_cloud(tags, img_path):
    def count_frequency(my_list):
        count = {}
        for item in my_list:
            count[item] = count.get(item, 0) + 1

        return count
    
    #워드 클라우드 배경에 필요한 이미지지
    cand_mask=np.array(Image.open(img_path))
    ## 워드클라우드에 출력할 딕셔너리를 만듦.
    words = count_frequency(tags)

    wordcloud = WordCloud(
        font_path = '/Users/admin/opt/anaconda3/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/NanumGothic.ttf', 
        background_color='white', 
        mask=cand_mask, 
    ).generate_from_frequencies(words)

    #사이즈 설정 및 출력
    fig, ax = plt.subplots()
    # plt.figure(figsize=(5,5))
    ax.imshow(wordcloud,interpolation='bilinear')
    ax.axis('off') # 차트로 나오지 않게
    st.pyplot(fig)

############################################################
st.markdown('#### :green[김멜론 님을 위한 플레이리스트 추천]')


with st.expander('Click to get recommended playlist'):
    #임의 random sampling으로 사용자의 플레이리스트 할당
    test = sorted(df_pl["id"])
    random.shuffle(test)
    test_ids = test[:7]
    tags, pl_title, song_id, genre_id, artist_names, song_names, artist_count, album_list = playlist_choose(df_pl, df_meta, test_ids)
    col1, col2, col3 = st.columns(3)

    with col1:
        art_chart(artist_count, 5)
    with col2:
        genre_chart(genre_id, df_genre)
    with col3:
        show_word_cloud(tags, "./data/melon.png")



#############################################################################
def show_recommendation(tags, model_path):

    loaded_model = KeyedVectors.load_word2vec_format(model_path)
    cand = []
    ap_tag = []

    for i in tags:
        try:
            for t_s in loaded_model.most_similar(i):
                if t_s[0] not in ap_tag:
                    cand.append(list(t_s))
                    ap_tag.append(t_s[0])
                else:
                    id = ap_tag.index(t_s[0])
                    cand[id][1] += t_s[1]
        except:
            pass

    cand.sort(key=lambda x: x[1], reverse=True)
    top_5_tags = cand[:5]

    tags = []
    num = []
    for s_t in top_5_tags:
        tags.append(s_t[0])
        num.append(s_t[1])

    num = np.array(num)/sum(num)*100

    df = pd.DataFrame(dict(
        r=num,
        theta=tags))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')
    st.plotly_chart(fig, theme=None, use_container_width=True)

############################################################################3
st.markdown("Word2Vec을 이용한 유사 태그 추천")

model_path = "model/model_1"
with st.expander('Click to get similar tag recommendations'):
    col1, col2 = st.columns(2)
    with col1:
        show_word_cloud(tags, "data/melon.png")
    with col2:
        show_recommendation(tags, model_path)

# words= st.text_input("찾고 싶은 태그", '드라이브')
# if words:
#     # st.write("You entered: ", words)
#     loaded_model = KeyedVectors.load_word2vec_format("model/model_1") # 모델 로드
#     word = words.split(",")
#     model_result = loaded_model.most_similar(word)
#     st.text(model_result)

#########################################################
# 태그 입력하면 관련 플리 목록 보여주기

def show_top(word, df_pl, num):
    def ddlook(x, word):
        if word in x:
            value =  True
        else:
            value = False
        return value

    df = df_pl.copy()
    df[word] = df["plylst_title"].apply(lambda x: ddlook(x, word))
    cols = list(df.columns)
    cols.remove(word)
    df_pt = df.loc[df[word]==True, cols]
    final = df_pt.sort_values(by=['like_cnt'], axis=0, ascending=False).head(num)

    # Figure 생성
    fig = go.Figure()

    #테이블 생성
    cols = ["plylst_title", "like_cnt", "updt_date"]
    fig.add_trace(go.Table(
        header=dict(values=list(cols),
                    fill_color='darkseagreen',
                    align='left',
                    font_size=16,
                    height=30,
                    font_color='black'),
        cells=dict(values=[final.plylst_title, final.like_cnt, final.updt_date],
                fill_color='mintcream',
                font_size=13,
                height=25,
                align='left')))
    fig.update_layout(go.Layout(paper_bgcolor="white"))
    st.plotly_chart(fig, theme=None, width=800)


st.title('')
st.markdown('#### :green[아래 플레이리스트를 들어보세요!]')
title = st.text_input('찾고 싶은 태그', 'ex) 재즈, 신나는, 드라이브')
show_top(title, df_pl, 10)

#########################################################
# 태그 입력하면 관련 아티스트 추천
tsn = pd.read_csv('./data/tagsongname.csv')
def get_artist_frequency(tags, top_n=None):
    
    artists = []
    for tag in tags:
        df = tsn[tsn['tags'].apply(lambda x: tag in x)]
        artists += list(df['artist'])

    artist_freq = {}
    for artist in artists:
        if artist in artist_freq:
            artist_freq[artist] += 1
        else:
            artist_freq[artist] = 1

    sorted_freq = {k: v for k, v in sorted(artist_freq.items(), key=lambda item: item[1], reverse=True)}
    if top_n is not None:
        sorted_freq = dict(list(sorted_freq.items())[:top_n])
    
    return sorted_freq

def scatter_artist_frequency(tags):
    if isinstance(tags, str):
        tags = [tags]

    freq = get_artist_frequency(tags, top_n=10)
    df = pd.DataFrame(list(freq.items()), columns=['artist', 'frequency'])
    fig = px.bar(df, x='frequency', y='artist', color='frequency', color_continuous_scale=px.colors.sequential.algae, orientation='h')
    fig.update_layout(title='해당 태그가 많이 매핑된 아티스트')
    st.write(fig)

st.title('')
st.markdown('#### :green[이런 아티스트는 어때요?]')
tags = st.text_input('찾고 싶은 태그', 'ex) 힙합, Pop')
scatter_artist_frequency(tags)

