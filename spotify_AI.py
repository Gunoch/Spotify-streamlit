import os
import base64
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import requests

load_dotenv()

# --- CONFIGURACAO DA PAGINA ---
st.set_page_config(
    page_title="Spotify AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ESTILO MODERN GLASS (VIDRO & DEGRADE) ---
st.markdown("""
    <style>
    /* Fundo geral com degrade moderno */
    .stApp {
        background: linear-gradient(to bottom right, #1a1a1a, #2d3436);
        color: #FFFFFF;
    }
    
    /* Sidebar: vidro escuro */
    section[data-testid="stSidebar"] {
        background-color: rgba(17, 17, 17, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Estilizacao das abas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #d3d3d3;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(29, 185, 84, 0.2) !important;
        color: #1DB954 !important;
        border: 1px solid #1DB954 !important;
    }
    
    /* Botoes com degrade verde (Spotify Glow) */
    .stButton>button {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        color: white;
        border-radius: 30px;
        border: none;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.4);
        font-weight: bold;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(29, 185, 84, 0.6);
    }
    
    /* Cartoes glassmorphism (efeito vidro) */
    .song-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 15px;
        transition: transform 0.3s, background 0.3s;
    }
    .song-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(29, 185, 84, 0.5);
    }
    
    /* Texto e titulos */
    h1, h2, h3 {
        color: #FFFFFF !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Barra de progresso arredondada */
    .stProgress > div > div > div > div {
        background-color: #1DB954;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. CARREGAMENTO DE DADOS ---
@st.cache_data
def load_data():
    data = {
        'Musica': ['Bohemian Rhapsody', 'Blinding Lights', 'Shape of You', 'Smells Like Teen Spirit', 'Hotel California', 'Someone Like You', 'Uptown Funk', 'Imagine', 'Bad Guy', 'Despacito', 'Lose Yourself', 'Hallelujah', 'Levitating', 'Rolling in the Deep', 'Sweet Child O\' Mine', 'Wonderwall', 'Thriller', 'Heroes', 'Respect', 'Dancing Queen', 'Numb', 'Enter Sandman', 'Toxic', 'Happy', 'Comfortably Numb', 'Billie Jean', 'Stayin\' Alive', 'I Will Always Love You', 'Gangsta\'s Paradise', 'Shallow', 'Old Town Road', 'Believer', 'Senorita', 'Bad Romance', 'Take on Me'],
        'Artista': ['Queen', 'The Weeknd', 'Ed Sheeran', 'Nirvana', 'Eagles', 'Adele', 'Mark Ronson', 'John Lennon', 'Billie Eilish', 'Luis Fonsi', 'Eminem', 'Leonard Cohen', 'Dua Lipa', 'Adele', 'Guns N\' Roses', 'Oasis', 'Michael Jackson', 'David Bowie', 'Aretha Franklin', 'ABBA', 'Linkin Park', 'Metallica', 'Britney Spears', 'Pharrell Williams', 'Pink Floyd', 'Michael Jackson', 'Bee Gees', 'Whitney Houston', 'Coolio', 'Lady Gaga', 'Lil Nas X', 'Imagine Dragons', 'Camila Cabello', 'Lady Gaga', 'a-ha'],
        'Genero': ['Rock', 'Pop', 'Pop', 'Rock', 'Rock', 'Pop', 'Funk', 'Classic', 'Pop', 'Latin', 'Hip-Hop', 'Folk', 'Pop', 'Pop', 'Rock', 'Rock', 'Pop', 'Classic', 'R&B', 'Pop', 'Rock', 'Metal', 'Pop', 'Pop', 'Rock', 'Pop', 'Disco', 'R&B', 'Hip-Hop', 'Pop', 'Hip-Hop', 'Rock', 'Latin', 'Pop', 'Pop'],
        'Vibracao': [0.39, 0.51, 0.82, 0.50, 0.58, 0.29, 0.86, 0.30, 0.70, 0.76, 0.69, 0.20, 0.70, 0.40, 0.45, 0.38, 0.77, 0.45, 0.75, 0.78, 0.50, 0.45, 0.79, 0.90, 0.25, 0.85, 0.80, 0.30, 0.75, 0.57, 0.88, 0.77, 0.75, 0.70, 0.57],
        'Humor': [0.90, 0.80, 0.70, 0.90, 0.50, 0.30, 0.90, 0.20, 0.40, 0.80, 0.90, 0.10, 0.80, 0.60, 0.90, 0.70, 0.90, 0.60, 0.70, 0.70, 0.95, 0.95, 0.80, 0.80, 0.40, 0.75, 0.75, 0.25, 0.60, 0.38, 0.60, 0.78, 0.70, 0.92, 0.90],
        'Vibe': [0.50, 0.30, 0.80, 0.20, 0.60, 0.20, 0.90, 0.70, 0.30, 0.90, 0.60, 0.10, 0.80, 0.40, 0.60, 0.40, 0.80, 0.50, 0.80, 0.95, 0.20, 0.30, 0.70, 0.95, 0.15, 0.80, 0.90, 0.20, 0.40, 0.30, 0.60, 0.50, 0.55, 0.70, 0.80]
    }
    return pd.DataFrame(data)

df = load_data()


# --- 1b. SPOTIFY API HELPERS ---
def _get_basic_token(client_id: str, client_secret: str) -> Optional[str]:
    token_url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {"Authorization": f"Basic {auth_header}"}
    data = {"grant_type": "client_credentials"}
    resp = requests.post(token_url, headers=headers, data=data, timeout=10)
    if resp.status_code != 200:
        return None
    return resp.json().get("access_token")


def _search_track(query: str, token: str) -> Optional[Dict[str, Any]]:
    url = "https://api.spotify.com/v1/search"
    params = {"q": query, "type": "track", "limit": 1}
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if resp.status_code != 200:
        return None
    items = resp.json().get("tracks", {}).get("items", [])
    if not items:
        return None
    return items[0]


def _recommendations(seed_track_id: str, token: str, limit: int, targets: Dict[str, float]) -> List[Dict[str, Any]]:
    url = "https://api.spotify.com/v1/recommendations"
    params = {"seed_tracks": seed_track_id, "limit": limit}
    params.update(targets)
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if resp.status_code != 200:
        return []
    return resp.json().get("tracks", [])

# --- HEADER ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=70)
with col_title:
    st.title("Spotify AI Recommender")

st.markdown("---")

# --- 2. SIDEBAR (Controles) ---
with st.sidebar:
    st.header("🎛️ Seu Mix")
    st.subheader("Credenciais Spotify")
    default_client_id = os.getenv("SPOTIFY_CLIENT_ID", "")
    default_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    client_id = st.text_input("Client ID", value=default_client_id)
    client_secret = st.text_input("Client Secret", type="password", value=default_client_secret)
    st.markdown("---")
    st.subheader("Referencia")
    ref_track_query = st.text_input("Digite uma musica que voce curta", value="")
    st.caption("Usaremos essa faixa como semente para recomendar similares via API.")
    st.markdown("---")
    humor_choice = st.radio(
        "⚡ Humor (intensidade emocional)",
        ["Calmo", "Equilibrado", "Intenso", "Explosivo"],
        index=2,
        help="Escolha o nivel de energia emocional."
    )
    vibe_choice = st.radio(
        "😊 Vibe (positividade)",
        ["Melancolico", "Neutro", "Positivo", "Radiante"],
        index=2,
        help="Qual o clima geral das faixas?"
    )
    mood_choice = st.radio(
        "🧠 Humor (calmo <-> euforico)",
        ["Relaxado", "Focado", "Festivo", "Euforico"],
        index=2,
        help="Grau de agitacao desejado."
    )
    ritmo_choice = st.radio(
        "🎚️ Ritmo / Groove",
        ["Suave", "Groove", "Dancante", "Porradao"],
        index=2,
        help="Quao marcada deve ser a batida."
    )

    humor_map = {"Calmo": 0.25, "Equilibrado": 0.5, "Intenso": 0.75, "Explosivo": 0.95}
    vibe_map = {"Melancolico": 0.2, "Neutro": 0.5, "Positivo": 0.7, "Radiante": 0.9}
    mood_map = {"Relaxado": 0.35, "Focado": 0.55, "Festivo": 0.72, "Euforico": 0.92}
    ritmo_map = {"Suave": 0.35, "Groove": 0.55, "Dancante": 0.75, "Porradao": 0.92}

    input_humor = humor_map[humor_choice]
    input_vibe = vibe_map[vibe_choice]
    input_mood = mood_map[mood_choice]
    input_ritmo = ritmo_map[ritmo_choice]
    feature_vibracao = (input_mood + input_ritmo) / 2
    feature_vibe = input_vibe
    
    st.markdown("---")
    with st.expander("Configuracoes Avancadas"):
        n_neighbors = st.slider("Qtd. de sugestoes", 2, 8, 4)
        genero_filter = st.multiselect("Filtrar Genero", sorted(df['Genero'].unique()))
    
    st.write("")
    btn_processar = st.button("🚀 Buscar no Spotify", use_container_width=True)

# --- 3. ESTRUTURA DE ABAS ---
tab1, tab2, tab3 = st.tabs(["🎵 Playlist", "📊 Analise de Dados", "ℹ️ Como Funciona"])

# --- TAB 1: PLAYLIST (PRINCIPAL) ---
with tab1:
    # fallback dados locais
    if genero_filter:
        df_modelo = df[df['Genero'].isin(genero_filter)].reset_index(drop=True)
    else:
        df_modelo = df

    k_final = min(n_neighbors, len(df_modelo))

    if btn_processar:
        if not client_id or not client_secret:
            st.error("Informe Client ID e Client Secret do Spotify.")
            st.stop()
        if not ref_track_query:
            st.error("Digite uma musica de referencia.")
            st.stop()

        with st.spinner("Conectando ao Spotify..."):
            token = _get_basic_token(client_id, client_secret)
        if not token:
            st.error("Nao foi possivel autenticar na API do Spotify. Verifique credenciais.")
            st.stop()

        with st.spinner("Buscando faixa de referencia..."):
            seed_track = _search_track(ref_track_query, token)
        if not seed_track:
            st.error("Nao encontrei essa faixa no Spotify. Tente outro nome.")
            st.stop()

        seed_id = seed_track["id"]
        targets = {
            "target_energy": round(input_humor, 2),
            "target_valence": round(feature_vibe, 2),
            "target_danceability": round(feature_vibracao, 2),
        }

        with st.spinner("Gerando recomendacoes..."):
            recs = _recommendations(seed_id, token, limit=k_final, targets=targets)

        st.subheader("🎧 Sua Playlist Spotify")
        st.caption(f"Semente: {seed_track['name']} — {seed_track['artists'][0]['name']}")

        if not recs:
            st.warning("Nao veio nada da API. Exibindo sugestoes locais.")
        tracks_to_show = recs if recs else []

        if not tracks_to_show and len(df_modelo) > 0:
            # fallback ML local
            X = df_modelo[['Vibracao', 'Humor', 'Vibe']].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = NearestNeighbors(n_neighbors=k_final, algorithm='brute', metric='euclidean')
            model.fit(X_scaled)
            user_vector = np.array([[feature_vibracao, input_humor, feature_vibe]])
            user_vector_scaled = scaler.transform(user_vector)
            distances, indices = model.kneighbors(user_vector_scaled)
            for idx in indices[0]:
                musica = df_modelo.iloc[idx]
                tracks_to_show.append({
                    "name": musica["Musica"],
                    "artists": [{"name": musica["Artista"]}],
                    "album": {"images": [{"url": f"https://placehold.co/300x300/2b2b2b/1DB954?text={musica['Genero']}"}]},
                    "match": None,
                })

        cols = st.columns(k_final)
        for i, track in enumerate(tracks_to_show[:k_final]):
            name = track["name"]
            artist = ", ".join([a["name"] for a in track.get("artists", [])])
            images = track.get("album", {}).get("images", [])
            cover = images[0]["url"] if images else "https://placehold.co/300x300/2b2b2b/1DB954?text=Spotify"
            if i < len(cols):
                with cols[i]:
                    st.markdown(f"""
                    <div class="song-card">
                        <img src="{cover}" style="width:100%; border-radius:10px; margin-bottom:10px;">
                        <div style="font-weight:bold; font-size:16px; margin-bottom:5px;">{name}</div>
                        <div style="color:#b3b3b3; font-size:14px; margin-bottom:10px;">{artist}</div>
                    </div>
                    """, unsafe_allow_html=True)
        st.write("---")
        with st.expander("🔍 Ver comparacao visual (radar)"):
            fig = go.Figure()
            categories = ['Mood/Ritmo (media)', 'Humor', 'Vibe']
            fig.add_trace(go.Scatterpolar(
                r=[feature_vibracao, input_humor, feature_vibe],
                theta=categories,
                fill='toself',
                name='Teu vibe',
                line=dict(color='#1DB954', width=3),
                marker=dict(size=8),
                fillcolor='rgba(29, 185, 84, 0.3)'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, 
                        range=[0, 1], 
                        showticklabels=False, 
                        gridcolor='rgba(255, 255, 255, 0.2)',
                        linecolor='rgba(255, 255, 255, 0.2)'
                    ),
                    angularaxis=dict(
                        gridcolor='rgba(255, 255, 255, 0.2)',
                        linecolor='rgba(255, 255, 255, 0.2)',
                        tickfont=dict(size=14, color='white', family='Arial Black')
                    ),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=40, r=40, t=20, b=20),
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=0,
                    xanchor="center",
                    x=0.5,
                    bgcolor='rgba(0,0,0,0)'
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.info("👈 Ajuste seus gostos na barra lateral, escolha uma musica de referencia e clique em BUSCAR.")
        st.markdown("### Tendencias no banco de dados")
        st.dataframe(df.sample(5), use_container_width=True)

# --- TAB 2: ANALISE DE DADOS (EDA) ---
with tab2:
    st.header("Analise exploratoria de dados")
    st.markdown("Visao geral estatistica do banco de dados musical.")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Distribuicao de generos")
        fig_pie = px.pie(df, names='Genero', hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=False)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.subheader("Humor vs Vibe")
        fig_scatter = px.scatter(df, x='Humor', y='Vibe', color='Genero', size='Vibracao', hover_data=['Musica'])
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(255,255,255,0.05)', 
            font_color='white', 
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'), 
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.subheader("Matriz de correlacao")
    corr = df[['Vibracao', 'Humor', 'Vibe']].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='Greens')
    fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 3: SOBRE O ALGORITMO ---
with tab3:
    st.header("Como funciona a IA?")
    
    st.markdown("""
    ### O algoritmo K-Nearest Neighbors (KNN)
    
    Este sistema usa um modelo de **aprendizado de maquina** que classifica itens pela proximidade geometrica.
    
    1. **Mapeamento:** Cada musica vira coordenadas (X, Y, Z).
    2. **Distancia:** Calculamos a distancia euclidiana entre teu gosto e as faixas disponiveis.
    3. **Normalizacao:** Usamos `StandardScaler` para dar o mesmo peso a todas as variaveis.

    ### Integracao Spotify
    - Autenticacao: usa Client ID/Secret (app Spotify) via Client Credentials para pegar um token.
    - Busca: achamos a faixa digitada na busca do Spotify.
    - Recomendacoes: chamamos `/v1/recommendations` com a faixa como semente e os alvos (energia, valence, danceability) derivados dos controles.
    - Fallback: se a API falhar, mostramos sugestoes locais do conjunto mockado.
    """)
    st.info("💡 Dica: Experimente mudar os controles radicalmente para ver como a recomendacao muda completamente!")
