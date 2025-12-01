import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Spotify AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ESTILO MODERN GLASS (VIDRO & DEGRADÊ) ---
st.markdown("""
    <style>
    /* Fundo Geral com Degradê Moderno */
    .stApp {
        background: linear-gradient(to bottom right, #1a1a1a, #2d3436);
        color: #FFFFFF;
    }
    
    /* Sidebar: Vidro Escuro */
    section[data-testid="stSidebar"] {
        background-color: rgba(17, 17, 17, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Estilização das Abas */
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
    
    /* Botões com Degradê Verde (Spotify Glow) */
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
    
    /* Cartões Glassmorphism (Efeito Vidro) */
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
    
    /* Texto e Títulos */
    h1, h2, h3 {
        color: #FFFFFF !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Barra de Progresso Arredondada */
    .stProgress > div > div > div > div {
        background-color: #1DB954;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. CARREGAMENTO DE DADOS ---
@st.cache_data
def load_data():
    # =========================================================================
    # 🧪 DADOS MOCKADOS
    # =========================================================================
    data = {
        'Musica': ['Bohemian Rhapsody', 'Blinding Lights', 'Shape of You', 'Smells Like You', 'Smells Like Teen Spirit', 'Hotel California', 'Someone Like You', 'Uptown Funk', 'Imagine', 'Bad Guy', 'Despacito', 'Lose Yourself', 'Hallelujah', 'Levitating', 'Rolling in the Deep', 'Sweet Child O\' Mine', 'Wonderwall', 'Thriller', 'Heroes', 'Respect', 'Dancing Queen', 'Numb', 'Enter Sandman', 'Toxic', 'Happy', 'Comfortably Numb', 'Billie Jean', 'Stayin\' Alive', 'I Will Always Love You', 'Gangsta\'s Paradise', 'Shallow', 'Old Town Road', 'Believer', 'Senorita', 'Bad Romance', 'Take on Me'],
        'Artista': ['Queen', 'The Weeknd', 'Ed Sheeran', 'Nirvana', 'Eagles', 'Adele', 'Mark Ronson', 'John Lennon', 'Billie Eilish', 'Luis Fonsi', 'Eminem', 'Leonard Cohen', 'Dua Lipa', 'Adele', 'Guns N\' Roses', 'Oasis', 'Michael Jackson', 'David Bowie', 'Aretha Franklin', 'ABBA', 'Linkin Park', 'Metallica', 'Britney Spears', 'Pharrell Williams', 'Pink Floyd', 'Michael Jackson', 'Bee Gees', 'Whitney Houston', 'Coolio', 'Lady Gaga', 'Lil Nas X', 'Imagine Dragons', 'Camila Cabello', 'Lady Gaga', 'a-ha'],
        'Genero': ['Rock', 'Pop', 'Pop', 'Rock', 'Rock', 'Pop', 'Funk', 'Classic', 'Pop', 'Latin', 'Hip-Hop', 'Folk', 'Pop', 'Pop', 'Rock', 'Rock', 'Pop', 'Classic', 'R&B', 'Pop', 'Rock', 'Metal', 'Pop', 'Pop', 'Rock', 'Pop', 'Disco', 'R&B', 'Hip-Hop', 'Pop', 'Hip-Hop', 'Rock', 'Latin', 'Pop', 'Pop'],
        'Dancabilidade': [0.39, 0.51, 0.82, 0.50, 0.58, 0.29, 0.86, 0.30, 0.70, 0.76, 0.69, 0.20, 0.70, 0.40, 0.45, 0.38, 0.77, 0.45, 0.75, 0.78, 0.50, 0.45, 0.79, 0.90, 0.25, 0.85, 0.80, 0.30, 0.75, 0.57, 0.88, 0.77, 0.75, 0.70, 0.57],
        'Energia':       [0.90, 0.80, 0.70, 0.90, 0.50, 0.30, 0.90, 0.20, 0.40, 0.80, 0.90, 0.10, 0.80, 0.60, 0.90, 0.70, 0.90, 0.60, 0.70, 0.70, 0.95, 0.95, 0.80, 0.80, 0.40, 0.75, 0.75, 0.25, 0.60, 0.38, 0.60, 0.78, 0.70, 0.92, 0.90],
        'Vibe':          [0.50, 0.30, 0.80, 0.20, 0.60, 0.20, 0.90, 0.70, 0.30, 0.90, 0.60, 0.10, 0.80, 0.40, 0.60, 0.40, 0.80, 0.50, 0.80, 0.95, 0.20, 0.30, 0.70, 0.95, 0.15, 0.80, 0.90, 0.20, 0.40, 0.30, 0.60, 0.50, 0.55, 0.70, 0.80]
    }
    return pd.DataFrame(data)

df = load_data()

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
    input_energia = st.slider("⚡ Energia", 0.0, 1.0, 0.6)
    input_vibe = st.slider("😊 Vibe (Positividade)", 0.0, 1.0, 0.7)
    input_danca = st.slider("💃 Dancabilidade", 0.0, 1.0, 0.8)
    
    st.markdown("---")
    with st.expander("Configurações Avançadas"):
        n_neighbors = st.slider("Qtd. de Sugestões", 2, 8, 4)
        genero_filter = st.multiselect("Filtrar Gênero", sorted(df['Genero'].unique()))
    
    st.write("")
    btn_processar = st.button("🚀 GERAR PLAYLIST", use_container_width=True)

# --- 3. ESTRUTURA DE ABAS ---
tab1, tab2, tab3 = st.tabs(["🎵 Playlist", "📊 Análise de Dados", "ℹ️ Como Funciona"])

# --- TAB 1: PLAYLIST (PRINCIPAL) ---
with tab1:
    if genero_filter:
        df_modelo = df[df['Genero'].isin(genero_filter)].reset_index(drop=True)
    else:
        df_modelo = df

    # Lógica de ML
    k_final = min(n_neighbors, len(df_modelo))
    X = df_modelo[['Dancabilidade', 'Energia', 'Vibe']].values
    scaler = StandardScaler()
    
    if len(X) > 0:
        X_scaled = scaler.fit_transform(X)
        model = NearestNeighbors(n_neighbors=k_final, algorithm='brute', metric='euclidean')
        model.fit(X_scaled)
    else:
        st.error("Nenhuma música com esse filtro.")
        st.stop()

    if btn_processar:
        user_vector = np.array([[input_danca, input_energia, input_vibe]])
        user_vector_scaled = scaler.transform(user_vector)
        distances, indices = model.kneighbors(user_vector_scaled)
        
        st.subheader("🎧 Sua Playlist Personalizada")
        st.write("") # Espaçamento
        
        cols = st.columns(k_final)
        
        for i, idx in enumerate(indices[0]):
            musica = df_modelo.iloc[idx]
            match_score = max(0, min(100, (1 - distances[0][i]) * 100))
            img_url = f"https://placehold.co/300x300/2b2b2b/1DB954?text={musica['Genero']}"
            
            if i < len(cols):
                with cols[i]:
                    # Início do Cartão "Vidro" (HTML puro para estilo avançado)
                    st.markdown(f"""
                    <div class="song-card">
                        <img src="{img_url}" style="width:100%; border-radius:10px; margin-bottom:10px;">
                        <div style="font-weight:bold; font-size:16px; margin-bottom:5px;">{musica['Musica']}</div>
                        <div style="color:#b3b3b3; font-size:14px; margin-bottom:10px;">{musica['Artista']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Barra de progresso fora do HTML puro para usar o componente nativo do Streamlit
                    cor_match = "#1DB954" if match_score > 70 else "#f1c40f"
                    st.markdown(f"<div style='display:flex; justify-content:space-between; font-size:12px;'><span>Match</span><span style='color:{cor_match}'>{int(match_score)}%</span></div>", unsafe_allow_html=True)
                    st.progress(int(match_score))
                    
        # Gráfico Radar
        st.write("---")
        with st.expander("🔍 Ver Comparação Visual (Radar Chart)"):
            top_rec = df_modelo.iloc[indices[0][0]]
            fig = go.Figure()
            categories = ['Dancabilidade', 'Energia', 'Vibe']
            
            # Trace do Usuário
            fig.add_trace(go.Scatterpolar(
                r=[input_danca, input_energia, input_vibe],
                theta=categories,
                fill='toself',
                name='Teu Vibe',
                line=dict(color='#1DB954', width=3),
                marker=dict(size=8),
                fillcolor='rgba(29, 185, 84, 0.3)'
            ))
            
            # Trace da Música Recomendada
            fig.add_trace(go.Scatterpolar(
                r=[top_rec['Dancabilidade'], top_rec['Energia'], top_rec['Vibe']],
                theta=categories,
                fill='toself',
                name=f"Top 1: {top_rec['Musica']}",
                line=dict(color='#ffffff', width=2, dash='dot'),
                marker=dict(size=6, symbol='diamond'),
                fillcolor='rgba(255, 255, 255, 0.1)'
            ))
            
            # Fundo transparente e Estilização de Eixos
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
        st.info("👈 Ajuste seus gostos na barra lateral e clique em **GERAR PLAYLIST**.")
        st.markdown("### Tendências no Banco de Dados")
        st.dataframe(df.sample(5), use_container_width=True)

# --- TAB 2: ANÁLISE DE DADOS (EDA) ---
with tab2:
    st.header("Análise Exploratória de Dados")
    st.markdown("Visão geral estatística do banco de dados musical.")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Distribuição de Gêneros")
        fig_pie = px.pie(df, names='Genero', hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=False)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.subheader("Energia vs Vibe")
        fig_scatter = px.scatter(df, x='Energia', y='Vibe', color='Genero', size='Dancabilidade', hover_data=['Musica'])
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(255,255,255,0.05)', 
            font_color='white', 
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'), 
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.subheader("Matriz de Correlação")
    corr = df[['Dancabilidade', 'Energia', 'Vibe']].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='Greens')
    fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 3: SOBRE O ALGORITMO ---
with tab3:
    st.header("Como funciona a IA?")
    
    st.markdown("""
    ### O Algoritmo K-Nearest Neighbors (KNN)
    
    Este sistema utiliza um modelo de **Aprendizado de Máquina** que classifica objetos baseados na proximidade geométrica.
    
    1. **Mapeamento:** Cada música é convertida em coordenadas (X, Y, Z).
    2. **Distância:** O algoritmo calcula a distância matemática (Euclidiana) entre o teu gosto e as músicas disponíveis.
    3. **Normalização:** Usamos `StandardScaler` para garantir que todas as variáveis tenham o mesmo peso.
    """)
    st.info("💡 Dica: Experimenta mudar os sliders radicalmente para veres como a recomendação muda completamente!")
