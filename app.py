import streamlit as st
import tiktoken
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
from groq import Groq


st.set_page_config(
    page_title="Desmontando los LLMs",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Taller: Desmontando los LLMs")
st.markdown("**Universidad EAFIT – Deep Learning y Arquitecturas Transformer**")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔤 Módulo 1: Tokenizador",
    "📐 Módulo 2: Embeddings",
    "⚡ Módulo 3: Groq API",
    "📊 Módulo 4: Métricas"
])

with tab1:
    st.header("Laboratorio del Tokenizador")
    
    texto = st.text_area(
        "Ingresa un texto para tokenizar:",
        value="Los transformers revolucionaron el procesamiento del lenguaje natural."
    )
    
    modelo_enc = st.selectbox(
        "Codificación:", ["cl100k_base", "p50k_base", "r50k_base"]
    )

    if st.button("Tokenizar"):
        enc = tiktoken.get_encoding(modelo_enc)
        token_ids = enc.encode(texto)
        tokens = [enc.decode([tid]) for tid in token_ids]

        # Métrica comparativa
        col1, col2, col3 = st.columns(3)
        col1.metric("Caracteres", len(texto))
        col2.metric("Tokens", len(token_ids))
        col3.metric("Ratio char/token", f"{len(texto)/len(token_ids):.2f}")

        # Visualización con colores alternos
        st.subheader("Tokens coloreados:")
        colores = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
        html = ""
        for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
            color = colores[i % len(colores)]
            html += f'<span style="background-color:{color};padding:2px 4px;margin:2px;border-radius:4px;font-family:monospace">{tok}<sub style="font-size:10px">{tid}</sub></span>'
        st.markdown(html, unsafe_allow_html=True)

        # Tabla de mapeo
        st.subheader("Mapeo Token → ID:")
        
        df = pd.DataFrame({"Token": tokens, "Token ID": token_ids, "Longitud": [len(t) for t in tokens]})
        st.dataframe(df, use_container_width=True)

with tab2:
    st.header("Geometría de las Palabras")

    palabras_default = "rey, reina, hombre, mujer, París, Francia, Madrid, España, perro, gato"
    palabras_input = st.text_input("Ingresa palabras separadas por coma:", palabras_default)
    
    if st.button("Visualizar Embeddings"):
        palabras = [p.strip() for p in palabras_input.split(",")]
        
        with st.spinner("Calculando embeddings..."):
            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            embeddings = model.encode(palabras)
        
        # PCA a 2D
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        
        df = pd.DataFrame({
            "Palabra": palabras,
            "PC1": coords[:, 0],
            "PC2": coords[:, 1]
        })
        
        fig = px.scatter(
            df, x="PC1", y="PC2", text="Palabra",
            title="Espacio de Embeddings (PCA 2D)",
            template="plotly_dark"
        )
        fig.update_traces(textposition="top center", marker=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"Varianza explicada: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")

with tab3:
    st.header("Inferencia y Razonamiento")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        modelo = st.selectbox("Modelo:", [
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mistral-saba-24b"
        ])
        temperatura = st.slider("🌡️ Temperatura", 0.0, 2.0, 0.7, 0.1)
        top_p = st.slider("🎯 Top-P", 0.0, 1.0, 0.9, 0.05)
        max_tokens = st.slider("Max Tokens", 100, 1000, 500)

        st.info(
            "**Temperatura baja (<0.3):** respuestas deterministas\n\n"
            "**Temperatura alta (>0.7):** más creatividad y variedad"
        )

    with col2:
        system_prompt = st.text_area(
            "System Prompt:", 
            "Eres un asistente experto en inteligencia artificial. Responde de forma concisa.",
            height=100
        )
        user_prompt = st.text_area("User Prompt:", "Explica qué es un transformer en 3 oraciones.", height=100)

        if st.button("🚀 Generar respuesta"):
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            
            with st.spinner("Generando..."):
                import time
                start = time.time()
                response = client.chat.completions.create(
                    model=modelo,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperatura,
                    top_p=top_p,
                    max_tokens=max_tokens
                )
                elapsed = time.time() - start

            st.markdown("### Respuesta:")
            st.write(response.choices[0].message.content)
            
            # Guardar métricas en session_state para el Módulo 4
            st.session_state["last_metrics"] = {
                "usage": response.usage,
                "elapsed": elapsed,
                "model": modelo
            }

with tab4:
    st.header("Métricas de Desempeño")

    if "last_metrics" not in st.session_state:
        st.warning("Primero genera una respuesta en el Módulo 3.")
    else:
        m = st.session_state["last_metrics"]
        usage = m["usage"]
        elapsed = m["elapsed"]
        
        total_tokens = usage.total_tokens
        output_tokens = usage.completion_tokens
        input_tokens = usage.prompt_tokens
        
        throughput = output_tokens / elapsed if elapsed > 0 else 0
        ms_per_token = (elapsed * 1000) / output_tokens if output_tokens > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⏱️ ms/token", f"{ms_per_token:.1f}")
        col2.metric("🚀 Tokens/seg", f"{throughput:.1f}")
        col3.metric("📥 Tokens entrada", input_tokens)
        col4.metric("📤 Tokens salida", output_tokens)
        
        # Gráfico de distribución de tokens
        fig = px.pie(
            names=["Input Tokens", "Output Tokens"],
            values=[input_tokens, output_tokens],
            title=f"Distribución de Tokens – {m['model']}"
        )
        st.plotly_chart(fig, use_container_width=True)