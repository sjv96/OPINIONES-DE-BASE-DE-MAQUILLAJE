import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

nltk.download('stopwords')
nltk.download('punkt')

# Cargar datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("opiniones_clientes.csv")

df = cargar_datos()

# Modelo de sentimientos
modelo = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(modelo)
model = AutoModelForSequenceClassification.from_pretrained(modelo)
clasificador = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Función para clasificar sentimiento
def interpretar(label):
    estrellas = int(label[0])
    if estrellas >= 4:
        return "Positivo"
    elif estrellas == 3:
        return "Neutro"
    else:
        return "Negativo"

# Layout
st.title("📝 Análisis de Opiniones de Clientes")
st.write("Este dashboard permite analizar sentimientos y extraer palabras clave de comentarios de clientes.")

# Mostrar primeras opiniones
st.subheader("Primeras 20 opiniones")
st.dataframe(df.head(20))

# Procesar comentarios
opiniones = df['Opinion'].astype(str).tolist()
texto = " ".join(opiniones).lower()
texto = re.sub(r'[^\w\s]', '', texto)
palabras = texto.split()
stop_words = set(stopwords.words('spanish'))
palabras_filtradas = [p for p in palabras if p not in stop_words and len(p) > 2]

# WordCloud
nube = WordCloud(width=800, height=400, background_color='white').generate(" ".join(palabras_filtradas))
st.subheader("Nube de Palabras")
st.image(nube.to_array(), use_column_width=True)

# Barras de palabras
conteo = Counter(palabras_filtradas)
etiquetas, valores = zip(*conteo.most_common(10))
fig, ax = plt.subplots()
ax.bar(etiquetas, valores, color='pink')
plt.xticks(rotation=45)
st.subheader("Top 10 Palabras Más Frecuentes")
st.pyplot(fig)

# Sentimientos
resultados = clasificador(opiniones)
df['Sentimiento'] = [interpretar(r['label']) for r in resultados]
st.subheader("Distribución de Sentimientos")
conteo_sent = df['Sentimiento'].value_counts()
st.write(conteo_sent)
fig2, ax2 = plt.subplots()
ax2.pie(conteo_sent, labels=conteo_sent.index, autopct='%1.1f%%', colors=['pink', 'violet', 'skyblue'])
ax2.axis('equal')
st.pyplot(fig2)

# Comentario nuevo
st.subheader("Analiza tu propio comentario")
comentario_usuario = st.text_input("Escribe tu comentario aquí")

if comentario_usuario:
    resultado = clasificador(comentario_usuario)[0]
    sentimiento = interpretar(resultado['label'])
    palabras_usuario = re.findall(r'\b\w+\b', comentario_usuario.lower())
    claves = [w for w in palabras_usuario if w not in stop_words][:5]

    st.write(f"**✅ Sentimiento detectado:** {sentimiento}")
    st.write(f"**🔍 Palabras clave:** {', '.join(claves) if claves else 'No se encontraron'}")
