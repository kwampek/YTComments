import streamlit as st

@st.cache_resource
def load_css(file_path: str = "style.css"):
    """Загружает CSS-файл и применяет его ко всему приложению"""
    try:
        with open(file_path, encoding="utf-8") as f:
            css = f.read()
        

        st.markdown(f"""
        <style>
        {css}
        </style>
        """, unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.error(f"Файл {file_path} не найден!")
    except Exception as e:
        st.error(f"Ошибка при загрузке CSS: {e}")
