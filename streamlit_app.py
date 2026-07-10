import streamlit as st

st.set_page_config(
    page_title="YouComments",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Общий фон */
    .stApp {
        background-color: #343541;
        color: #ECECF1;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #202123;
        border-right: 1px solid #2f3035;
    }

    /* Логотип */
    .app-logo {
        font-size: 42px;
        text-align: center;
        margin-top: 20px;
    }

    .app-title {
        font-size: 20px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 20px;
        color: #ECECF1;
    }

    /* Навигация */
    [data-testid="stSidebarNav"] a {
        border-radius: 8px;
        padding: 10px 14px !important;
        margin: 4px 8px;
        color: #ECECF1;
        transition: 0.2s;
    }

    [data-testid="stSidebarNav"] a:hover {
        background-color: #2A2B32;
    }

    [data-testid="stSidebarNav"] [aria-current="page"] {
        background-color: #343541;
        border: 1px solid #565869;
    }

    /* Разделитель */
    .sidebar-divider {
        border-top: 1px solid #2f3035;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="app-logo">💬</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-title">YouComments</div>', unsafe_allow_html=True)
    
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    
    st.caption("Генератор комментариев на YouTube")
    st.caption("🤖 Powered by LLM")



def generate_comments():
    pass



pages = {
    "Основное": [
        st.Page(
            "pages/main_page.py",
            title="YouComment",
            icon="💬",
        ),
    ],



    "О проекте": [
        st.Page(
            "pages/paper.py",
            title="Paper",
            icon="📄"
        ),
    ],
}

pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()