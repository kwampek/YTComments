import streamlit as st
import random
import re
import html
import time
import torch

from googleapiclient.discovery import build
from pages.loader import load_css
from models.model import load_youcomment_model

st.set_page_config(layout="wide")

load_css("pages/main_style.css")

if "comments_data" not in st.session_state:
    st.session_state.comments_data = []

if "generated_results" not in st.session_state:
    st.session_state.generated_results = {}

if "id_counter" not in st.session_state:
    st.session_state.id_counter = 0

if "meta" not in st.session_state:
    st.session_state.meta = None

if "model" not in st.session_state:
    st.session_state.model = load_youcomment_model()

default_meta = {
    "video_title": "",
    "video_description": "",
    "channel_title": "",
    "channel_description": "",
    "views": 0
}


model = st.session_state.model
tokenizer = model.tokenizer


def generate_id():
    st.session_state.id_counter += 1
    return st.session_state.id_counter



def extract_video_id(url):
    regex = r"(?:v=|youtu.be/)([a-zA-Z0-9_-]+)"
    match = re.search(regex, url)
    return match.group(1) if match else None


def generate_comment(comment_id, text=None):
    """
    Генерирует ответ на комментарий с помощью модели SteroidGPT
    и сохраняет результат в st.session_state.generated_results
    """
    model = st.session_state.model
    meta = st.session_state.meta

    generated_text = model.generate_comment(meta, text)

    st.session_state.generated_results[comment_id] = {
        "text": generated_text
    }


def render_comments(comments, level=0):
    for comment in comments:
        cid = comment["id"]

        author = html.escape(comment["author"])
        text = st.session_state.generated_results.get(cid, {}).get("text", "")
        likes = comment.get("likes", 0)


        with st.container():
            col1, col2 = st.columns([10, 1])
            with col1:
                st.markdown(f"""
<div class="comment-wrapper" data-level="{level}">
<div class="comment-container" style="margin-left: {level * 50}px;">
<div class="comment-avatar"></div>
<div class="comment-content">
<div class="comment-author">
    {author}
    <span class="comment-meta">• сейчас</span>
</div>
<div class="comment-text">
    {text}
</div>
<div class="comment-actions">
    👍 {likes}
</div>
</div>
</div>
</div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="generate-button-wrapper">', unsafe_allow_html=True)

                if st.button("💬", key=f"generate_reply_{cid}"):
                    reply = {
                        "id": generate_id(),
                        "author": "AI_Bot",
                        "likes": 0,
                        "replies": []
                    }
                    comment["replies"].append(reply)
                    generate_comment(reply["id"], text)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        if comment.get("replies"):
            render_comments(comment["replies"], level + 1)


def get_channel_description(channel_id):
    youtube = build("youtube", "v3", developerKey="AIzaSyDTlQNcRKF55LjiN9ThExNSnEBoF8NoOr0")

    response = youtube.channels().list(
        part="snippet",
        id=channel_id
    ).execute()

    if not response["items"]:
        return ""

    return response["items"][0]["snippet"]["description"]



def get_video_meta(video_id):
    try: 
        youtube = build("youtube", "v3", developerKey="AIzaSyDTlQNcRKF55LjiN9ThExNSnEBoF8NoOr0")

        response = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        ).execute()
    except Exception:
        response = {}

    if not response["items"]:
        return {
            "video_title": "",
            "video_description": "",
            "channel_title": "",
            "channel_description": "",
            "views": 0
        }

    item = response["items"][0]
    snippet = item["snippet"]

    channel_id = snippet["channelId"]

    try:
        channel_desc = get_channel_description(channel_id) if channel_id else ""
    except Exception:
        channel_desc = ""


    statistics = item.get("statistics", {})
    views = statistics.get("viewCount", "0")
    
    def format_views(view_count):
        try:
            num = int(view_count)
            if num >= 1_000_000:
                return f"{num / 1_000_000:.1f}M"
            elif num >= 1_000:
                return f"{num / 1_000:.1f}K"
            else:
                return str(num)
        except (ValueError, TypeError):
            return str(view_count)

    return {
        "video_title": snippet["title"],
        "video_description": snippet["description"],
        "channel_title": snippet["channelTitle"],
        "channel_description": channel_desc,
        "views": format_views(views)
    }


st.title("📺 YouComment")


url = st.text_input(
    "Введите ссылку на YouTube видео",
    value="https://www.youtube.com/watch?v=RenH8uy57ZY",
    key="video_url",
    placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
    help="Ссылка на видео будет использоваться для генерации комментариев",
)

video_id = extract_video_id(url)

if video_id:
    st.video(f"https://www.youtube.com/watch?v={video_id}")
    

    if st.session_state.get("current_video_id") != video_id:
        st.session_state.meta = get_video_meta(video_id)
        st.session_state.current_video_id = video_id
        st.session_state.comments_data = []

    st.markdown('</div>', unsafe_allow_html=True) 

    st.markdown(f'<div class="video-title">{st.session_state.meta["video_title"]}</div>', unsafe_allow_html=True) 
    st.markdown(f'<div class="video-meta">{st.session_state.meta["views"]}</div>', unsafe_allow_html=True) 
    st.markdown(f'<div class="channel-box"> <div class="channel-avatar"></div> <div class="channel-name">{st.session_state.meta["channel_title"]}</div> </div>', unsafe_allow_html=True) 
    st.markdown(f'<div class="description-box"> {st.session_state.meta["video_description"]} </div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("➕ Сгенерировать комментарий"):
        new_comment = {
            "id": generate_id(),
            "author": f"User_{random.randint(1,1000)}",
            "likes": 0,
            "replies": []
        }

        st.session_state.comments_data.append(new_comment)

        generate_comment(new_comment["id"])

        st.rerun()

    st.markdown("---")

    render_comments(st.session_state.comments_data)

else:
    st.info("Введите ссылку")