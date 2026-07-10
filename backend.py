import streamlit as st
import re
import random 
import threading
import html 
import queue
import time

from models.model import load_youcomment_model



def generate_empty_comments(n=8, prob=0.6, max_depth=4, depth=0, start_id=0):
    comments = []
    path = ""

    cnt = 0

    for id in range(start_id, start_id + n):
        comment = {
            "author": f"User_{100}",
            "text": "",
            "likes": random.randint(3, 245),
            "id": id,
            "ready": False,
            "replies": [],
        }

        if depth < max_depth and random.random() < prob:
            comment["replies"] = generate_empty_comments(
                random.randint(1, 4), prob, max_depth, depth + 1, start_id + n
            )

        comments.append(comment)
        cnt += 1

    return comments, cnt



def render_comments_helper(comments, level=0):
    st.session_state.last_comment_queue = []

    for idx, comment in enumerate(comments):

        toggle_key = f"toggle_{level}_{idx}_{comment['author']}"

        vertical_line = '<div class="comment-vertical-line"></div>' if level > 0 else ""

        author = html.escape(str(comment.get("author", "")))
        text = html.escape(str(comment.get("text", ""))).replace("\n", "<br>")
        likes = comment.get("likes", 0)

        st.markdown(f"""
<div class="comment-container" style="margin-left: {level * 36}px;">
{vertical_line}

<div class="comment-avatar"></div>

<div class="comment-content">

<div class="comment-author">
{author}
<span class="comment-meta">• 2 ч назад</span>
</div>

<div class="comment-text">
{text}
</div>

<div class="comment-actions">
👍 {likes}
&nbsp;&nbsp;
<span>👎</span>
<span class="comment-reply">Ответить</span>
</div>

</div>
</div>
        """, unsafe_allow_html=True)

        if comment.get("replies"):
            num_replies = len(comment["replies"])

            show_replies = st.toggle(
                f"Показать ответы ({num_replies})",
                key=toggle_key,
                value=(level == 0)
            )

            if show_replies:
                render_comments_helper(comment["replies"], level + 1)


@st.fragment(run_every=0.5)
def generate_comments():
    st.session_state.comments = generate_empty_comments() # TODO

    comments = st.session_state.comments 
    
    while len(comments) > 0:
        for comment in comments:



    render_comments_helper(st.session_state.comments, 0)


def background_generator(task_queue: queue.Queue):
    """Работает в отдельном потоке. Получает задачи и отправляет результаты через очередь."""
    while True:
        try:
            task = task_queue.get()
            
            if task is None:
                break
                
            comments_data = task.get("comments_data")
            
            results = {}
            total = len(comments_data)
            
            for i, comment_id in enumerate(comments_data):
                # Ваш код генерации
                result = st.session_state.model.generate(comment_id, ...)  
                
                results[comment_id] = result
                
                # Отправляем прогресс в UI через очередь
                progress_msg = {
                    "type": "progress",
                    "current": i + 1,
                    "total": total,
                    "comment_id": comment_id,
                    "result": result
                }
                st.session_state.generation_queue.put(progress_msg)
                
                time.sleep(0.01)  # имитация работы, уберите в реальном коде
            
            # Отправляем финальное сообщение
            st.session_state.generation_queue.put({
                "type": "finished",
                "results": results
            })
            
        except queue.Empty:
            continue
        except Exception as e:
            st.session_state.generation_queue.put({"type": "error", "error": str(e)})