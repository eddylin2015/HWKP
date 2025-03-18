"""
pip install streamlit
pip install streamlit_chatbox
"""
import requests
import json
import streamlit as st
from streamlit_chatbox import *
import time
import simplejson as json


llm = FakeLLM()
chat_box = ChatBox(
    use_rich_markdown=True, # use streamlit-markdown
    user_theme="green", # see streamlit_markdown.st_markdown for all available themes
    assistant_theme="blue",
)
chat_box.use_chat_name("chat1") # add a chat conversatoin

def on_chat_change():
    chat_box.use_chat_name(st.session_state["chat_name"])
    chat_box.context_to_session() # restore widget values to st.session_state when chat name changed


with st.sidebar:
    st.subheader('start to chat using streamlit')
    chat_name = st.selectbox("Chat Session:", ["default", "chat1"], key="chat_name", on_change=on_chat_change)
    chat_box.use_chat_name(chat_name)
    streaming = st.checkbox('streaming', key="streaming")
    in_expander = st.checkbox('show messages in expander', key="in_expander")
    show_history = st.checkbox('show session state', key="show_history")
    chat_box.context_from_session(exclude=["chat_name"]) # save widget values to chat context

    st.divider()

    btns = st.container()

    file = st.file_uploader(
        "chat history json",
        type=["json"]
    )

    if st.button("Load Json") and file:
        data = json.load(file)
        chat_box.from_dict(data)


chat_box.init_session()
chat_box.output_messages()

def on_feedback(
    feedback,
    chat_history_id: str = "",
    history_index: int = -1,
):
    reason = feedback["text"]
    score_int = chat_box.set_feedback(feedback=feedback, history_index=history_index) # convert emoji to integer
    # do something
    st.session_state["need_rerun"] = True


feedback_kwargs = {
    "feedback_type": "thumbs",
    "optional_text_label": "wellcome to feedback",
}

if query := st.chat_input('input your question here'):
    chat_box.user_say(query)
    #generator = llm.chat_stream(query)
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "deepseek-r1:8b",
        "prompt": query
    }
    response = requests.post(url, json=data, stream=True)
    text=""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            x=json.loads(decoded_line)["response"]     
            text += x
            chat_box.update_msg(text, element_index=0, streaming=True)
                # update the element without focus
