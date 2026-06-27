# app2_qa.py
import os
import streamlit as st
import teacher_qa
from config import DB_PATH

st.set_page_config(page_title="Teacher Assistant QA", page_icon="🎓")
st.title("🎓 მასწავლებლის ასისტენტი - კითხვა-პასუხი")

question = st.text_area("შეიყვანეთ შეკითხვა:", height=120)

if st.button("პასუხის მიღება"):
    if question.strip() and os.path.exists(DB_PATH):
        with st.spinner("პასუხი მუშავდება..."):
            answer = teacher_qa.answer_question(question)
            st.info(answer)
    else:
        st.error("ჩაწერეთ შეკითხვა ან შექმენით ბაზა!")
