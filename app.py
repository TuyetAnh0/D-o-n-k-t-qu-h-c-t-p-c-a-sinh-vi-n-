import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="AI Prediction System", layout="centered")

if os.path.exists("style.css"):
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🤖 AI PREDICTION SYSTEM")

st.markdown("""
<div class='intro-box'>
    <strong>HỆ THỐNG DỰ BÁO HỌC THUẬT</strong><br>
    Sử dụng thuật toán học máy phân tích dữ liệu sinh viên thời gian thực.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    study_time = st.slider("Số giờ tự học (giờ):", 0.0, 30.0, 10.0)
    gpa = st.slider("Điểm GPA hiện tại:", 0.0, 4.0, 2.5, 0.1)
with col2:
    absences = st.slider("Số buổi nghỉ học:", 0, 30, 2)

if st.button("🚀 KHỞI CHẠY PHÂN TÍCH"):
    try:
        model = joblib.load("best_model.pkl")
        data = pd.DataFrame([[study_time, absences, gpa]], columns=['StudyTimeWeekly', 'Absences', 'GPA'])
        
        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1] * 100
        
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        if pred == 1:
            st.markdown(f"<div class='result-pass'>✅ KẾT QUẢ: ĐẠT<br>Độ tin cậy: {prob:.2f}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-fail'>❌ KẾT QUẢ: CẢNH BÁO TRƯỢT<br>Tỷ lệ rủi ro: {100-prob:.2f}%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    except:
        st.error("Chưa tìm thấy tệp mô hình!")
