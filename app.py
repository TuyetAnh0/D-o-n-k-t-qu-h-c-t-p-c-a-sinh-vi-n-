import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Dự Đoán Kết Quả Học Tập", page_icon="🎓", layout="centered")

if os.path.exists("style.css"):
    with open("style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("⚠️ Không tìm thấy file 'style.css' trong thư mục.")

st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title("🎓 Hệ Thống Dự Đoán Kết Quả Học Phần Sinh Viên")
st.write("Nhập thông số quá trình để kiểm tra nguy cơ trượt học phần thông qua mô hình AI.")
st.markdown("</div>", unsafe_allow_html=True)

st.header("📊 Thông số học tập thực tế")
col1, col2 = st.columns(2)

with col1:
    study_time = st.slider("Số giờ tự học trong tuần (giờ):", min_value=0.0, max_value=30.0, value=10.0, step=0.5)
    gpa = st.slider("Điểm trung bình tích lũy hiện tại (GPA hệ 4.0):", min_value=0.0, max_value=4.0, value=2.5, step=0.1)

with col2:
    absences = st.slider("Số buổi nghỉ học không phép (buổi):", min_value=0, max_value=30, value=3, step=1)

if st.button("🔍 Tiến hành phân tích", type="primary"):
    try:
        if not os.path.exists("best_model.pkl"):
            raise FileNotFoundError("Không tìm thấy tệp mô hình 'best_model.pkl'.")

        model = joblib.load("best_model.pkl")
        input_data = pd.DataFrame([[study_time, absences, gpa]], columns=['StudyTimeWeekly', 'Absences', 'GPA'])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.write("---")
        st.subheader("📋 Kết quả phân tích từ hệ thống AI:")

        if prediction == 1:
            st.markdown(f"""
            <div class='result-pass'>
                🎉 CHÚC MỪNG: Sinh viên có khả năng cao sẽ ĐẠT học phần này!<br>
                <span style='font-weight: normal; font-size: 15px;'>
                    Độ tự tin của mô hình (Xác suất đạt): {probability * 100:.2f}%
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-fail'>
                ⚠️ CẢNH BÁO: Sinh viên có nguy cơ TRƯỢT học phần này!<br>
                <span style='font-weight: normal; font-size: 15px;'>
                    Tỷ lệ rủi ro (Xác suất trượt): {(1 - probability) * 100:.2f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi hệ thống: {str(e)}")