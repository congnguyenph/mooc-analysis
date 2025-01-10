import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu đã làm sạch
data_cleaned = pd.read_csv('D:/HCMUE/ThS/Data Mining/data/cleaned_mooc_dataset.csv')

# Chuyển đổi các cột thời gian từ chuỗi sang định dạng datetime
data_cleaned['start_time_DI'] = pd.to_datetime(data_cleaned['start_time_DI'])
data_cleaned['last_event_DI'] = pd.to_datetime(data_cleaned['last_event_DI'])

# Tính toán thời gian học (giờ)
data_cleaned['time_spent'] = data_cleaned['last_event_DI'] - data_cleaned['start_time_DI']
data_cleaned['time_spent_hours'] = data_cleaned['time_spent'].dt.total_seconds() / 3600

# Tính các số liệu cho biểu đồ
data_cleaned['week'] = data_cleaned['start_time_DI'].dt.isocalendar().week
weekly_activity = data_cleaned.groupby('week')['userid_DI'].nunique()
course_popularity = data_cleaned['course_id'].value_counts()
completion_rate = data_cleaned.groupby('course_id')['incomplete_flag'].apply(lambda x: (1 - x.mean()) * 100)

# Tạo giao diện Streamlit
st.title("KHAI THÁC DỮ LIỆU MOOC: PHÂN TÍCH XU HƯỚNG HỌC TẬP VÀ HIỆU QUẢ GIÁO DỤC TRỰC TUYẾN")

# Widget lọc theo giới tính
gender_filter = st.selectbox(
    "Lọc theo giới tính",
    options=['Tất cả', 'Nam', 'Nữ']
)

# Widget lọc theo tuần học (slider)
week_range = st.slider(
    "Chọn tuần học",
    min_value=int(data_cleaned['week'].min()),
    max_value=int(data_cleaned['week'].max()),
    value=(int(data_cleaned['week'].min()), int(data_cleaned['week'].max())),
    step=1
)

# Lọc dữ liệu theo tuần
data_filtered = data_cleaned[(data_cleaned['week'] >= week_range[0]) & (data_cleaned['week'] <= week_range[1])]

# Lọc dữ liệu theo giới tính (nếu có)
if gender_filter == 'Nam':
    data_filtered = data_filtered[data_filtered['gender'] == 'm']
elif gender_filter == 'Nữ':
    data_filtered = data_filtered[data_filtered['gender'] == 'f']

# Cập nhật lại các số liệu sau khi lọc
weekly_activity = data_filtered.groupby('week')['userid_DI'].nunique()
course_popularity = data_filtered['course_id'].value_counts()
completion_rate = data_filtered.groupby('course_id')['incomplete_flag'].apply(lambda x: (1 - x.mean()) * 100)

# Widget cho phép người dùng tìm kiếm theo khóa học
search_course = st.text_input("Tìm kiếm khóa học theo mã hoặc tên:")
if search_course:
    data_filtered = data_filtered[data_filtered['course_id'].str.contains(search_course, case=False)]

# Widget cho phép người dùng chọn cột để hiển thị
columns_to_display = st.multiselect(
    "Chọn các cột dữ liệu để hiển thị",
    options=data_cleaned.columns,
    default=['userid_DI', 'course_id', 'start_time_DI', 'last_event_DI', 'time_spent_hours', 'grade']
)

st.subheader("Bộ dữ liệu đã lọc")
st.write(data_filtered[columns_to_display])

# Biểu đồ phân bố điểm số
st.header("Phân bố điểm số")
fig_score, ax_score = plt.subplots(figsize=(10, 6))
sns.histplot(data_filtered['grade'], kde=True, color='purple', ax=ax_score)
ax_score.set_title('Phân bố điểm số')
ax_score.set_xlabel('Điểm số')
ax_score.set_ylabel('Tần suất')
st.pyplot(fig_score)

### Biểu đồ 1: Tần suất truy cập theo tuần ###
st.header("Tần suất truy cập theo tuần")
fig1, ax1 = plt.subplots(figsize=(10, 6))
weekly_activity.plot(kind='line', marker='o', color='blue', ax=ax1)
ax1.set_title('Tần suất truy cập theo tuần')
ax1.set_xlabel('Tuần')
ax1.set_ylabel('Số người học')
ax1.grid(True)
st.pyplot(fig1)

### Biểu đồ 2: Top 5 khóa học phổ biến nhất ###
st.header("Top 5 khóa học phổ biến nhất")
fig2, ax2 = plt.subplots(figsize=(10, 6))
course_popularity.head(5).plot(kind='bar', color='orange', ax=ax2)
ax2.set_title('Top 5 khóa học phổ biến nhất')
ax2.set_xlabel('Mã khóa học')
ax2.set_ylabel('Số người tham gia')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
st.pyplot(fig2)

### Biểu đồ 3: Tỷ lệ hoàn thành khóa học ###
st.header("Tỷ lệ hoàn thành khóa học")
fig3, ax3 = plt.subplots(figsize=(10, 6))
completion_rate.plot(kind='barh', color='green', ax=ax3)
ax3.set_title('Tỷ lệ hoàn thành khóa học')
ax3.set_xlabel('Tỷ lệ hoàn thành (%)')
ax3.set_ylabel('Khóa học')
st.pyplot(fig3)

### Biểu đồ 4: Mối quan hệ giữa thời gian học và điểm số ###
st.header("Mối quan hệ giữa thời gian học và điểm số")
fig4 = sns.lmplot(
    x='time_spent_hours', 
    y='grade', 
    data=data_filtered,  # Dữ liệu đã lọc theo giới tính và tuần
    aspect=2, 
    height=6, 
    scatter_kws={'alpha': 0.6}
)
plt.title('Mối quan hệ giữa thời gian học và điểm số')
plt.xlabel('Thời gian học (giờ)')
plt.ylabel('Điểm số')
plt.grid(True)
st.pyplot(fig4)

# Widget tải xuống dữ liệu đã lọc
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(data_filtered)
st.download_button(
    label="Tải xuống dữ liệu đã lọc",
    data=csv,
    file_name='filtered_data.csv',
    mime='text/csv',
)
# Hiển thị thông tin cơ bản
st.sidebar.title("Thông tin cá nhân")
st.sidebar.write("Họ và tên: Phước Công Nguyên")
st.sidebar.write("Mã số Học viên: KHMT835018")
st.sidebar.write("Tiểu luận")
st.sidebar.write("Đề tài: KHAI THÁC DỮ LIỆU MOOC: PHÂN TÍCH XU HƯỚNG HỌC TẬP VÀ HIỆU QUẢ GIÁO DỤC TRỰC TUYẾN")
st.sidebar.write("Học phần: Khai thác dữ liệu và Ứng dụng")
st.sidebar.write("Ngành: Khoa học máy tính")
st.sidebar.write("Khoá 35 (2024 - 2026)")
st.sidebar.write("Giảng viên giảng day: TS. Huỳnh Lê Tấn Tài")
st.sidebar.write("Thành phố Hồ Chí Minh, ngày 08 tháng 01 năm 2025")