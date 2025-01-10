import pandas as pd
import os

# Đường dẫn đến tệp dữ liệu
file_path = 'D:/HCMUE/ThS/Data Mining/TL/data/big_student_clear_third_version.csv'

# CHUẨN BỊ DỮ LIỆU (LÀM SẠCH DỮ LIỆU)
# Bước 1: Đọc và kiểm tra dữ liệu
try:
    data = pd.read_csv(file_path)
    print("Dữ liệu đã được đọc thành công.")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp tại đường dẫn: {file_path}")
    exit()

# Hiển thị thông tin tổng quan về dữ liệu
print("\nThông tin tổng quan về dữ liệu:")
print(data.info())

# Kiểm tra số lượng giá trị thiếu trong từng cột
print("\nSố lượng giá trị thiếu trong từng cột:")
print(data.isnull().sum())

# Bước 2: Xử lý giá trị thiếu
# Loại bỏ các dòng dữ liệu bị thiếu ở cột "gender"
if 'gender' in data.columns:
    data_cleaned = data.dropna(subset=['gender'])
    print("\nĐã loại bỏ các dòng dữ liệu bị thiếu ở cột 'gender'.")
else:
    print("\nCột 'gender' không tồn tại trong dữ liệu.")
    exit()

# Điền giá trị trung bình cho cột "age" nếu cột này tồn tại
if 'age' in data_cleaned.columns:
    mean_age = data_cleaned['age'].mean()
    data_cleaned['age'] = data_cleaned['age'].fillna(mean_age)
    print(f"\nĐã điền giá trị trung bình ({mean_age:.2f}) vào các giá trị thiếu trong cột 'age'.")
else:
    print("\nCột 'age' không tồn tại trong dữ liệu.")

# Kiểm tra lại dữ liệu sau khi xử lý
print("\nThông tin dữ liệu sau khi xử lý giá trị thiếu:")
print(data_cleaned.info())

print("\nSố lượng giá trị thiếu còn lại trong từng cột:")
print(data_cleaned.isnull().sum())

# Bước 3: Chuyển đổi định dạng thời gian từ object sang datetime
datetime_columns = ['start_time_DI', 'last_event_DI']

# Kiểm tra xem các cột này có tồn tại trong dữ liệu không
for col in datetime_columns:
    if col in data_cleaned.columns:
        # Chuyển đổi các cột thời gian từ object (chuỗi) thành datetime
        # Sử dụng errors='coerce' để chuyển những giá trị không hợp lệ thành NaT
        data_cleaned[col] = pd.to_datetime(data_cleaned[col], errors='coerce')
        print(f"Đã chuyển đổi định dạng thời gian cho cột '{col}'.")
    else:
        print(f"Cột '{col}' không tồn tại trong dữ liệu.")

# Kiểm tra kiểu dữ liệu của các cột sau khi chuyển đổi
print("\nKiểu dữ liệu của các cột thời gian sau khi chuyển đổi:")
print(data_cleaned[['start_time_DI', 'last_event_DI']].dtypes)

# Kiểm tra số lượng giá trị NaT trong các cột thời gian (chỉ số bị thiếu sau khi chuyển đổi)
print("\nSố lượng giá trị NaT (Not a Time) trong các cột thời gian:")
print(data_cleaned[['start_time_DI', 'last_event_DI']].isna().sum())

# Thay thế giá trị NaT bằng một ngày mặc định
default_date = pd.Timestamp('2020-01-01')  # Ngày mặc định, có thể thay đổi theo yêu cầu
# Thay thế NaT (Not a Time) bằng giá trị ngày mặc định
data_cleaned.loc[:, 'start_time_DI'] = data_cleaned['start_time_DI'].fillna(default_date)
data_cleaned.loc[:, 'last_event_DI'] = data_cleaned['last_event_DI'].fillna(default_date)



print("\nĐã thay thế giá trị NaT bằng ngày mặc định.")

# Bước 4: Lưu dữ liệu sạch vào tệp mới
# Sử dụng thư viện os để đảm bảo đường dẫn được tạo đúng
output_dir = 'D:/HCMUE/ThS/Data Mining/data/'
os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
cleaned_file_path = os.path.join(output_dir, 'cleaned_mooc_dataset.csv')

# Cố gắng lưu dữ liệu sạch vào tệp mới
try:
    data_cleaned.to_csv(cleaned_file_path, index=False)
    print(f"\nDữ liệu sạch đã được lưu tại: {cleaned_file_path}")
except PermissionError:
    print(f"\nLỗi: Không có quyền ghi tệp tại: {cleaned_file_path}")
except FileNotFoundError:
    print(f"\nLỗi: Không tìm thấy thư mục: {output_dir}")
except Exception as e:
    print(f"\nLỗi khi lưu dữ liệu sạch: {e}")

# PHÂN TÍCH DỮ LIỆU
# B1: Khám phá dữ liệu ban đầu
print("\nThông tin tổng quan về dữ liệu:")
print(data_cleaned.info())
print("\nThống kê chung:")
print(data_cleaned.describe())

# B2: Phân tích xu hướng học tập
# Phân bố thời gian truy cập theo tuần
import pandas as pd

data_cleaned['week'] = data_cleaned['start_time_DI'].dt.isocalendar().week
weekly_activity = data_cleaned.groupby('week')['userid_DI'].nunique()
print(weekly_activity)

# Tần suất truy cập theo ngày
daily_activity = data_cleaned.groupby(data_cleaned['start_time_DI'].dt.date)['userid_DI'].nunique()
print(daily_activity)

# Khoá học phổ biến nhất
course_popularity = data_cleaned['course_id'].value_counts()
print(course_popularity.head(5))

# B3: Phân tích hiệu quả giáo dục
# Tỷ lệ hoàn thành khoá học# Tỷ lệ hoàn thành khóa học
completion_rate = data_cleaned.groupby('course_id')['incomplete_flag'].apply(lambda x: (1 - x.mean()) * 100)
print("Tỷ lệ hoàn thành từng khóa học (%):")
print(completion_rate)

# Tính hệ số tương quan giữa sự chênh lệch thời gian và điểm số
data_cleaned['time_spent'] = data_cleaned['last_event_DI'] - data_cleaned['start_time_DI']

correlation = data_cleaned['time_spent'].corr(data_cleaned['grade'])
print(f"Hệ số tương quan giữa sự chênh lệch thời gian và điểm số: {correlation:.2f}")



# Phân tích yếu tố gây bỏ khoá học
dropout_analysis = data_cleaned[data_cleaned['incomplete_flag'] == 1]
print("Phân tích đặc điểm của người bỏ khóa học:")
print(dropout_analysis.describe())

# TRỰC QUAN HOÁ DỮ LIỆU
# Biểu đồ tần suất truy cập theo tuần
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
weekly_activity.plot(kind='line', marker='o', color='blue')
plt.title('Tần suất truy cập theo tuần')
plt.xlabel('Tuần')
plt.ylabel('Số người học')
plt.grid(True)
plt.show()

# Biểu đồ Top 5 khóa học phổ biến

course_popularity.head(5).plot(kind='bar', color='orange')
plt.title('Top 5 khóa học phổ biến nhất')
plt.xlabel('Mã khóa học')
plt.ylabel('Số người tham gia')
plt.xticks(rotation=45)
plt.show()

# Biểu đồ tỷ lệ hoàn thành khóa học
completion_rate.plot(kind='barh', color='green')
plt.title('Tỷ lệ hoàn thành khóa học')
plt.xlabel('Tỷ lệ hoàn thành (%)')
plt.ylabel('Khóa học')
plt.show()



# Vẽ biểu đồ mối quan hệ giữa thời gian học và điểm số
import seaborn as sns
import matplotlib.pyplot as plt

sns.lmplot(x='time_spent_hours', y='grade', data=data_cleaned, aspect=2, height=6, scatter_kws={'alpha': 0.6})
plt.title('Mối quan hệ giữa thời gian học và điểm số')
plt.xlabel('Thời gian học (giờ)')
plt.ylabel('Điểm số')
plt.grid(True)
plt.show()

