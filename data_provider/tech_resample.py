# User: 廖宇
# Date Development：2024/1/22 14:16
import pandas as pd

# 读取CSV文件
df = pd.read_csv('./tech_cleaned.csv')

# 重新采样，并计算每个新时间间隔的平均值
resampled_df = df.groupby(df.index // 3).mean().reset_index(drop=True)

# 构建新的date列，从1开始
resampled_df['date'] = range(1, len(resampled_df) + 1)

# 输出结果到新的CSV文件
resampled_df.to_csv('resampled_file.csv', index=False)

# 提示保存成功
print("保存成功，新的CSV文件名为：resampled_file.csv")

