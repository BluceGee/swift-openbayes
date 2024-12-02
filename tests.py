import pandas as pd
import json

# 读取CSV文件
input_file = '/openbayes/home/swift/datas/xingce/output.csv'  # 替换为你的CSV文件名
output_file = '/openbayes/home/swift/datas/xingce/output.json'  # 输出的JSON文件名

# 读取CSV数据
df = pd.read_csv(input_file)

# 检查列名
print(df.columns)

# 创建新的DataFrame，仅保留“题目”和“答案解析”列，并重命名列
result_df = df.rename(columns={'题目': 'instruction', '答案解析': 'output'})

# 转换为字典列表
result_dict = result_df.to_dict(orient='records')

# 保存为JSON文件
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(result_dict, json_file, ensure_ascii=False, indent=4)

print(f"转换完成，输出文件为：{output_file}")