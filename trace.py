import pickle

# 获取数据
infile = open("AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt", "r")

# 跳过第一行（表头）
# app,func,end_timestamp,duration
_ = infile.readline()

# 读取一行数据
line = infile.readline()

# 初始化一个字典，用于存储不同函数调用的跟踪数据
traces = {}

# 初始化计数器，用于统计特定条件下的函数调用数量
count = 0

# 循环读取文件中的每一行数据
while line:
    # 解析当前行的数据，分割成多个部分
    # 例如：7b2c43a2bc30f6bb438074df88b603d2cb982d3e7961de05270735055950a568,e3cdb48830f66eb8689cc0223514569a69812b77e6611e3d59814fac0747bd2f,0.07949090003967285,0.078
    line_split = line.split(",")

    # 获取函数名称和结束时间
    func_name = line_split[1]
    end_time = float(line_split[2])

    # 如果结束时间小于一天的秒数（3600秒 * 24），则增加计数器
    if end_time < 3600 * 24:
        count += 1

        # 如果函数名称在字典中已存在，则将结束时间添加到对应列表中
        if func_name in traces.keys():
            traces[func_name].append(end_time)
        else:
            # 如果函数名称在字典中不存在，则创建新的列表并添加结束时间
            traces[func_name] = [end_time]

    # 读取下一行数据，准备继续循环
    line = infile.readline()
with open("traces.pkl", "wb") as f:
    pickle.dump(traces, f)
"""
import matplotlib.pyplot as plt
parameters = {
    "figure.figsize": [12, 8],
    "axes.labelsize": 36,
    "xtick.labelsize": 36,
    "ytick.labelsize": 36,
    "lines.markersize": 32,
    "lines.linewidth": 5,
    "font.family": "Arial",
    "font.size": 36,
}
plt.rcParams.update(parameters)
func1 = "cc5bb2108cc7daf53f9728ad21f661a8ef9c8b36284bacfcb712e2be87eef842"
func2 = "8e5f533dbf1092f56ac6c7542ef3bdec4661bd442c9b5e7537fabc7b8c03f5a8"
func3 = "762835950e81a11cd04cedcb05275dc111c651625d575077fce49f82170e0986"


# 指定要绘制图表的函数名称
func_name = func3

# 初始化一个字典，用于存储每分钟的请求数量
request_per_minutes = {}
# 获取特定函数调用的跟踪数据
trace = traces[func_name]

# 遍历函数调用的结束时间列表，统计每分钟的请求数量
for timestamp in trace:
    minute = int(timestamp) // 60

    if minute in request_per_minutes.keys():
        request_per_minutes[minute] += 1
    else:
        request_per_minutes[minute] = 1

# 创建 x 轴和 y 轴数据
x_axis = sorted(list(request_per_minutes.keys()))
y_axis = [request_per_minutes[x] for x in x_axis]

# 绘制图表
plt.xlabel("8e5f53")
plt.ylabel("Request/Minute")

# 筛选 x 轴和 y 轴的数据，仅保留特定范围的数据
x_filtered = [x for x in x_axis if 950 <= x <= 1200]
y_filtered = [request_per_minutes[x] for x in x_filtered]

# 绘制线图，并显示图表
plt.plot(x_filtered, y_filtered)
plt.show()"""
