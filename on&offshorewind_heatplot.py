import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

plt.rcParams['font.family'] = 'Arial'

# ====== 1. 数据处理 (保留你的逻辑) ======
input_path = r"E:/2024seaice/2025/excel/daily_proj_wind_single.csv"
df = pd.read_csv(input_path, parse_dates=['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

monthly_avg = df.groupby(['Year', 'Month']).agg(
    Onshore_Avg=('Projected_Wind', lambda x: x[x > 0].mean()),
    Offshore_Avg=('Projected_Wind', lambda x: x[x < 0].mean())
).reset_index()

month_names = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May.', 'Jun.',
              'Jul.', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.']
monthly_avg['Month_Name'] = monthly_avg['Month'].apply(lambda x: month_names[x-1])

onshore_data = monthly_avg.pivot(index='Month_Name', columns='Year', values='Onshore_Avg').loc[month_names]
offshore_data = monthly_avg.pivot(index='Month_Name', columns='Year', values='Offshore_Avg').abs().loc[month_names]
combined_data = onshore_data.fillna(0) - offshore_data.fillna(0)


for col in combined_data.columns:
    if col == 2025:
        combined_data.loc[combined_data.index[9:], col] = np.nan


height_weights = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
plot_data = combined_data.iloc[::-1].values
plot_weights = height_weights[::-1]
y_boundaries = np.concatenate([[0], np.cumsum(plot_weights)])
y_centers = y_boundaries[:-1] + np.array(plot_weights) / 2
x_boundaries = np.arange(combined_data.shape[1] + 1)
x_centers = np.arange(combined_data.shape[1]) + 0.5


neg_boundaries = np.linspace(-2.1, 0, 4)  # 产生 [-2.1, -1.4, -0.7, 0] (3个区间)
pos_boundaries = np.linspace(0, 3.3, 6)   # 产生 [0, 0.66, 1.32, 1.98, 2.64, 3.3] (5个区间)
# 合并边界并去重
custom_boundaries = np.unique(np.concatenate([neg_boundaries, pos_boundaries]))


selected_colors = [
    '#4393c3', '#92c5de',  '#d1e5f0',  # 3种蓝色 (深到浅)
    '#fddbc7', '#d6604d', '#b2182b', '#8b0000', '#4d0000'  # 5种红色 (浅到深)
]

cmap = LinearSegmentedColormap.from_list('custom_diverging', selected_colors)
norm = BoundaryNorm(custom_boundaries, cmap.N)

# ====== 4. 绘图 ======
fig, ax = plt.subplots(figsize=(15, 5.16))

# 隐藏边框
for spine in ax.spines.values():
    spine.set_visible(False)


mesh = ax.pcolormesh(x_boundaries, y_boundaries, plot_data, cmap=cmap, norm=norm, edgecolors='white', linewidth=0.5)


mask = np.isnan(plot_data)
for y in range(plot_data.shape[0]):
    for x in range(plot_data.shape[1]):
        if mask[y, x]:
            ax.text(x_centers[x], y_centers[y], '/', ha='center', va='center', color='black', fontsize=19, rotation=45)


ax.set_yticks(y_centers)
ax.set_yticklabels(month_names[::-1])


target_months = {'May.', 'Jun.', 'Jul.', 'Aug.'}
for label in ax.get_yticklabels():
    if label.get_text() in target_months:
        label.set_fontsize(14)
        label.set_fontweight('bold')
    else:
        label.set_fontsize(12)

ax.set_xticks(x_centers)
year_labels = combined_data.columns
ax.set_xticklabels(year_labels, rotation=60, fontsize=16)


highlight_years = {1980, 1981, 1982, 1983, 1985, 1986, 1987, 1988, 1992, 1994, 1998, 2000, 2001, 2024}
for i, label in enumerate(ax.get_xticklabels()):
    if int(year_labels[i]) in highlight_years:
        label.set_color('red')
        label.set_fontweight('bold')


cbar = fig.colorbar(mesh, ax=ax, ticks=custom_boundaries, pad=0.02)
cbar.outline.set_visible(False) # 去掉色条边框线
cbar.set_ticklabels([f"{x:.1f}" for x in custom_boundaries], fontsize=14)

plt.xlabel('Year', fontsize=16)
plt.ylabel('Month', fontsize=16)
plt.tight_layout()
plt.show()