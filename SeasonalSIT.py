import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 图表样式设置
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 10
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3
rcParams['axes.labelsize'] = 11
rcParams['legend.fontsize'] = 9
rcParams['legend.title_fontsize'] = 10

# 1. 读取长格式Excel（每行一个Year、Month）
excel_path = r"E:/2024seaice/2025/excel/piomas_monthly_thickness_smallshp_1979_202505.xlsx"
df_long = pd.read_excel(excel_path)


df_long.columns = df_long.columns.str.strip().str.lower()
df_wide = df_long.pivot(index='year', columns='month', values='thickness')

#
month_names = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.',
               'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.']
df_wide.columns = month_names


monthly_avg_values = {i: {} for i in range(1, 13)}  # 1~12月
for month_num, month_name in enumerate(month_names, 1):
    if month_name in df_wide.columns:
        for year, val in df_wide[month_name].items():
            monthly_avg_values[month_num][year] = val


seasons = {
    'Winter (DJF)': [12, 1, 2],
    'Spring (MAM)': [3, 4, 5],
    'Summer (JJA)': [6, 7, 8],
    'Autumn (SON)': [9, 10, 11]
}
season_colors = {
    'Winter (DJF)': plt.cm.Blues(np.linspace(0.4, 0.8, 3)),
    'Spring (MAM)': plt.cm.Greens(np.linspace(0.4, 0.8, 3)),
    'Summer (JJA)': plt.cm.Oranges(np.linspace(0.4, 0.8, 3)),
    'Autumn (SON)': plt.cm.Purples(np.linspace(0.4, 0.8, 3))
}


all_vals = []
for months in seasons.values():
    for m in months:
        all_vals.extend(list(monthly_avg_values[m].values()))
ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
y_margin = (ymax - ymin) * 0.1
ymin -= y_margin
ymax += y_margin

# 7. 创建绘图
fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
axs = axs.flatten()

for i, (ax, (season, months)) in enumerate(zip(axs, seasons.items())):
    colors = season_colors[season]

    # --- 1. 背景色块 (仅在 Spring/MAM 展示对比逻辑更专业) ---
    ax.axvspan(2002, 2020, color='#f47570', alpha=0.15, lw=0)
    ax.axvspan(2020, 2024, color='#b7e7fe', alpha=0.15, lw=0)

    # 坐标轴刻度向内
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    for idx, m in enumerate(months):

        year_list = np.array(sorted(monthly_avg_values[m].keys()))
        val_list = np.array([monthly_avg_values[m][y] for y in year_list])


        ax.plot(
            year_list, val_list,
            label=month_names[m - 1],
            color=colors[idx],
            marker='o',
            markersize=3.5,
            linewidth=1.2,
            alpha=0.9,
            zorder=5
        )

        # --- 3. 秋季 9 月特定年份标红 ---
        if m == 9:
            red_years = [1980, 1981, 1982, 1983, 1985, 1986, 1987, 1988, 1992, 1994, 1998, 2000, 2001, 2024]
            mask_red = np.isin(year_list, red_years)
            ax.scatter(
                year_list[mask_red], val_list[mask_red],
                color='red', s=35, edgecolors='white', linewidths=0.5,
                label='Sea-ice retention years', zorder=12
            )


    ax.set_title(season, fontsize=14, pad=10)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(1978, 2026)


    ax.set_xticks([1980, 1990, 2000, 2010, 2020, 2024])


    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=8, frameon=False)

    ax.set_title(season, fontsize=14, pad=10)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linestyle='--', alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels,

            loc='upper right' if season == 'Winter (DJF)' else 'best',
            fontsize=8,
            framealpha=0.8,
            frameon = False  # 去掉图例边框
        )

    if i in [0, 2]:
        ax.set_ylabel('Sea Ice Thickness (m)', fontsize=12)
    if i in [2, 3]:
        ax.set_xlabel('Year', fontsize=12)

    ax.set_xlim(1978, 2026)

    custom_ticks = [1980, 1990, 2000, 2010, 2020, 2024]
    ax.set_xticks(custom_ticks)

plt.tight_layout(pad=2.0)
plt.savefig(r"E:/2024seaice/fig79-24/from_long_excel_seasonal_lines_2025.png", dpi=300, bbox_inches='tight')
plt.show()
