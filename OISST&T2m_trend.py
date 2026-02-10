import os
import glob
from shapely.geometry import box, mapping  #要放在前面
import numpy as np
import rasterio
from rasterio.mask import mask

import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import linregress
from datetime import datetime

# ========= 参数 =========
sst_folder = r"E:\2024seaice\2025\oisst\daily\dailytiff"
t2m_folder = r"E:\2024seaice\2025\T2m_Monthly\t2m"
plt.rcParams['font.family'] = 'Arial'
months_78910 = [7, 8,9]

# SST数据区域定义 (0-360度)
sst_regions = {
    'Arctic': box(1, 65, 359, 90),
    'Chukchi': box(173, 66, 198, 73),  # 原 -180 到 -160 → 对应 180 到 200
    'Beaufort': box(198, 65, 220, 75),  # 原 -160 到 -130 → 对应 200 到 230
    'Barents': box(27, 65, 60, 80)  # Barents 区域本身就是正经度，无需更改
}

# T2M数据区域定义 (-180-180度)
t2m_regions = {
    'Arctic': box(-179, 65, 179, 90),
    'Chukchi': box(-180, 66, -160, 73),  # -180 到 -160
    'Beaufort': box(-160, 65, -130, 75),  # -160 到 -130
    'Barents': box(27, 65, 60, 80)  # Barents 区域不变
}

# ========= 读取SST文件并筛选 7–9 月 =========
file_list = sorted(glob.glob(os.path.join(sst_folder, "*.tif")))
quarter_files = defaultdict(list)  # quarter_files[year] = [ (filename, date), ...]

for f in file_list:
    try:
        date_str = os.path.basename(f)[6:14]
        date = datetime.strptime(date_str, "%Y%m%d")
        if date.month in months_78910:
            quarter_files[date.year].append((f, date))
    except:
        continue


t2m_files = sorted(glob.glob(os.path.join(t2m_folder, "*_t2m.tif")))
t2m_quarter_files = defaultdict(list)  # t2m_quarter_files[year] = [ (filename, date), ...]

for f in t2m_files:
    try:
        date_str = os.path.basename(f)[:8]
        date = datetime.strptime(date_str, "%Y%m%d")
        if date.month in months_78910:
            t2m_quarter_files[date.year].append((f, date))
    except:
        continue

quarter_data = defaultdict(lambda: defaultdict(list))  # quarter_data[region][year] = [temps]

for year, files in tqdm(quarter_files.items()):
    for f, date in files:
        with rasterio.open(f) as src:
            for region_name, region_box in sst_regions.items():
                try:
                    out_image, _ = mask(src, [mapping(region_box)], crop=True)
                    data = out_image[0]
                    data[data <= -1.64] = np.nan  # 海冰以下为无效
                    mean_temp = np.nanmean(data)
                    if not np.isnan(mean_temp):
                        quarter_data[region_name][year].append(mean_temp)
                except:
                    continue


t2m_quarter_data = defaultdict(lambda: defaultdict(list))  # t2m_quarter_data[region][year] = [temps]

for year, files in tqdm(t2m_quarter_files.items()):
    for f, date in files:
        with rasterio.open(f) as src:
            for region_name, region_box in t2m_regions.items():
                try:
                    out_image, _ = mask(src, [mapping(region_box)], crop=True)
                    data = out_image[0]
                    data[data < -100] = np.nan  # 无效值处理
                    mean_temp = np.nanmean(data)
                    if not np.isnan(mean_temp):
                        t2m_quarter_data[region_name][year].append(mean_temp)
                except:
                    continue


fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
axs = axs.flatten()
fig.suptitle(f"SST & T2m Anomaly Time Series (Jul.–Sept., 1982–2025)", fontsize=13)

for idx, region_name in enumerate(sst_regions):
    # 处理SST数据
    years = sorted(quarter_data[region_name].keys())
    mean_vals, std_vals = [], []

    for y in years:
        temps = quarter_data[region_name][y]
        temps = [t for t in temps if not np.isnan(t)]
        if temps:
            mean_vals.append(np.mean(temps))
            std_vals.append(np.std(temps))
        else:
            mean_vals.append(np.nan)
            std_vals.append(np.nan)

    # 计算SST anomaly和±1σ范围
    baseline_years = [y for y in years if 2002 <= y <= 2019]
    baseline_vals = [mean_vals[years.index(y)] for y in baseline_years if not np.isnan(mean_vals[years.index(y)])]
    baseline = np.nanmean(baseline_vals)

    anomaly_vals = [v - baseline if not np.isnan(v) else np.nan for v in mean_vals]
    anomaly_upper = [v + s - baseline if not np.isnan(v) and not np.isnan(s) else np.nan for v, s in
                     zip(mean_vals, std_vals)]
    anomaly_lower = [v - s - baseline if not np.isnan(v) and not np.isnan(s) else np.nan for v, s in
                     zip(mean_vals, std_vals)]

    ax = axs[idx]
    ax.plot(years, anomaly_vals, color='blue', label='SST Anomaly')
    ax.fill_between(years, anomaly_lower, anomaly_upper, color='lightblue', alpha=0.5)

    # 处理T2M数据
    t2m_years = sorted(t2m_quarter_data[region_name].keys())
    t2m_mean_vals, t2m_std_vals = [], []

    for y in t2m_years:
        temps = t2m_quarter_data[region_name][y]
        temps = [t for t in temps if not np.isnan(t)]
        if temps:
            t2m_mean_vals.append(np.mean(temps))
            t2m_std_vals.append(np.std(temps))
        else:
            t2m_mean_vals.append(np.nan)
            t2m_std_vals.append(np.nan)

    # 计算T2M anomaly
    t2m_baseline_vals = [t2m_mean_vals[t2m_years.index(y)] for y in baseline_years
                         if y in t2m_years and not np.isnan(t2m_mean_vals[t2m_years.index(y)])]
    t2m_baseline = np.nanmean(t2m_baseline_vals)

    t2m_anomaly_vals = [v - t2m_baseline if not np.isnan(v) else np.nan for v in t2m_mean_vals]
    t2m_anomaly_upper = [v + s - t2m_baseline if not np.isnan(v) and not np.isnan(s) else np.nan
                         for v, s in zip(t2m_mean_vals, t2m_std_vals)]
    t2m_anomaly_lower = [v - s - t2m_baseline if not np.isnan(v) and not np.isnan(s) else np.nan
                         for v, s in zip(t2m_mean_vals, t2m_std_vals)]

    # 创建右侧坐标轴并绘制T2M
    ax2 = ax.twinx()
    ax2.plot(t2m_years, t2m_anomaly_vals, color='darkviolet', label='T2m Anomaly', alpha=0.8)
    #ax2.fill_between(t2m_years, t2m_anomaly_lower, t2m_anomaly_upper, color='violet', alpha=0.2)
    # 改为：
    if idx in [1, 3]:  # 只对右下两个子图设置右侧标签
        ax2.set_ylabel('T2m Anomaly (°C)', color='darkviolet')
    else:  # 左侧两个子图不显示右侧标签
        ax2.set_ylabel('')
    ax2.tick_params(axis='y', colors='darkviolet')

    # 整体趋势线 (SST)
    valid_years = np.array([y for y, v in zip(years, anomaly_vals) if not np.isnan(v)])
    valid_vals = np.array([v for v in anomaly_vals if not np.isnan(v)])
    if len(valid_years) >= 5:
        slope, intercept, r_val, p_val, std_err = linregress(valid_years, valid_vals)
        trend_line = intercept + slope * valid_years
        ax.plot(valid_years, trend_line, color='dimgray', linestyle='--', linewidth=2, label='SST Trend (1982–2025)')

        trend_label = f"{'*' if p_val < 0.05 else ''}SST Trend: {slope:.3f}±{std_err:.3f} °C/yr"
        ax.text(0.90, 0.05, trend_label, transform=ax.transAxes, fontsize=11, ha='right', va='bottom', color='dimgray')

    # 整体趋势线 (T2M)
    t2m_valid_years = np.array([y for y, v in zip(t2m_years, t2m_anomaly_vals) if not np.isnan(v)])
    t2m_valid_vals = np.array([v for v in t2m_anomaly_vals if not np.isnan(v)])
    if len(t2m_valid_years) >= 5:
        t2m_slope, t2m_intercept, t2m_r_val, t2m_p_val, t2m_std_err = linregress(t2m_valid_years, t2m_valid_vals)
        t2m_trend_line = t2m_intercept + t2m_slope * t2m_valid_years
        # ax2.plot(t2m_valid_years, t2m_trend_line, color='purple', linestyle=':', linewidth=2,
        #          label='T2M Trend (1982–2024)')

        # t2m_trend_label = f"{'*' if t2m_p_val < 0.05 else ''}T2M Trend: {t2m_slope:.3f}±{t2m_std_err:.3f} °C/yr"
        # ax2.text(0.45, 0.15, t2m_trend_label, transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
        #          color='purple')

    # Chukchi 区域 2019–2024 红线
    if region_name == 'Chukchi':
        years_2019 = [y for y in years if 2020 <= y <= 2024 and not np.isnan(anomaly_vals[years.index(y)])]
        vals_2019 = [anomaly_vals[years.index(y)] for y in years_2019]
        if len(years_2019) >= 3:
            slope19, intercept19, r_val19, p_val19, std_err19 = linregress(years_2019, vals_2019)
            red_line = intercept19 + slope19 * np.array(years_2019)
            ax.plot(years_2019, red_line, color='Crimson', linestyle='--', linewidth=2, label='SST Trend (2020–2024)')
            trend19_label = f"{'*' if p_val19 < 0.05 else ''}2020–2024:\n {slope19:.3f}±{std_err19:.3f} °C/yr"
            ax.text(0.98, 0.95, trend19_label, transform=ax.transAxes, fontsize=11, color='red', ha='right',
                    va='top')

    # 设置标题、标签、网格
    ax.set_title(region_name)

    if idx in [0, 1]:  # 上面两图去掉 x 轴标签但保留刻度
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Year", fontsize=13)

    if idx in [1, 3]:  # 右边两图去掉 y 轴标签但保留刻度
        ax.set_ylabel("")
    else:
        ax.set_ylabel("SST Anomaly (°C)", fontsize=13)

    ax.grid(True)
    # 添加2025年标签（保持原有所有刻度）
    if 2025 in years:
        # 获取当前所有数据年份
        all_years = sorted(list(set(years + t2m_years)))

        # 选择要显示的刻度年份（每10年一个，加上2025，排除1979）
        major_ticks = [y for y in all_years
                      if (y % 10 == 0 or y == 2025 or y == all_years[-1])
                      and y != 1979]

        # 确保刻度排序正确
        major_ticks = sorted(list(set(major_ticks)))

        # 设置主要刻度
        ax.set_xticks(major_ticks)

        # 设置刻度标签（保持原有样式）
        ax.set_xticklabels(major_ticks, rotation=0, ha='center')
    # 合并图例
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left', frameon=False)


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"SST_T2M_Anomaly_Timeseries_78910.png", dpi=300)
plt.show()