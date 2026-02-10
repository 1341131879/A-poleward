import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
from datetime import datetime
from scipy.interpolate import interp1d

# 设置全局样式
plt.style.use('seaborn')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.figsize'] = (15, 10)


def fill_missing_data(series, max_gap=0):
    """
    填充缺失数据，优先使用前向填充，然后后向填充
    参数:
        series: 包含NaN的Pandas Series
        max_gap: 最大允许填充的连续缺失天数
    返回:
        填充后的Series
    """

    result = series.copy()


    original_missing = result.isna().sum()
    print(f"  - 原始缺失值: {original_missing}")

    if original_missing == 0:
        return result


    result_ffill = result.ffill(limit=max_gap)
    ffill_filled = (result_ffill.notna() & result.isna()).sum()
    print(f"  - 前向填充了 {ffill_filled} 个值")


    result_bfill = result_ffill.bfill(limit=max_gap)
    bfill_filled = (result_bfill.notna() & result_ffill.isna()).sum()
    print(f"  - 后向填充了 {bfill_filled} 个值")


    if result_bfill.isna().any():
        result_interp = interpolate_with_gaps(result_bfill, max_gap=5)
        interp_filled = (result_interp.notna() & result_bfill.isna()).sum()
        print(f"  - 线性插值了 {interp_filled} 个值")
        final_result = result_interp
    else:
        final_result = result_bfill

    final_missing = final_result.isna().sum()
    print(f"  - 最终缺失值: {final_missing}")
    print(f"  - 总填充值: {original_missing - final_missing}")

    return final_result


def interpolate_with_gaps(series, max_gap=5):
    """
    带间隙限制的线性插值（用于处理较大间隙）
    参数:
        series: 包含NaN的Pandas Series
        max_gap: 最大允许插值的连续缺失天数
    返回:
        插值后的Series
    """
    # 标记有效数据点
    valid = series.notna()

    if not valid.any() or valid.all():
        return series


    result = series.copy()


    valid_idx = np.where(valid)[0]

    if len(valid_idx) < 2:
        return result


    for i in range(len(valid_idx) - 1):
        start_idx = valid_idx[i]
        end_idx = valid_idx[i + 1]
        gap_size = end_idx - start_idx - 1


        if 0 < gap_size <= max_gap:
            start_val = series.iloc[start_idx]
            end_val = series.iloc[end_idx]


            for j in range(1, gap_size + 1):
                weight = j / (gap_size + 1)
                interp_val = start_val + (end_val - start_val) * weight
                result.iloc[start_idx + j] = interp_val

    return result


def calculate_freeze_melt_dates(df, threshold=7, window_size=11):
    """
    计算每年的冻结日期和融化日期
    """
    years = sorted(df['year'].unique())
    results = []

    for year in years:
        year_data = df[df['year'] == year].sort_values('day_of_year').copy()

        if year_data.empty:
            continue

        # 计算滑动平均值
        year_data['rolling_avg'] = year_data['concentration'].rolling(
            window=window_size, center=True, min_periods=1).mean()

        # 找出首次低于阈值（融化开始）- 使用单日值
        first_below = year_data[year_data['concentration'] < threshold].first_valid_index()

        if first_below is not None:
            melt_date = year_data.loc[first_below, 'date']
            melt_day = year_data.loc[first_below, 'day_of_year']

            # 在融化开始后找冻结开始
            after_below = year_data.loc[first_below:].copy()
            first_above = after_below[after_below['rolling_avg'] >= threshold].first_valid_index()

            if first_above is not None:
                freeze_date = year_data.loc[first_above, 'date']
                freeze_day = year_data.loc[first_above, 'day_of_year']
                open_water_days = freeze_day - melt_day

                results.append({
                    'year': year,
                    'melt_date': melt_date,
                    'melt_day': melt_day,
                    'freeze_date': freeze_date,
                    'freeze_day': freeze_day,
                    'open_water_days': open_water_days
                })
            else:
                # 如果没有找到冻结日期
                results.append({
                    'year': year,
                    'melt_date': melt_date,
                    'melt_day': melt_day,
                    'freeze_date': None,
                    'freeze_day': None,
                    'open_water_days': None
                })

    return pd.DataFrame(results)


def plot_from_csv(csv_path, start_month=4, end_month=12):
    """
    从CSV文件直接绘制海冰密集度热力图
    """
    # 读取CSV数据
    df = pd.read_csv(csv_path, parse_dates=['date'])

    # 打印原始数据信息
    print("=" * 60)
    print("数据预处理信息")
    print("=" * 60)
    print(f"原始数据形状: {df.shape}")
    print(f"数据年份范围: {df['year'].min()} 到 {df['year'].max()}")

    # 检查1982年的数据
    if 1982 in df['year'].unique():
        year_1982 = df[df['year'] == 1982]
        print(f"\n1982年原始数据:")
        print(f"  - 总行数: {len(year_1982)}")
        print(f"  - 有效数据点: {year_1982['concentration'].notna().sum()}")
        print(f"  - 缺失数据点: {year_1982['concentration'].isna().sum()}")

        # 检查日期连续性
        year_1982_sorted = year_1982.sort_values('date')
        date_diff = year_1982_sorted['date'].diff().dt.days
        gaps = date_diff[date_diff > 1]
        if not gaps.empty:
            print(f"  - 发现日期间隙: {gaps.tolist()}")
            print(f"  - 最大间隙: {gaps.max()} 天")

    # 筛选月份范围并从1979年开始
    df = df[(df['month'] >= start_month) &
            (df['month'] <= end_month) &
            (df['year'] >= 1979)]

    if df.empty:
        print(f"No data found for months {start_month}-{end_month}")
        return

    # 按年份分组并填充数据
    dfs = []
    print(f"\n开始处理各年份数据...")

    for year, group in df.groupby('year'):
        print(f"\n处理 {year} 年:")

        # 创建完整的日期范围
        start_date = datetime(year, start_month, 1)
        if end_month == 12:
            end_date = datetime(year, 12, 31)
        else:
            end_date = datetime(year, end_month + 1, 1) - pd.Timedelta(days=1)

        date_range = pd.date_range(start_date, end_date, freq='D')

        # 创建包含所有日期的DataFrame
        full_df = pd.DataFrame({'date': date_range})
        full_df['year'] = year
        full_df['month'] = full_df['date'].dt.month
        full_df['day'] = full_df['date'].dt.day
        full_df['day_of_year'] = full_df['date'].dt.dayofyear

        # 检查原始数据
        print(f"  - 原始数据天数: {len(group)}")
        print(f"  - 原始有效浓度值: {group['concentration'].notna().sum()}")

        # 合并数据
        merged = pd.merge(full_df, group[['date', 'concentration']],
                          on='date', how='left')

        # 保存原始浓度值
        merged['concentration_original'] = merged['concentration']

        # 填充缺失数据
        merged['concentration'] = fill_missing_data(merged['concentration'], max_gap=1)


        dfs.append(merged)


    df = pd.concat(dfs, ignore_index=True)



    freeze_melt_df = calculate_freeze_melt_dates(df)


    years = sorted(df['year'].unique())
    start_day = datetime(2020, start_month, 1).timetuple().tm_yday
    end_day = 365 if end_month == 12 else datetime(2020, end_month + 1, 1).timetuple().tm_yday - 1
    days_in_range = end_day - start_day + 1

    # 创建二维数组
    heatmap_data = np.full((days_in_range, len(years)), np.nan)


    for i, year in enumerate(years):
        year_data = df[df['year'] == year]
        for j in range(days_in_range):
            day = start_day + j
            day_data = year_data[year_data['day_of_year'] == day]
            if not day_data.empty:
                concentration = day_data['concentration'].values[0]
                if concentration < 1:
                    heatmap_data[j, i] = -1
                else:
                    heatmap_data[j, i] = concentration


    bounds = [-1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors = ['white']
    colors.extend(plt.cm.Blues(np.linspace(0.2, 1, 10)))


    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)


    fig, ax = plt.subplots(figsize=(10, 5))

    cmap.set_bad('lightgray')


    im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap,
                   norm=norm, interpolation='nearest',
                   extent=[-0.5, len(years) - 0.5, days_in_range + 0.5, 0.5])


    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


    circle_points = []
    triangle_points = []


    threshold = 15
    window_size = 7

    for i, year in enumerate(years):
        year_data = df[df['year'] == year].sort_values('day_of_year')

        # 找出首次<15%的日期
        first_below = year_data[year_data['concentration'] < threshold].first_valid_index()
        if first_below is not None:
            day_of_year = year_data.loc[first_below, 'day_of_year']
            y_pos = day_of_year - start_day + 1
            circle_points.append((i, y_pos))
            ax.plot(i, y_pos, 'ro', markersize=4, zorder=5)

        # 找出冻结开始日期
        if first_below is not None:
            after_below = year_data.loc[first_below:].copy()
            after_below['rolling_avg'] = after_below['concentration'].rolling(
                window=window_size, center=True, min_periods=1).mean()

            first_above = after_below[after_below['rolling_avg'] >= threshold].first_valid_index()

            if first_above is not None:
                day_of_year = year_data.loc[first_above, 'day_of_year']
                y_pos = day_of_year - start_day + 1
                triangle_points.append((i, y_pos))
                ax.plot(i, y_pos, 'gray', marker='o', markersize=4, zorder=5)

    # 绘制曲线
    if len(circle_points) > 1:
        x_vals, y_vals = zip(*circle_points)
        ax.plot(x_vals, y_vals, 'r-', linewidth=2, alpha=0.3, zorder=4)

    if len(triangle_points) > 1:
        x_vals, y_vals = zip(*triangle_points)
        ax.plot(x_vals, y_vals, 'gray', linewidth=2, alpha=0.5, zorder=4)

    # 设置坐标轴
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Day of Year', fontsize=12)

    # 设置月份刻度
    month_pos = []
    month_labels = []
    for month in range(start_month, end_month + 1):
        day = datetime(2020, month, 15).timetuple().tm_yday - start_day + 1
        month_pos.append(day)
        month_labels.append(datetime(1900, month, 1).strftime('%b'))

        if month == 9:
            sep1_day = datetime(2020, 9, 1).timetuple().tm_yday - start_day + 1
            ax.axhline(y=sep1_day, color='black', linestyle='--', linewidth=1, alpha=0.5)

            sep30_day = datetime(2020, 9, 30).timetuple().tm_yday - start_day + 1
            ax.axhline(y=sep30_day, color='black', linestyle='--', linewidth=1, alpha=0.5)

            star_years = {1980, 1981, 1982, 1983, 1985, 1986, 1987, 1988,
                          1992, 1994, 1998, 2000, 2001, 2024}

            for i, year in enumerate(years):
                if year in star_years:
                    year_data = df[df['year'] == year]
                    sep_data = year_data[year_data['month'] == 9]
                    if not sep_data.empty:
                        avg_concentration = sep_data['concentration'].mean()
                        if not np.isnan(avg_concentration):
                            text_y = (sep1_day + sep30_day) / 2
                            ax.text(i, text_y, f"{avg_concentration:.1f}%",
                                    ha='center', va='center', fontsize=8,
                                    rotation=45)

    # 设置x轴刻度
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, ha='right')

    # 设置y轴刻度
    ax.set_yticks(month_pos)
    ax.set_yticklabels(month_labels)

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('SIC(%)', fontsize=12)
    cbar.set_ticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    cbar.set_ticklabels(['1', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])

    # 设置x轴标签（添加星号）
    star_years = {1980, 1981, 1982, 1983, 1985, 1986, 1987, 1988,
                  1992, 1994, 1998, 2000, 2001, 2024}

    xticklabels = []
    for i, year in enumerate(years):
        if year in star_years:
            xticklabels.append(f"*{year}")
        else:
            xticklabels.append(str(year))

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(xticklabels, rotation=45, ha='center')

    for label in ax.get_xticklabels():
        if label.get_text().startswith('*'):
            label.set_color('red')

    # 添加图例
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    # 保存图像
    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir,
                               f"seaice_heatmap_with_curves_{start_month:02d}-{end_month:02d}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n热力图已保存到: {output_path}")

    # 打印统计信息
    if not freeze_melt_df.empty:
        print("\n冻结融化统计信息:")
        print(freeze_melt_df[['year', 'melt_day', 'freeze_day', 'open_water_days']].to_string(index=False))


# 使用示例
if __name__ == "__main__":
    # 指定CSV文件路径
    csv_file = r"E:\2024seaice\osidailytif\chukchi_sea_ice_daily_data.csv"

    # 绘制4-12月数据
    plot_from_csv(csv_file, 4, 12)