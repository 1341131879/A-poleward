import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import detrend

plt.rcParams['font.family'] = 'Arial'

# ---------- Parameters ----------
bbox_latlon = (-180, 66, -160, 73)  # Lon/Lat bounds
year_min, year_max = 1979, 2025
climatology_start, climatology_end = 2002, 2019  # Climatology base period
months_target = {7, 8, 9}  # JAS season
seconds_per_day = 86400  # For unit conversion

# ---------- File paths ----------
pna_file = Path(r"E:\2024seaice\2025\SST\PNA_monthly_normals.xlsx")
out_plot = Path(r"E:\2024seaice\2025\SST\analysis\Variables_Detrended_Anomalies_Curves_Combined_with_q.png")
data_dirs = {
    "SW": Path(r"E:\2024seaice\2025\flux\sw"),
    "LW": Path(r"E:\2024seaice\2025\flux\lw"),
    "NET": Path(r"E:\2024seaice\2025\flux\net"),
    "DLR": Path(r"E:\2024seaice\2025\flux\dlr"),
    "DSR": Path(r"E:\2024seaice\2025\flux\dsr"),
    "LCC": Path(r"E:\2024seaice\2025\cloud\cloudtif"),
    "T2m": Path(r"E:\2024seaice\2025\T2m_Monthly\t2m"),
    "SST": Path(r"E:\2024seaice\2025\SST\sst"),
    "SIC": Path(r"E:\2024seaice\osimonthlytif"),
    "q": Path(r"E:\2024seaice\2025\ERA5Humidity\q_average")  # 添加湿度平均数据路径
}

# Projection transform for SIC data (EPSG:3413)
bbox_proj = transform_bounds("EPSG:4326", "EPSG:3413", *bbox_latlon, densify_pts=21)


# ---------- Unit conversion ----------
def convert_j_to_w(flux_j):
    """Convert flux from J m⁻² to W m⁻² (assuming daily accumulated)"""
    return flux_j / seconds_per_day


def convert_kgkg_to_gkg(q_kgkg):
    """Convert specific humidity from kg/kg to g/kg"""
    return q_kgkg * 1000


def fill_pos_neg(ax, x, y, pos_color, hatch_color="black",
                 alpha_pos=0.25, hatch_pattern="///"):
    """
    >0 用 pos_color 填充
    <0 用斜杠黑色 hatch 填充
    自动补全穿越 0 的交点
    """

    x = np.array(x)
    y = np.array(y)

    # 构造新的点（含穿越 0 的补点）
    new_x = [x[0]]
    new_y = [y[0]]

    for i in range(1, len(x)):
        x0, y0 = new_x[-1], new_y[-1]
        x1, y1 = x[i], y[i]

        # 穿越 0，补交点
        if (y0 > 0 and y1 < 0) or (y0 < 0 and y1 > 0):
            t = -y0 / (y1 - y0)
            x_cross = x0 + t * (x1 - x0)
            new_x.append(x_cross)
            new_y.append(0.0)

        new_x.append(x1)
        new_y.append(y1)

    new_x = np.array(new_x)
    new_y = np.array(new_y)

    # === 正值填充（保持原色） ===
    ax.fill_between(
        new_x, 0, new_y,
        where=new_y > 0,
        color=pos_color,
        alpha=alpha_pos,
        interpolate=True,
        zorder=1
    )

    # === 负值区域用黑色斜杠填充 ===
    ax.fill_between(
        new_x, 0, new_y,
        where=new_y < 0,
        facecolor="none",  # 不填色
        edgecolor=hatch_color,  # 斜杠颜色
        hatch=hatch_pattern,  # 斜杠样式
        linewidth=0.0,
        interpolate=True,
        zorder=1
    )


# ---------- Read PNA data ----------
print("Reading PNA index data...")
pna_df = pd.read_excel(pna_file)
pna_long = (
    pna_df
    .melt(id_vars="Year", var_name="Month", value_name="PNA")
    .assign(Month=lambda d: d["Month"].map({
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }))
    .query("@year_min <= Year <= @year_max and Month in @months_target")
)

# Calculate seasonal mean PNA
pna_seasonal = (
    pna_long
    .groupby("Year")
    .agg(PNA_JAS=("PNA", "mean"))
    .dropna()
)


# ---------- Read variable data ----------
def read_JAS_mean(data_dir: Path, varname: str, tif_re_pattern: str, is_epsg3413=False):
    """Read JAS seasonal mean data"""
    records = []
    tif_re = re.compile(tif_re_pattern, re.I)

    for tif_path in tqdm(sorted(data_dir.glob("*.tif")), desc=f"Reading {varname}"):
        m = tif_re.match(tif_path.name)
        if not m:
            continue

        # 对湿度数据特殊处理
        if varname == "q":
            # 湿度文件名格式: YYYYMM01_q.tif
            yr, mo = int(m.group(1)), int(m.group(2))
        else:
            yr, mo = int(m.group(1)), int(m.group(2))

        if not (year_min <= yr <= year_max and mo in months_target):
            continue

        with rasterio.open(tif_path) as src:
            if is_epsg3413:
                if src.crs.to_epsg() != 3413:
                    raise ValueError(f"{tif_path} is not in EPSG:3413!")
                window = from_bounds(*bbox_proj, transform=src.transform)
            else:
                window = from_bounds(*bbox_latlon, transform=src.transform)
            data = src.read(1, window=window, masked=True)

            # Replace invalid values
            data = np.ma.masked_invalid(data)

            # For SST, mask values below freezing point
            if varname == "SST":
                ice_threshold = 271.49  # Kelvin
                data = np.ma.masked_where(data < ice_threshold, data)

            mean_val = np.ma.mean(data)

            # Skip if all values are invalid
            if np.ma.is_masked(mean_val):
                continue

            value = float(mean_val)

            # 辐射通量转换为 W/m²（如果原始是 J/m²）
            if varname in ["DLR", "DSR"]:
                value = convert_j_to_w(value)
            # 低云量转换为百分比
            elif varname == "LCC":
                value = value * 100
            # 温度转换为摄氏度
            elif varname in ["T2m", "SST"]:
                if value > 200:  # 假设是开尔文
                    value = value - 273.15
            # 湿度单位转换：kg/kg → g/kg
            elif varname == "q":
                value = convert_kgkg_to_gkg(value)

            records.append({"Year": yr, "Month": mo, varname: value})

    df = pd.DataFrame.from_records(records)
    return df.groupby("Year")[varname].mean().reset_index()


# ---------- Merge data ----------
data_vars_patterns = {
    "SW": r"(\d{4})(\d{2})(\d{2})_sw\.tif$",
    "LW": r"(\d{4})(\d{2})(\d{2})_lw\.tif$",
    "NET": r"(\d{4})(\d{2})(\d{2})_net\.tif$",
    "DLR": r"(\d{4})(\d{2})(\d{2})_dlr\.tif$",
    "DSR": r"(\d{4})(\d{2})(\d{2})_dsr\.tif$",
    "LCC": r"(\d{4})(\d{2})(\d{2})_lcc\.tif$",
    "T2m": r"(\d{4})(\d{2})(\d{2})_t2m\.tif$",
    "SST": r"(\d{4})(\d{2})(\d{2})_sst\.tif$",
    "SIC": r"osi_monthly_(\d{4})(\d{2})\.tif$",
    "q": r"(\d{4})(\d{2})01_q\.tif$"  # 湿度文件名模式: YYYYMM01_q.tif
}

df = pna_seasonal.copy()
for var, pattern in data_vars_patterns.items():
    try:
        var_df = read_JAS_mean(data_dirs[var], var, pattern, is_epsg3413=(var == "SIC"))
        df = df.merge(var_df, on="Year", how="inner")
    except Exception as e:
        print(f"Error processing {var}: {e}")
        continue

# Remove rows with NaN values
df = df.dropna()

# ===== Calculate anomalies relative to climatology =====
print(f"\nCalculating anomalies (Climatology base: {climatology_start}-{climatology_end})")

clim_mask = (df["Year"] >= climatology_start) & (df["Year"] <= climatology_end)

clim_means = {}
for var in data_vars_patterns.keys():
    if var in df.columns:
        clim_means[var] = df.loc[clim_mask, var].mean()

anomalies = df.copy()
for var in data_vars_patterns.keys():
    if var in df.columns:
        anomalies[var] = df[var] - clim_means[var]

# ===== PNA anomaly & detrend =====
if "PNA_JAS" in df.columns:
    pna_clim = df.loc[clim_mask, "PNA_JAS"].mean()
    anomalies["PNA_JAS"] = df["PNA_JAS"] - pna_clim

# 打印各变量的气候态平均值
print("\nClimatology means (2002-2019):")
for var, mean_val in clim_means.items():
    unit = ""
    if var == "q":
        unit = "g/kg"
    elif var in ["T2m", "SST"]:
        unit = "°C"
    elif var == "LCC" or var == "SIC":
        unit = "%"
    elif var in ["SW", "LW", "NET", "DLR", "DSR"]:
        unit = "W/m²"

    print(f"  {var}: {mean_val:.3f} {unit}")

if "PNA_JAS" in df.columns:
    print(f"  PNA_JAS: {pna_clim:.3f}")

print("Applying detrending to anomalies...")
detrended_anomalies = anomalies.copy()

for var in list(data_vars_patterns.keys()) + ["PNA_JAS"]:
    if var in anomalies.columns:
        try:
            detrended_anomalies[var] = detrend(anomalies[var].values)
        except Exception as e:
            print(f"Error detrending {var}: {e}")
            detrended_anomalies[var] = np.nan

# ===== Create single figure with all variables (including q) =====
print("Creating combined time series plot with humidity...")

# 定义要绘制的变量，将q加在合适的位置
variables = [
    ("LCC", "LCC", r"%", "#FF6B6B"),
    ("DSR", "DSR", r"$\mathrm{W\,m^{-2}}$", "#1F77B4"),
    ("SW", "SW", r"$\mathrm{W\,m^{-2}}$", "#65B89E"),
    #("LW", "LW", r"$\mathrm{W\,m^{-2}}$", "#B22222"),
    ("DLR", "DLR", r"$\mathrm{W\,m^{-2}}$", "#EE9834"),
    ("q", "q", r"$\mathrm{g·kg^{-1}}$", "#8A2BE2"),  # 添加湿度，使用紫色
    ("T2m", "T2m", r"$^\circ$C", "blue"),
    ("SST", "SST", r"$^\circ$C", "#FF8C00"),
    ("SIC", "SIC", r"%", "black"),
    #("PNA_JAS", "PNA", "", "#FF1493"),  # 可选：添加PNA
]

# 过滤掉没有数据的变量
available_variables = []
for var in variables:
    if var[0] in detrended_anomalies.columns:
        available_variables.append(var)
    else:
        print(f"Warning: {var[0]} not in detrended anomalies, skipping")

print(f"\nPlotting {len(available_variables)} variables: {[v[0] for v in available_variables]}")

fig, axs = plt.subplots(
    len(available_variables), 1, figsize=(12, 16),
    sharex=True, gridspec_kw={'hspace': 0.05}
)

years = detrended_anomalies["Year"].values

for i, (var_name, var_abbr, var_unit, color) in enumerate(available_variables):
    ax = axs[i]

    if var_name not in detrended_anomalies.columns:
        ax.text(0.5, 0.5, f'No data for {var_name}',
                transform=ax.transAxes, ha='center', va='center')
        continue

    data = detrended_anomalies[var_name].values

    # === 关键修改：计算对称 y 轴范围实现 0 居中 ===
    max_abs_val = np.nanmax(np.abs(data))
    if max_abs_val == 0 or np.isnan(max_abs_val):
        max_abs_val = 1.0  # 防止全 0 数据导致除零错误

    # 给予 10% 的边距空间，防止数据点贴边
    y_limit = max_abs_val * 1.1
    ax.set_ylim(-y_limit, y_limit)
    # ==========================================

    # 画线
    ax.plot(
        years, data,
        color=color,
        linewidth=1.5,
        marker='o' if i % 2 == 0 else 's',
        markersize=0,
        zorder=2
    )

    # 统一设置脊柱（Spines）
    ax.spines['top'].set_visible(False)

    # 偶数行：左轴；奇数行：右轴
    if i % 2 == 0:
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()
        ax.tick_params(right=False, labelright=False)
    else:
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(True)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.tick_params(left=False, labelleft=False)

    # y 轴标签
    ylabel = f'{var_abbr} ({var_unit})' if var_unit else var_abbr
    ax.set_ylabel(ylabel, fontsize=14, color=color)

    # 零线（增强零线的存在感，因为它现在是几何中心）
    ax.axhline(
        y=0, color='black',
        linestyle='-', alpha=0.3, linewidth=0.8,
        zorder=1
    )

    # 阴影填充
    fill_pos_neg(ax, years, data, pos_color=color)

    # 控制 X 轴显示
    if i < len(available_variables) - 1:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)
# 底部子图：显示 x 轴
bottom_ax = axs[-1]
bottom_ax.set_xlabel('Year', fontsize=12, fontweight='bold')

# 设置 X 轴范围
bottom_ax.set_xlim(1979, 2024)

# 设置 5 年间隔 + 显示 2024
ticks = list(range(1980, 2025, 5))
if 2024 not in ticks:
    ticks.append(2024)
ticks = sorted(ticks)

bottom_ax.set_xticks(ticks)
bottom_ax.set_xticklabels(ticks, fontsize=14)

# 只关掉顶部脊
bottom_ax.spines['top'].set_visible(False)

fig.suptitle(
    'Detrended Anomalies of Meteorological Variables (1979–2025) - JAS Season',
    fontsize=14, fontweight='bold', y=0.98
)

plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
plt.show()

# ===== Save detrended anomalies to CSV =====
output_csv = Path(r"E:\2024seaice\2025\SST\analysis\detrended_anomalies_1979_2025_with_q.csv")
detrended_anomalies.to_csv(output_csv, index=False)
print(f"\nDetrended anomalies saved to: {output_csv}")

# ===== 打印湿度异常值的统计信息 =====
if "q" in detrended_anomalies.columns:
    q_clim = clim_means.get("q", np.nan)
    q_anomaly_mean = detrended_anomalies["q"].mean()

    print(f"\nHumidity (q) statistics:")
    print(f"  Original unit: kg/kg")
    print(f"  Converted to: g/kg (multiplied by 1000)")
    print(f"  Climatology mean (2002-2019): {q_clim:.3f} g/kg")
    print(f"  Years with data: {len(detrended_anomalies['q'].dropna())}")
    print(f"  Mean detrended anomaly: {q_anomaly_mean:.3f} g/kg")
    print(f"  Std detrended anomaly: {detrended_anomalies['q'].std():.3f} g/kg")
    print(f"  Min detrended anomaly: {detrended_anomalies['q'].min():.3f} g/kg")
    print(f"  Max detrended anomaly: {detrended_anomalies['q'].max():.3f} g/kg")

    # 打印原始数据范围（转换前）
    if "q" in df.columns:
        q_raw = df["q"].values
        print(f"\n  Original humidity (kg/kg) range:")
        print(f"    Mean: {q_raw.mean():.6f} kg/kg")
        print(f"    Min: {q_raw.min():.6f} kg/kg")
        print(f"    Max: {q_raw.max():.6f} kg/kg")

    # 打印湿度数据的年份范围
    q_years = detrended_anomalies[detrended_anomalies['q'].notna()]['Year'].tolist()
    if q_years:
        print(f"\n  Available years: {min(q_years)}-{max(q_years)}")
        print(f"  Number of years: {len(q_years)}")