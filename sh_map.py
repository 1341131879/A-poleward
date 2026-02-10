import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from rasterio.warp import transform as warp_transform
import matplotlib.ticker as mticker
from rasterio.merge import merge
import glob

plt.rcParams['font.family'] = 'Arial'

# ==== Parameters ====
data_dir = r"E:\2024seaice\2025\ERA5Humidity\JAS_anomaly"
out_dir = r"E:\2024seaice\2025\ERA5Humidity\plots"
os.makedirs(out_dir, exist_ok=True)

# Full Arctic extent (-180–180)
lon_min, lon_max = -179.9, 180
lat_min, lat_max = 39, 90

years = list(range(2020, 2025))  # 2020-2024
season = "JAS"

print(f"开始处理 {years[0]}-{years[-1]} 年 {season} 季节湿度异常平均...")

# 收集所有年份的数据文件
data_files = []
for year in years:
    humidity_file = os.path.join(data_dir, f"{year}_{season}_humidity_anomaly.tif")
    if os.path.exists(humidity_file):
        data_files.append(humidity_file)
        print(f"找到文件: {year}_{season}_humidity_anomaly.tif")
    else:
        print(f"⚠️ 警告: 文件不存在: {humidity_file}")

if not data_files:
    print("❌ 没有找到任何数据文件")
    exit()

print(f"共找到 {len(data_files)} 个数据文件")

# 读取并计算平均值
humidity_data_avg = None
count = 0
meta_info = None

for file_path in data_files:
    with rasterio.open(file_path) as src:
        data = src.read(1)

        # 保存第一个文件的元数据
        if meta_info is None:
            src_transform = src.transform
            crs = src.crs
            height, width = src.height, src.width
            meta_info = {
                'transform': src_transform,
                'crs': crs,
                'height': height,
                'width': width
            }

        # 初始化或累加数据
        if humidity_data_avg is None:
            humidity_data_avg = data.copy()
        else:
            # 检查数据维度是否一致
            if data.shape == humidity_data_avg.shape:
                humidity_data_avg += data
            else:
                print(f"⚠️ 数据维度不匹配: {file_path}")
                continue

        count += 1

if count == 0:
    print("❌ 没有有效数据可处理")
    exit()

# 计算平均值
humidity_data_avg = humidity_data_avg / count
print(f"计算了 {count} 个年份的平均值")

# 生成经纬度网格
cols, rows = np.meshgrid(np.arange(meta_info['width']), np.arange(meta_info['height']))
xs, ys = rasterio.transform.xy(meta_info['transform'], rows, cols, offset='center')
lons, lats = warp_transform(meta_info['crs'], "EPSG:4326", np.array(xs).ravel(), np.array(ys).ravel())
lons = np.array(lons).reshape((meta_info['height'], meta_info['width']))
lats = np.array(lats).reshape((meta_info['height'], meta_info['width']))

# Arctic mask
mask = (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)
humidity_data_avg[~mask] = np.nan

# 计算统计信息
mean_val = np.nanmean(humidity_data_avg)
max_val = np.nanmax(humidity_data_avg)
min_val = np.nanmin(humidity_data_avg)
std_val = np.nanstd(humidity_data_avg)
abs_mean = np.nanmean(np.abs(humidity_data_avg))  # 平均绝对异常

# 动态确定颜色范围
# 使用平均值±2倍标准差，但设置最小范围
data_range = max_val - min_val
if data_range > 0:
    vmin = max(min_val, mean_val - 2 * std_val)
    vmax = min(max_val, mean_val + 2 * std_val)
else:
    vmin, vmax = -1, 1

# 确保颜色范围对称（适用于异常数据）
max_abs = max(abs(vmin), abs(vmax))
vmin, vmax = -max_abs, max_abs

# 如果范围太小，使用固定范围
if (vmax - vmin) < 0.1:
    vmin, vmax = -0.001, 0.001
    print("使用固定颜色范围: -1 to 1")

print(f"湿度异常统计 (2020-2024平均):")
print(f"  Mean: {mean_val:.4f}")
print(f"  Min: {min_val:.4f}")
print(f"  Max: {max_val:.4f}")
print(f"  Std: {std_val:.4f}")
print(f"  Mean Absolute: {abs_mean:.4f}")
print(f"  颜色范围: vmin={vmin:.4f}, vmax={vmax:.4f}")

# ---- Plotting ----
proj = ccrs.NorthPolarStereo(central_longitude=180)
fig = plt.figure(figsize=(16, 12))
ax = plt.subplot(1, 1, 1, projection=proj)

# Set extent for full Arctic
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# 添加地理特征
land = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                    linewidth=1.5, edgecolor='darkgray',
                                    facecolor='lightgray')  # 改为浅灰色填充
ax.add_feature(land, zorder=2)
ax.coastlines(resolution='10m', zorder=3, linewidth=0.5)

# 添加海洋特征
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
                                     edgecolor='none',
                                     facecolor='lightblue')
ax.add_feature(ocean, zorder=1, alpha=0.3)

# 绘制湿度异常
pcm = ax.pcolormesh(
    lons, lats, humidity_data_avg,
    transform=ccrs.PlateCarree(),
    cmap='PRGn_r',
    shading='auto',
    vmin=vmin,
    vmax=vmax,
    zorder=2
)

# Gridlines
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
gl.ylocator = mticker.FixedLocator(np.arange(40, 91, 10))
gl.top_labels = True
gl.bottom_labels = True
gl.right_labels = True
gl.left_labels = True
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black', 'rotation': 0}

# 添加colorbar
cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8,
                    extend='both')  # 添加箭头指示超出范围的值
# 请根据您的湿度异常单位修改下面的标签
# 如果是比湿异常，单位可能是 g/kg 或 kg/kg
cbar.set_label("Humidity Anomaly (g/kg)", fontsize=14, weight='bold')
cbar.ax.tick_params(labelsize=12)

# 设置colorbar刻度
# 根据范围自动确定刻度间隔
range_val = vmax - vmin
if range_val <= 1:
    tick_step = 0.2
elif range_val <= 2:
    tick_step = 0.4
elif range_val <= 5:
    tick_step = 1
elif range_val <= 10:
    tick_step = 2
else:
    tick_step = 5

ticks = np.arange(np.floor(vmin / tick_step) * tick_step,
                  np.ceil(vmax / tick_step) * tick_step + 0.0001,
                  tick_step)
cbar.set_ticks(ticks)


# 添加数据点数量信息
count_text = f'Data points: {np.sum(~np.isnan(humidity_data_avg)):,}'
ax.text(0.98, 0.02, count_text, transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9,
                  edgecolor='gray'),
        verticalalignment='bottom', horizontalalignment='right')

# 标题
title = f"Humidity Anomaly Composite\n{years[0]}-{years[-1]} {season} Average\n(n={count} years)"
ax.set_title(title, fontsize=16, pad=20, fontweight='bold')

# 保存图片
out_png = os.path.join(out_dir, f"Humidity_Anomaly_{years[0]}-{years[-1]}_{season}_Average_FullArctic.png")
plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✅ 多年代平均湿度异常图保存为: {out_png}")

# 可选：保存处理后的平均数据
output_tiff = os.path.join(out_dir, f"Humidity_Anomaly_{years[0]}-{years[-1]}_{season}_Average.tif")
profile = {
    'driver': 'GTiff',
    'height': meta_info['height'],
    'width': meta_info['width'],
    'count': 1,
    'dtype': rasterio.float32,
    'crs': meta_info['crs'],
    'transform': meta_info['transform'],
    'compress': 'lzw',
    'nodata': np.nan
}

with rasterio.open(output_tiff, 'w', **profile) as dst:
    dst.write(humidity_data_avg.astype(np.float32), 1)

print(f"✅ 平均数据保存为: {output_tiff}")



