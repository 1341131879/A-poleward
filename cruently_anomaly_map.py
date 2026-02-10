import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import rasterio
from rasterio.transform import xy

import matplotlib.pyplot as plt

# ====== 年均 JAS v 的路径（2002-2025） ======
v_mean_dir = r"E:\2024seaice\2025\cruently\v_mean_JAS_all"

# 输出：气候态 & 异常 GeoTIFF
clim_tif = os.path.join(v_mean_dir, "2002_2019_JAS_oscar_v_climatology.tif")
anom_dir = r"E:\2024seaice\2025\cruently\v_JAS_anom_2020_2025"
os.makedirs(anom_dir, exist_ok=True)

# 输出图的目录
fig_clim_dir = r"E:\2024seaice\2025\cruently\fig_JAS_v_climatology"
fig_anom_dir = r"E:\2024seaice\2025\cruently\fig_JAS_v_anom_2020_2025"
fig_mean2020_24_dir = r"E:\2024seaice\2025\cruently\fig_JAS_v_mean_2020_2024"
os.makedirs(fig_clim_dir, exist_ok=True)
os.makedirs(fig_anom_dir, exist_ok=True)
os.makedirs(fig_mean2020_24_dir, exist_ok=True)

# 白令海峡区域
lon_min, lon_max = 160, 210
lat_min, lat_max = 60, 75

# 颜色范围：气候态用 0.3，异常用 0.1（你可以之后自己调）
v_abs_max = 0.3
anom_abs_max = 0.1

# ====== 1. 计算 2002-2019 JAS v 气候态 ======
clim_years = range(2002, 2020)

v_sum = None
count_valid = None
profile_base = None
nodata_val = None

print("Computing 2002-2019 JAS climatology of v ...")

for year in clim_years:
    v_tif = os.path.join(v_mean_dir, f"{year}_JAS_oscar_v_mean.tif")
    if not os.path.exists(v_tif):
        print(f"  Missing {v_tif}, skip this year.")
        continue

    with rasterio.open(v_tif) as src:
        v = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
        if nodata is None:
            nodata = -999.0

        if profile_base is None:
            profile_base = src.profile.copy()
            profile_base.update(
                driver="GTiff",
                count=1,
                dtype=rasterio.float32,
                nodata=nodata,
            )
            nodata_val = nodata

        if v_sum is None:
            v_sum = np.zeros_like(v, dtype=np.float64)
            count_valid = np.zeros_like(v, dtype=np.int32)

    # 有效值
    mask_valid = (v != nodata) & np.isfinite(v)
    v_sum[mask_valid] += v[mask_valid]
    count_valid[mask_valid] += 1

# 生成气候态场
if v_sum is None:
    raise RuntimeError("No valid v_mean files found for 2002-2019!")

v_clim = np.full_like(v_sum, nodata_val, dtype=np.float32)
mask_clim = count_valid > 0
v_clim[mask_clim] = (v_sum[mask_clim] / count_valid[mask_clim]).astype(np.float32)

# 保存气候态 GeoTIFF
with rasterio.open(clim_tif, "w", **profile_base) as dst:
    dst.write(v_clim, 1)

print(f"Climatology saved: {clim_tif}")

# ====== 画 2002-2019 气候态 v 图（白令海峡） ======
height, width = v_clim.shape
rows = np.arange(height)
cols = np.arange(width)

lon_1d, _ = xy(transform, 0 * np.ones_like(cols), cols, offset="center")
lon_1d = np.array(lon_1d)
_, lat_1d = xy(transform, rows, 0 * np.ones_like(rows), offset="center")
lat_1d = np.array(lat_1d)

lon_mask = (lon_1d >= lon_min) & (lon_1d <= lon_max)
lat_mask = (lat_1d >= lat_min) & (lat_1d <= lat_max)
lon_idx = np.where(lon_mask)[0]
lat_idx = np.where(lat_mask)[0]

if len(lon_idx) == 0 or len(lat_idx) == 0:
    raise ValueError("Region not found in climatology data, adjust lon/lat range.")

col_start, col_end = lon_idx[0], lon_idx[-1] + 1
row_start, row_end = lat_idx[0], lat_idx[-1] + 1

v_clim_sub = v_clim[row_start:row_end, col_start:col_end]
lon_sub = lon_1d[col_start:col_end]
lat_sub = lat_1d[row_start:row_end]
Lon, Lat = np.meshgrid(lon_sub, lat_sub)

# nodata -> nan
v_clim_sub = np.where(v_clim_sub == nodata_val, np.nan, v_clim_sub)

proj = ccrs.NorthPolarStereo(central_longitude=180)
pc = ccrs.PlateCarree()

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=proj)
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=pc)


# # 经纬网格线改为实线
# gl = ax.gridlines(draw_labels=False, linewidth=0.5,
#                   color='gray', alpha=0.5, linestyle='-')

im = ax.pcolormesh(
    Lon, Lat, v_clim_sub,
    transform=pc,
    cmap="RdBu_r",
    vmin=-v_abs_max,
    vmax=v_abs_max,
    shading="auto"
)
ax.coastlines(resolution='110m', linewidth=0.8)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.BORDERS, linewidth=0.3,linestyle='-')
cb = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, pad=0.05)
cb.set_label("Meridional velocity v (m/s)")

ax.set_title(
    "Meridional Surface Currents (v) near Bering Strait\n"
    "OSCAR JAS Mean 2002–2019 (Climatology)",
    fontsize=11
)

plt.tight_layout()
clim_png = os.path.join(fig_clim_dir, "2002_2019_JAS_oscar_v_bering_climatology.png")
plt.savefig(clim_png, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Climatology figure saved: {clim_png}")

# ====== 2. 计算并画 2020-2025 的 JAS 异常 (year - climatology) ======
anom_years = range(2020, 2026)

for year in anom_years:
    print(f"\nComputing anomaly for {year} ...")

    v_tif = os.path.join(v_mean_dir, f"{year}_JAS_oscar_v_mean.tif")
    if not os.path.exists(v_tif):
        print(f"  Missing {v_tif}, skip.")
        continue

    with rasterio.open(v_tif) as src:
        v_year = src.read(1).astype(np.float32)
        transform_y = src.transform
        nodata_y = src.nodata
        if nodata_y is None:
            nodata_y = nodata_val

    # 确保大小一致
    if v_year.shape != v_clim.shape:
        print(f"  Shape mismatch for {year}, skip.")
        continue

    # 构建异常场：year - climatology
    v_anom = np.full_like(v_year, nodata_val, dtype=np.float32)
    mask_valid = (
        (v_year != nodata_y) & (v_clim != nodata_val) &
        np.isfinite(v_year) & np.isfinite(v_clim)
    )
    v_anom[mask_valid] = (v_year[mask_valid] - v_clim[mask_valid]).astype(np.float32)

    # 保存 anomaly GeoTIFF
    anom_tif = os.path.join(anom_dir, f"{year}_JAS_oscar_v_anom_2002_2019.tif")
    anom_profile = profile_base.copy()
    with rasterio.open(anom_tif, "w", **anom_profile) as dst:
        dst.write(v_anom, 1)
    print(f"  Anomaly tif saved: {anom_tif}")

    # 对异常场做同样的区域裁剪
    v_anom_sub = v_anom[row_start:row_end, col_start:col_end]
    v_anom_sub = np.where(v_anom_sub == nodata_val, np.nan, v_anom_sub)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=proj)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=pc)

    # 先画海岸线（底层）
    ax.coastlines(resolution='110m', linewidth=0.8)

    # 只画负值：正值（>=0） 和 NaN 都不要显示
    data_to_plot = np.ma.masked_where(v_anom_sub >= 0, v_anom_sub)
    data_to_plot = np.ma.masked_invalid(data_to_plot)  # 再把 NaN 也 mask 掉

    # 蓝色系 colormap，掩膜区域设为“透明”而不是白色
    cmap_neg = plt.get_cmap("Blues_r").copy()
    cmap_neg.set_bad((1, 1, 1, 0))  # RGBA, alpha=0 → 完全透明

    im = ax.pcolormesh(
        Lon, Lat, data_to_plot,
        transform=pc,
        cmap=cmap_neg,
        vmin=-anom_abs_max,
        vmax=0.0,
        shading="auto"
    )

    # 陆地最后盖上（最上层）
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='-', zorder=11)

    cb = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, pad=0.05)
    cb.set_label("Negative v anomaly (m/s)\n(Year JAS − 2002–2019 JAS,\npositive omitted)")

    ax.set_title(
        f"Meridional Surface Currents (v) Anomaly near Bering Strait\n"
        f"OSCAR JAS {year} − Climatology (2002–2019)",
        fontsize=11
    )

    plt.tight_layout()
    anom_png = os.path.join(fig_anom_dir, f"{year}_JAS_oscar_v_anom_bering.png")
    plt.savefig(anom_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Anomaly figure saved: {anom_png}")

# ====== 3. 计算并画 2020-2024 JAS 平均 ======
print("\nComputing 2020-2024 JAS mean of v ...")

mean_2020_24_years = range(2020, 2025)

v_sum_2020_24 = None
count_valid_2020_24 = None

for year in mean_2020_24_years:
    v_tif = os.path.join(v_mean_dir, f"{year}_JAS_oscar_v_mean.tif")
    if not os.path.exists(v_tif):
        print(f"  Missing {v_tif}, skip {year}.")
        continue

    with rasterio.open(v_tif) as src:
        v_y = src.read(1).astype(np.float32)
        nodata_y = src.nodata
        if nodata_y is None:
            nodata_y = nodata_val

    if v_sum_2020_24 is None:
        v_sum_2020_24 = np.zeros_like(v_y, dtype=np.float64)
        count_valid_2020_24 = np.zeros_like(v_y, dtype=np.int32)

    mask_valid_y = (v_y != nodata_y) & np.isfinite(v_y)
    v_sum_2020_24[mask_valid_y] += v_y[mask_valid_y]
    count_valid_2020_24[mask_valid_y] += 1

if v_sum_2020_24 is None:
    print("No valid data found for 2020-2024, skip 2020-2024 mean.")
else:
    v_mean_2020_24 = np.full_like(v_sum_2020_24, nodata_val, dtype=np.float32)
    mask_2020_24 = count_valid_2020_24 > 0
    v_mean_2020_24[mask_2020_24] = (v_sum_2020_24[mask_2020_24] / count_valid_2020_24[mask_2020_24]).astype(np.float32)

    # 保存 2020-2024 JAS 平均 GeoTIFF
    mean_2020_24_tif = os.path.join(v_mean_dir, "2020_2024_JAS_oscar_v_mean.tif")
    with rasterio.open(mean_2020_24_tif, "w", **profile_base) as dst:
        dst.write(v_mean_2020_24, 1)
    print(f"2020-2024 JAS mean tif saved: {mean_2020_24_tif}")

    # 区域裁剪并画图
    v_mean_2020_24_sub = v_mean_2020_24[row_start:row_end, col_start:col_end]
    v_mean_2020_24_sub = np.where(v_mean_2020_24_sub == nodata_val, np.nan, v_mean_2020_24_sub)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=proj)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=pc)


    # gl = ax.gridlines(draw_labels=False, linewidth=0.5,
    #                   color='gray', alpha=0.5, linestyle='-')

    im = ax.pcolormesh(
        Lon, Lat, v_mean_2020_24_sub,
        transform=pc,
        cmap="RdBu_r",
        vmin=-v_abs_max,
        vmax=v_abs_max,
        shading="auto"
    )
    ax.coastlines(resolution='110m', linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3,linestyle='-')
    cb = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, pad=0.05)
    cb.set_label("Meridional velocity v (m/s)")

    ax.set_title(
        "Meridional Surface Currents (v) near Bering Strait\n"
        "OSCAR JAS Mean 2020–2024",
        fontsize=11
    )

    plt.tight_layout()
    mean_2020_24_png = os.path.join(fig_mean2020_24_dir, "2020_2024_JAS_oscar_v_bering_mean.png")
    plt.savefig(mean_2020_24_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"2020-2024 mean figure saved: {mean_2020_24_png}")

print("\nAll climatology, 2020-2025 anomalies, and 2020-2024 mean maps done.")
