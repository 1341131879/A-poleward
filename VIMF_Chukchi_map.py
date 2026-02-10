import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from rasterio.warp import transform as warp_transform
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

plt.rcParams['font.family'] = 'Arial'

# ==== Parameters ====
data_dir = r"E:\2024seaice\2025\ERA5Humidity\JAS_anomaly"
vimf_dir = r"E:\2024seaice\2025\ERA5VIMF_JAS_Total"
out_dir = r"E:\2024seaice\2025\ERA5Humidity\plots_sector"
os.makedirs(out_dir, exist_ok=True)


lon_min_s, lon_max_s = 140, 220
lat_min_s, lat_max_s = 49, 77

years = list(range(2020, 2025))
season = "JAS"


data_files = [os.path.join(data_dir, f"{year}_{season}_humidity_anomaly.tif") for year in years]
data_files = [f for f in data_files if os.path.exists(f)]

humidity_data_avg = None
count = 0
meta_info = None

for file_path in data_files:
    with rasterio.open(file_path) as src:
        data = src.read(1)
        if meta_info is None:
            meta_info = {'transform': src.transform, 'crs': src.crs, 'h': src.height, 'w': src.width}
        if humidity_data_avg is None:
            humidity_data_avg = data.copy()
        else:
            if data.shape == humidity_data_avg.shape:
                humidity_data_avg += data
            count += 1
count += 1
humidity_data_avg /= count


u_sum, v_sum = None, None
vimf_count = 0
for year in years:
    u_file = os.path.join(vimf_dir, f"{year}_{season}_uivwv.tif")
    v_file = os.path.join(vimf_dir, f"{year}_{season}_vivwv.tif")
    if os.path.exists(u_file) and os.path.exists(v_file):
        with rasterio.open(u_file) as src_u, rasterio.open(v_file) as src_v:
            u_d, v_d = src_u.read(1), src_v.read(1)
            if u_sum is None:
                u_sum, v_sum = u_d.copy(), v_d.copy()
            else:
                u_sum += u_d; v_sum += v_d
            vimf_count += 1

u_avg = u_sum / vimf_count if vimf_count > 0 else None
v_avg = v_sum / vimf_count if vimf_count > 0 else None


cols, rows = np.meshgrid(np.arange(meta_info['w']), np.arange(meta_info['h']))
xs, ys = rasterio.transform.xy(meta_info['transform'], rows, cols, offset='center')
lons, lats = warp_transform(meta_info['crs'], "EPSG:4326", np.array(xs).ravel(), np.array(ys).ravel())
lons = np.array(lons).reshape((meta_info['h'], meta_info['w']))
lats = np.array(lats).reshape((meta_info['h'], meta_info['w']))

lons_360 = np.where(lons < 0, lons + 360, lons)
sort_idx = np.argsort(lons_360[0, :])

lons_sorted = lons_360[:, sort_idx]
lats_sorted = lats[:, sort_idx]
humidity_sorted = humidity_data_avg[:, sort_idx]


mask_s = (lons_sorted >= lon_min_s) & (lons_sorted <= lon_max_s) & (lats_sorted >= lat_min_s) & (lats_sorted <= lat_max_s)
humidity_sorted[~mask_s] = np.nan

if u_avg is not None:
    u_sorted = u_avg[:, sort_idx]
    v_sorted = v_avg[:, sort_idx]
    u_sorted[~mask_s] = np.nan
    v_sorted[~mask_s] = np.nan


mean_val = np.nanmean(humidity_sorted)
max_val = np.nanmax(humidity_sorted)
min_val = np.nanmin(humidity_sorted)
std_val = np.nanstd(humidity_sorted)
abs_mean = np.nanmean(np.abs(humidity_sorted))

# 动态确定颜色范围 (还原你的逻辑)
data_range = max_val - min_val
if data_range > 0:
    vmin_tmp = max(min_val, mean_val - 2 * std_val)
    vmax_tmp = min(max_val, mean_val + 2 * std_val)
else:
    vmin_tmp, vmax_tmp = -1, 1
max_abs = max(abs(vmin_tmp), abs(vmax_tmp))
vmin, vmax = -max_abs, max_abs
if (vmax - vmin) < 0.1: vmin, vmax = -0.001, 0.001


proj = ccrs.NorthPolarStereo(central_longitude=180)
fig = plt.figure(figsize=(16, 12))
ax = plt.subplot(1, 1, 1, projection=proj)
ax.set_extent([lon_min_s, lon_max_s, lat_min_s, lat_max_s], crs=ccrs.PlateCarree())


land = cfeature.NaturalEarthFeature('physical', 'land', '10m', linewidth=1.5, edgecolor='darkgray', facecolor='lightgray')
ax.add_feature(land, zorder=2)
ax.coastlines(resolution='10m', zorder=3, linewidth=0.5)


boundaries = np.linspace(vmin, vmax, 11)
cmap_base = plt.get_cmap('PRGn_r')
norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap_base.N, clip=False)

pcm = ax.pcolormesh(lons_sorted, lats_sorted, humidity_sorted,
                    transform=ccrs.PlateCarree(), cmap=cmap_base, norm=norm,
                    shading='auto', zorder=2)


if u_avg is not None:
    step = 15

    q = ax.quiver(
        lons_sorted[::step, ::step],
        lats_sorted[::step, ::step],
        u_sorted[::step, ::step],
        v_sorted[::step, ::step],
        transform=ccrs.PlateCarree(),
        scale=300,  # 保持原有的缩放
        width=0.005,  # 保持原有的宽度
        color='black',  # 箭头主体颜色：黑色
        edgecolors='white',  # 箭头边框颜色：白色
        linewidth=2,  # 边框线条粗细
        zorder=4
    )
    ax.quiverkey(q, 0.9, 0.05, 30, label='30 kg m⁻¹ s⁻¹', labelpos='E', coordinates='axes', fontproperties={'size': 10})


# gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# gl.xlocator = mticker.FixedLocator(np.arange(140, 221, 10))
# gl.ylocator = mticker.FixedLocator(np.arange(50, 81, 5))
# gl.xlabel_style = {'size': 10, 'color': 'black'}
# gl.ylabel_style = {'size': 10, 'color': 'black', 'rotation': 0}


cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8, extend='both', ticks=boundaries)
cbar.set_label("Humidity Anomaly (g/kg)", fontsize=14, weight='bold')
cbar.ax.tick_params(labelsize=12)
cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

range_val = vmax - vmin
if range_val <= 1: tick_step = 0.2
elif range_val <= 2: tick_step = 0.4
elif range_val <= 5: tick_step = 1
elif range_val <= 10: tick_step = 2
else: tick_step = 5
ticks = np.arange(np.floor(vmin / tick_step) * tick_step, np.ceil(vmax / tick_step) * tick_step + 0.0001, tick_step)
cbar.set_ticks(ticks)


stats_text = f'Mean: {mean_val:.4f}\nMax: {max_val:.4f}\nMin: {min_val:.4f}\nStd: {std_val:.4f}\nAbs Mean: {abs_mean:.4f}'
ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), verticalalignment='bottom')


count_text = f'Data points: {np.sum(~np.isnan(humidity_sorted)):,}'
ax.text(0.98, 0.02, count_text, transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='gray'),
        verticalalignment='bottom', horizontalalignment='right')


title = f"Humidity Anomaly Composite with VIMF (Chukchi Sector)\n{years[0]}-{years[-1]} {season} Average (n={count} years)"
ax.set_title(title, fontsize=16, pad=20, fontweight='bold')


out_png = os.path.join(out_dir, f"Humidity_VIMF_Sector_{season}_{years[0]}-{years[-1]}.png")
plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
