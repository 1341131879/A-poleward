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


flux_dir = r"E:\2024seaice\2025\flux\anomaly_results"
out_dir = r"E:\2024seaice\2025\flux\anomaly_plots_sector"
os.makedirs(out_dir, exist_ok=True)


lon_min_sector, lon_max_sector = 140, 220
lat_min_sector, lat_max_sector = 49, 77

flux_components = {
    'sh': {'name': 'Sensible Heat Flux', 'unit': 'W/m²', 'cmap': 'RdBu_r', 'vmin': -20, 'vmax': 20},
    'sw': {'name': 'Shortwave Radiation', 'unit': 'W/m²', 'cmap': 'RdBu_r', 'vmin': -20, 'vmax': 20},
    'lw': {'name': 'Longwave Radiation', 'unit': 'W/m²', 'cmap': 'RdBu_r', 'vmin': -20, 'vmax': 20},
    'lh': {'name': 'Latent Heat Flux', 'unit': 'W/m²', 'cmap': 'RdBu_r', 'vmin': -20, 'vmax': 20}
}

season = "JAS"
period = "2020-2024"
reference_period = "2002-2019"


for flux_type, flux_info in flux_components.items():
    flux_file = os.path.join(flux_dir, f"{flux_type}_2020-2024_JAS_anomaly.tif")

    if not os.path.exists(flux_file):
        print(f"⚠️ 文件不存在，跳过: {flux_file}")
        continue

    print(f"🚀 正在绘制 {flux_info['name']} 扇区图...")


    with rasterio.open(flux_file) as src:
        data = src.read(1)
        src_transform = src.transform
        crs = src.crs
        height, width = src.height, src.width


        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src_transform, rows, cols, offset='center')
        lons, lats = warp_transform(crs, "EPSG:4326", np.array(xs).ravel(), np.array(ys).ravel())
        lons = np.array(lons).reshape((height, width))
        lats = np.array(lats).reshape((height, width))


    lons_360 = np.where(lons < 0, lons + 360, lons)


    lon_1d = lons_360[0, :]
    sort_idx = np.argsort(lon_1d)


    lons_sorted = lons_360[:, sort_idx]
    lats_sorted = lats[:, sort_idx]
    data_sorted = data[:, sort_idx]


    mask = (lons_sorted >= lon_min_sector) & (lons_sorted <= lon_max_sector) & \
           (lats_sorted >= lat_min_sector) & (lats_sorted <= lat_max_sector)
    data_final = np.where(mask, data_sorted, np.nan)


    proj = ccrs.NorthPolarStereo(central_longitude=180)  # 180度对准观察者
    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(1, 1, 1, projection=proj)


    ax.set_extent([lon_min_sector, lon_max_sector, lat_min_sector, lat_max_sector], crs=ccrs.PlateCarree())


    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='gray', facecolor='gray')
    ax.add_feature(land, zorder=3)


    levels = np.linspace(flux_info['vmin'], flux_info['vmax'], 11)
    base_cmap = plt.get_cmap(flux_info['cmap'])
    cmap = mcolors.ListedColormap(base_cmap(np.linspace(0, 1, len(levels) - 1)))
    norm = mcolors.BoundaryNorm(levels, cmap.N)


    pcm = ax.pcolormesh(
        lons_sorted, lats_sorted, data_final,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading='auto',
        zorder=1
    )


    # gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
    #                   color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=5)
    # gl.xlocator = mticker.FixedLocator(np.arange(140, 221, 10))  # 10度经度间隔
    # gl.ylocator = mticker.FixedLocator(np.arange(50, 81, 5))  # 5度纬度间隔
    # gl.top_labels = False
    # gl.right_labels = False
    # gl.xlabel_style = {'size': 10}
    # gl.ylabel_style = {'size': 10}

    # 添加分段色标
    cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.08, shrink=0.75, ticks=levels)
    cbar.set_label(f"{flux_info['name']} Anomaly ({flux_info['unit']})", fontsize=12)
    cbar.ax.tick_params(labelsize=10)


    # valid_data = data_final[~np.isnan(data_final)]
    # if valid_data.size > 0:
    #     stats_text = f'Mean: {np.mean(valid_data):.2f}\nMax: {np.max(valid_data):.2f}\nMin: {np.min(valid_data):.2f}'
    #     ax.text(0.02, 0.05, stats_text, transform=ax.transAxes, fontsize=9,
    #             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7), zorder=6)


    title = f"{flux_info['name']} Anomaly\n{season} {period} Sector: Chukchi Sea"
    ax.set_title(title, fontsize=14, pad=20)


    out_png = os.path.join(out_dir, f"{flux_type}_Anomaly_{season}_Sector.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {out_png}")

print("\n🎉 所有辐射分量扇区图绘制完成！")