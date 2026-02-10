import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import rasterio
from rasterio.warp import transform as warp_transform
import matplotlib.ticker as mticker

plt.rcParams['font.family'] = 'Arial'

base_lcc_path = r"E:\2024seaice\2025\cloud\lcc_anomaly_seasonal"
save_dir = r"E:\2024seaice\2025\cloud\plot_results\lcc_2020_2024_sector"
os.makedirs(save_dir, exist_ok=True)


lon_min, lon_max = 140, 220  # 扇区经度
lat_min, lat_max = 49, 77  # 扇区纬度


bounds = np.arange(-10, 11, 2)
cmap = plt.get_cmap('BrBG_r', len(bounds) - 1)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

seasons = ['JAS']
years = [2020, 2021, 2022, 2023, 2024]


def calculate_multi_year_mean(season):
    lcc_stack = []
    lons_ref, lats_ref = None, None

    for year in years:
        lcc_file = os.path.join(base_lcc_path, f"LCC_anomaly_{season}_{year}01.tif")
        if not os.path.exists(lcc_file):
            continue

        with rasterio.open(lcc_file) as src:
            lcc = src.read(1) * 100
            if lons_ref is None:
                transform = src.transform
                crs = src.crs
                width, height = src.width, src.height
                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                xs, ys = rasterio.transform.xy(transform, rows, cols)
                xs, ys = np.array(xs), np.array(ys)

                # 获取原始经纬度
                lons, lats = warp_transform(crs, {'init': 'EPSG:4326'}, xs.flatten(), ys.flatten())
                lons_ref = np.array(lons).reshape(xs.shape)
                lats_ref = np.array(lats).reshape(ys.shape)

            lcc_stack.append(lcc)

    if not lcc_stack:
        return None, None, None


    lcc_mean = np.nanmean(np.stack(lcc_stack), axis=0)


    lons_360 = np.where(lons_ref < 0, lons_ref + 360, lons_ref)


    lon_1d = lons_360[0, :]
    sort_idx = np.argsort(lon_1d)  # 获取从小到大的索引顺序

    lon_sorted_1d = lon_1d[sort_idx]
    lcc_sorted = lcc_mean[:, sort_idx]  # 对数据列进行重排
    lats_sorted = lats_ref[:, sort_idx]


    lons_sorted_2d = np.tile(lon_sorted_1d, (height, 1))


    mask = (lons_sorted_2d >= lon_min) & (lons_sorted_2d <= lon_max) & (lats_sorted >= lat_min) & (
                lats_sorted <= lat_max)
    lcc_final = np.where(mask, lcc_sorted, np.nan)

    return lcc_final, lons_sorted_2d, lats_sorted
# ========== 绘图函数 ==========
def plot_sector_mean():
    for season in seasons:
        print(f"📊 正在绘制 {season} 扇区图...")
        lcc_mean, lons_360, lats = calculate_multi_year_mean(season)
        if lcc_mean is None: continue

        fig = plt.figure(figsize=(10, 8))

        # 使用 NorthPolarStereo，中心经度设为扇区的正中心 (180°)
        # 这样扇区在图中是正对着观察者的
        proj = ccrs.NorthPolarStereo(central_longitude=180)
        ax = plt.axes(projection=proj)

        # 核心修改：设置地图显示范围为指定扇区
        # 注意：PlateCarree 坐标下，140-220 范围在绘图时会被自动处理
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        # 添加地理特征
        land = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='gray', facecolor='gray')
        ax.add_feature(land, zorder=4)
        ax.coastlines(resolution='10m', zorder=5, linewidth=0)
        ax.add_feature(cfeature.OCEAN, color='white', alpha=0.3, zorder=1)

        # 绘制扇区数据
        pcm = ax.pcolormesh(lons_360, lats, lcc_mean,
                            cmap=cmap, norm=norm, shading='nearest',
                            transform=ccrs.PlateCarree(), zorder=1)

        # 调整色标位置和大小
        cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', shrink=0.6, pad=0.08, extend='both')
        cbar.set_label("Low Cloud Cover Anomaly (%)", fontsize=11)


        # gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
        #                   color='gray', linestyle='--', linewidth=0.6, alpha=0.5, zorder=5)
        #
        # # # 设置经纬度刻度间隔
        # gl.xlocator = mticker.FixedLocator(np.arange(lon_min, lon_max + 1, 10))
        # gl.ylocator = mticker.FixedLocator(np.arange(lat_min, lat_max + 1, 5))
        #
        # gl.top_labels = False
        # gl.right_labels = False
        # gl.xlabel_style = {'size': 9}
        # gl.ylabel_style = {'size': 9}

        plt.title(f"2020-2024 {season} LCC Anomaly Mean\n(Chukchi Sea Sector: {lon_min}°-{lon_max}°E)",
                  fontsize=14, pad=15)

        save_path = os.path.join(save_dir, f"LCC_Sector_{season}_2020_2024.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存扇区图: {save_path}")


if __name__ == "__main__":
    plot_sector_mean()