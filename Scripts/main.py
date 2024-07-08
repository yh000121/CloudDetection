import xarray as xr

import data_loading as dl



if __name__ == "__main__":
    # def path
    path = 'D:../images/S1'

    # combine all
    ds = xr.open_mfdataset(f'{path}/S*_radiance_an.nc', combine='by_coords')

    # 预处理数据
    all_features, rows, cols = dl.preprocess_data(ds)

    # 打印结果以检查
    print(f"All features shape: {all_features.shape}")
    print(f"Rows: {rows}, Columns: {cols}")