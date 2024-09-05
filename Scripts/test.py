import h5netcdf
import xarray as xr

file_path = "D:/CloudDetection/images/162/clear_labels.nc"

# 尝试 netcdf4 引擎
try:
    ds = xr.open_dataset(file_path, engine="netcdf4")
    print(ds)
except Exception as e:
    print(f"Error with netcdf4: {e}")

# 如果上面的尝试失败，可以再尝试 h5netcdf 引擎
try:
    ds = xr.open_dataset(file_path, engine="h5netcdf")
    print(ds)
except Exception as e:
    print(f"Error with h5netcdf: {e}")
