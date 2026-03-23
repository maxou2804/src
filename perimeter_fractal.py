import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import numpy as np
import porespy as ps

# load csv
city='Bangkok'
df = pd.read_csv(f"perimeter_fractal_{city}_500.csv")
city_df=pd.read_csv(f'test_{city}.csv')
df_year=df[df['year']==1985]

rows=df_year['row'].values
cols=df_year['col'].values

angle=df_year['sector_angle_rad'].values
distance=df_year['distance_pixels'].values

row_min, row_max = rows.min(), rows.max()
col_min, col_max = cols.min(), cols.max()

rows_scaled = ((rows - row_min) / (row_max - row_min) * 999).astype(int)
cols_scaled = ((cols - col_min) / (col_max - col_min) * 999).astype(int)


# Create grid
grid = np.zeros((1000, 1000), dtype=np.uint8)

# Set perimeter pixels
grid[rows_scaled, cols_scaled] = 1


plt.figure()
plt.imshow(grid,alpha=0.8)
plt.show()

fig = plt.figure(figsize=(5, 8))
ax = fig.add_subplot(projection='polar')                
ax.plot(angle, distance)
plt.title(f'Perimeter of {city} ')
ax.grid(True)
plt.show()


data=ps.metrics.boxcount(grid)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_xlabel("box edge length")
ax1.set_ylabel("number of boxes spanning phases")
ax2.set_xlabel("box edge length")
ax2.set_ylabel("slope")
ax2.set_xscale("log")
ax1.plot(data.size, data.count, "-o")
ax2.plot(data.size, data.slope, "-o")


df = pd.read_csv("perimeter_fractal_Bangkok_500.csv")

df_year=df[df['year']==1985]

rows=df_year['row'].values
cols=df_year['col'].values


row_min, row_max = rows.min(), rows.max()
col_min, col_max = cols.min(), cols.max()

rows_scaled = ((rows - row_min) / (row_max - row_min) * 999).astype(int)
cols_scaled = ((cols - col_min) / (col_max - col_min) * 999).astype(int)


# Create grid
grid = np.zeros((1000, 1000), dtype=np.uint8)

# Set perimeter pixels
grid[rows_scaled, cols_scaled] = 1




data=ps.metrics.boxcount(grid)





fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_xlabel("box edge length")
ax1.set_ylabel("number of boxes spanning phases")
ax2.set_xlabel("box edge length")
ax2.set_ylabel("slope")
ax2.set_xscale("log")
ax1.plot(data.size, data.count, "-o")
ax2.plot(data.size, data.slope, "-o")
plt.show()

