import pandas as pd
import numpy as np
from matplotlib import image, pyplot as plt
from matplotlib import image as img

# 1) loading and filtering dataset
ds = pd.read_csv("../datasets/NYC_POIs/pois_all_info", index_col=False, sep="\t", low_memory=False)
ny_municipality = pd.read_csv("./NYC_POIs/ny_municipality_pois_id.csv", squeeze=True)
ds = pd.merge(ds, ny_municipality, on="@id")

# 2) counting missing values
missing_values =np.zeros(len(ds.columns))
for i in range(len(ds.columns)):
    missing_values[i] = ds[ds.columns[i]].isna().sum()
print(missing_values)

# 3) for each category plot a histogram showing types distribution
amenity = ds["amenity_name"].value_counts().sort_values(ascending=False)
shop = ds["shop"].value_counts().sort_values(ascending=False)
public_transport = ds["public_transport"].value_counts().sort_values(ascending=False)
highway = ds["highway"].value_counts().sort_values(ascending=False)
"""
plt.figure(1)
amenity.iloc[:10].plot.bar()
plt.title("amenity")

plt.figure(2)
shop.iloc[:10].plot.bar()
plt.title("shop")

plt.figure(3)
public_transport.iloc[:10].plot.bar()
plt.title("public transport")

plt.figure(4)
highway.iloc[:10].plot.bar()
plt.title("highway")

plt.show()"""

# 4) scatter plot in NY map image each point, different color for each category
get_category_samples = lambda category_name: ds.loc[ds[category_name].notnull(), ["@lat", "@lon"]]

amenity_samples = get_category_samples("amenity_name")
shop_samples = get_category_samples("shop")
public_transport = get_category_samples("public_transport")
highway_samples = get_category_samples("highway")

image_map = img.imread("./NYC_POIs/New_York_City_Map.PNG")

plt.imshow(image_map)
plt.scatter(amenity_samples["@lon"], amenity_samples["@lat"])
plt.show()