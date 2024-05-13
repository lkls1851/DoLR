import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

# Path to the GeoTIFF file
tiff_path = 'path_to_merged.tif'

# Read the GeoTIFF file
with rasterio.open(tiff_path) as src:
    # Convert the raster to a numpy array
    image = src.read(1)

    # Extract vector shapes from the raster
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(
            shapes(image, mask=None, transform=src.transform)))

    # Create a GeoDataFrame from the vector shapes
    gdf = gpd.GeoDataFrame.from_features(list(results), crs=src.crs)

# Path to save the Shapefile
output_shapefile = 'merged.shp'

# Save the GeoDataFrame to a Shapefile
gdf.to_file(output_shapefile)

print("Shapefile saved successfully.")
