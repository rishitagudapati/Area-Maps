#!/usr/bin/env python3
"""
map_fsa.py - Standalone script to generate maps for a specific FSA

This script extracts data for a single Forward Sortation Area (FSA) from an OSM file
and generates a map visualization showing roads and points of interest. It is much faster 
than processing the entire province.

Usage:
    python map_fsa.py FSA_CODE  [--output OUTPUT_PATH] 
                                [--osm OSM_FILE] 
                                [--boundaries BOUNDARIES_FILE] 
                                [--poi_csv POI_CSV_FILE]
                                [--id_field ID_FIELD] 
                                [--poi_types POI_TYPES]
                                [--base_map BASE_MAP] 
                                [--buffer BUFFER_KM]

Arguments:
    FSA_CODE              The Forward Sortation Area code (e.g., M6M, L5V)
    --output              Optional output path for the map image (default: ../plots/FSA_CODE_map.png)
    --osm                 Path to the OSM file (default: ../data/raw/ontario-latest.osm.pbf)
    --boundaries          Path to the FSA boundaries file (default: ../data/ontario_fsas.gpkg)
    --poi_csv               Path to a custom POI CSV file with latitude and longitude columns (optional)
    --id_field              Field in the boundaries file that contains area IDs (default: FSA)
    --poi_types             Comma-separated list of POI types to extract
                            (default: amenity:charging_station,amenity:fuel,amenity:parking,shop:mall,shop:car)
    --base_map              Base map provider (default: CartoDB.Voyager, options: CartoDB.Positron, 
                            CartoDB.DarkMatter, OpenStreetMap.Mapnik, Stamen.Terrain)
    --buffer                Buffer distance in km around the area boundary (default: 1.0)

Examples:
    python map_fsa.py M5S
    python map_fsa.py K1P --poi_types amenity:cafe,amenity:restaurant,shop:bakery,shop:supermarket
"""

import os
import sys
import argparse
import time
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import pyrosm
import contextily as ctx
import numpy as np
import pandas as pd
from shapely.geometry import Point
import warnings
import math
import matplotlib.ticker as mticker
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Earth's radius in meters, used for Web Mercator projection conversions
R = 6378137

def mercator_to_lon(x):
    """
    Convert Web Mercator x-coordinate to longitude in degrees.
    
    Parameters:
    -----------
    x : float
        x-coordinate in Web Mercator projection (EPSG:3857)
        
    Returns:
    --------
    float
        Longitude in degrees (EPSG:4326)
    """
    return (x * 180) / (math.pi * R)

def mercator_to_lat(y):
    """
    Convert Web Mercator y-coordinate to latitude in degrees.
    
    Parameters:
    -----------
    y : float
        y-coordinate in Web Mercator projection (EPSG:3857)
        
    Returns:
    --------
    float
        Latitude in degrees (EPSG:4326)
    """
    return (2 * math.atan(math.exp(y / R)) - math.pi/2) * 180 / math.pi

def parse_arguments():
    """
    Parse command line arguments for the FSA map generator.
    
    Returns:
    --------
    argparse.Namespace
        Parsed command line arguments with the following attributes:
        - fsa_code: The Forward Sortation Area code (uppercase)
        - output: Path to save the output map image
        - osm: Path to the OSM file
        - boundaries: Path to the FSA boundaries file
        - poi_csv: Path to the custom POI CSV file
        - id_field: Field in the boundaries file containing area identifiers
        - poi_types: Comma-separated list of POI types to extract
        - base_map: Base map provider to use
        - buffer: Buffer distance in km around the area boundary
    """
    parser = argparse.ArgumentParser(
        description='Generate a map for a specific FSA',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Define the base directory relative to this script
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Required argument
    parser.add_argument('fsa_code', metavar='FSA_CODE', type=str,
                        help='The Forward Sortation Area code (e.g., M5S, K1P)')
    
    # Optional arguments with defaults
    parser.add_argument('--output', type=str, 
                        help='Output path for the map image (default: ./FSA_CODE_map.png)')
    parser.add_argument('--osm', type=str, 
                        default=os.path.join(base_dir, 'data', 'raw', 'ontario-latest.osm.pbf'),
                        help=f'Path to the OSM file (default: ../data/raw/ontario-latest.osm.pbf)')
    parser.add_argument('--boundaries', type=str, 
                        default=os.path.join(base_dir, 'data', 'ontario_fsas.gpkg'),
                        help=f'Path to the FSA boundaries file (default: ../data/ontario_fsas.gpkg)')
    parser.add_argument('--poi_csv', type=str,

                        help='Path to a custom POI CSV file with latitude and longitude columns (optional)')

    parser.add_argument('--id_field', type=str, default='FSA',

                        help='Field in the boundaries file that contains area IDs (default: FSA)')

    parser.add_argument('--poi_types', type=str,

                        default='amenity:charging_station,amenity:fuel,amenity:parking,shop:mall,shop:car',

                        help='Comma-separated list of POI types to extract (default includes EV stations, gas, parking, etc.)')

    parser.add_argument('--base_map', type=str, default='CartoDB.Voyager',

                        help='Base map provider (default: CartoDB.Voyager)')

    parser.add_argument('--buffer', type=float, default=1.0,

                        help='Buffer distance in km around the area boundary (default: 1.0)')
    
    args = parser.parse_args()
    
    # Standardize FSA code to uppercase
    args.fsa_code = args.fsa_code.upper()
    
    # Set default output path if not provided
    if not args.output:
        # Use ../plots as the default directory
        maps_dir = os.path.join(base_dir, 'plots')
        args.output = os.path.join(maps_dir, f"{args.fsa_code.upper()}_map.png")
    
    return args

def check_files_exist(osm_file, boundaries_file, poi_csv_file=None):
    """
    Check if required input files exist and exit with error message if any are missing.
    
    Parameters:
    -----------
    osm_file : str
        Path to the OpenStreetMap file (.osm.pbf)
    boundaries_file : str
        Path to the FSA boundaries file (.gpkg)
    poi_csv_file : str, optional
        Path to the custom POI CSV file
        
    Returns:
    --------
    None
        Function exits the program with error code 1 if any required files are missing
    """
    missing_files = []
    
    # Try to resolve absolute paths for better error messages
    osm_abs_path = os.path.abspath(osm_file)
    boundaries_abs_path = os.path.abspath(boundaries_file)
    
    if not os.path.exists(osm_file):
        missing_files.append(f"OSM file: {osm_file} (absolute path: {osm_abs_path})")
    
    if not os.path.exists(boundaries_file):
        missing_files.append(f"FSA boundaries file: {boundaries_file} (absolute path: {boundaries_abs_path})")
    
    if poi_csv_file and not os.path.exists(poi_csv_file):
        poi_csv_abs_path = os.path.abspath(poi_csv_file)
        missing_files.append(f"POI CSV file: {poi_csv_file} (absolute path: {poi_csv_abs_path})")
    
    if missing_files:
        print("Error: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        
        # Print current working directory for debugging
        print(f"\nCurrent working directory: {os.getcwd()}")
        print("Please check that the file paths are correct.")
        sys.exit(1)

def load_fsa_boundary(boundaries_file, fsa_code):
    """
    Load the boundary polygon for a specific FSA from a GeoPackage file.
    
    Parameters:
    -----------
    boundaries_file : str
        Path to the FSA boundaries GeoPackage file
    fsa_code : str
        The Forward Sortation Area code to load (e.g., 'M5S', 'K1P')
        
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame containing the boundary polygon for the specified FSA
        
    Raises:
    -------
    SystemExit
        If the FSA code is not found in the boundaries file or if there's an error loading the file
    """
    try:
        print(f"\nLoading FSA boundaries from {boundaries_file}...")
        all_fsas = gpd.read_file(boundaries_file)
        
        # Filter to the specific FSA
        fsa_boundary = all_fsas[all_fsas['FSA'] == fsa_code]
        
        if len(fsa_boundary) == 0:
            print(f"Error: FSA code '{fsa_code}' not found in the boundaries file.")
            print(f"Available FSA codes include: {', '.join(sorted(all_fsas['FSA'].unique()[:10]))}...")
            sys.exit(1)
        
        print(f"Successfully loaded boundary for FSA {fsa_code}")
        return fsa_boundary
    
    except Exception as e:
        print(f"Error loading FSA boundary: {e}")
        sys.exit(1)

def get_bounding_box(geometry, buffer_km=1.0):
    """
    Get the bounding box of a geometry with an optional buffer in kilometers.
    
    Parameters:
    -----------
    geometry : geopandas.GeoSeries
        The geometry to get the bounding box for
    buffer_km : float, optional
        Buffer distance in kilometers to add around the geometry (default: 1.0)
        
    Returns:
    --------
    numpy.ndarray
        Bounding box coordinates as (minx, miny, maxx, maxy)
    """
    # Convert to Statistics Canada Lambert projection (EPSG:3347) for accurate distance calculations
    utm_crs = 'EPSG:3347'  
    geom_utm = geometry.to_crs(utm_crs)
    
    # Buffer by specified kilometers (converted to meters)
    buffered_geom = geom_utm.buffer(buffer_km * 1000)
    
    # Convert back to original CRS
    buffered_geom = buffered_geom.to_crs(geometry.crs)
    
    # Get bounds
    bounds = buffered_geom.total_bounds
    return bounds

def load_custom_pois(poi_csv_file, area_boundary=None, fsa_code=None, crs='EPSG:4326'):
    """
    Load custom points of interest from a CSV file and convert to GeoDataFrame.
    
    Parameters:
    -----------
    poi_csv_file : str
        Path to the CSV file containing POI data
    area_boundary : geopandas.GeoDataFrame, optional
        Area boundary to clip/filter POIs by (default: None, which loads all POIs)
    fsa_code : str
        The Forward Sortation Area code to load (e.g., 'M6M', 'L5V')
    crs : str, optional
        Coordinate reference system to use for the GeoDataFrame (default: 'EPSG:4326')
        
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame containing POIs with Point geometries
        Returns an empty GeoDataFrame with the correct structure if no POIs are found
        or if there's an error loading the data
    """
    try:
        print(f"\nLoading custom POIs from {poi_csv_file}...")
        # Read CSV file
        poi_df = pd.read_csv(poi_csv_file)

        print(f"Loaded CSV with columns: {poi_df.columns.tolist()}")
        
        # Check for required columns (longitude and latitude)
        longitude_col = None
        latitude_col = None

        # Try to identify longitude and latitude columns
        for col in poi_df.columns:
            col_lower = col.lower()
            if 'lon' in col_lower:
                longitude_col = col
            elif 'lng' in col_lower:
                longitude_col = col
            elif 'x' == col_lower:
                longitude_col = col
            elif 'lat' in col_lower:
                latitude_col = col
            elif 'y' == col_lower:
                latitude_col = col
        
        if not longitude_col or not latitude_col:
            print("Error: Could not automatically identify longitude and latitude columns.")
            print("Please rename your columns to include 'lon'/'lng' and 'lat', or use 'x' and 'y'.")
            sys.exit(1)

        print(f"Using {longitude_col} as longitude and {latitude_col} as latitude")
        
        # Create geometry from longitude and latitude columns
        geometry = [Point(xy) for xy in zip(poi_df[longitude_col], poi_df[latitude_col])]
        
        # Create GeoDataFrame
        poi_gdf = gpd.GeoDataFrame(poi_df, geometry=geometry, crs=crs)

        print(f"Successfully loaded {len(poi_gdf)} POIs from CSV")

        # Filter to area if specified
        if area_boundary is not None:
            # Ensure CRS matches
            if poi_gdf.crs != area_boundary.crs:
                poi_gdf = poi_gdf.to_crs(area_boundary.crs)
            
            # Clip to boundary
            try:
                area_poi = gpd.clip(poi_gdf, area_boundary)
                print(f"Filtered to {len(area_poi)} POIs within area boundary")

                # Add id column if not present (required for compatibility with other POIs)
                if 'id' not in area_poi.columns:
                    area_poi['id'] = area_poi['ID'].astype(str)

                return area_poi
            
            except Exception as e:
                print(f"Error clipping POIs to area boundary: {e}")
                print("Falling back to spatial join...")
                # Fallback to spatial join if clipping fails
                area_poi = gpd.sjoin(poi_gdf, area_boundary, predicate='intersects')
                print(f"Filtered to {len(area_poi)} POIs within area boundary")

                # Add id column if not present (required for compatibility with other POIs)
                if 'id' not in area_poi.columns:
                    area_poi['id'] = area_poi['ID'].astype(str)

                return area_poi
            
        # Filter by FSA if specified
        elif fsa_code:
            fsa_poi = poi_df[poi_df['FSA'] == fsa_code]
            print(f"Filtered to {len(fsa_poi)} POIs within {fsa_code} boundary")

            # Add id column if not present (required for compatibility with other POIs)
            if 'id' not in fsa_poi.columns:
                fsa_poi['id'] = fsa_poi['ID'].astype(str)

            return fsa_poi

        else:
            print(f"Loaded {len(poi_df)} POIs without filtering")
        

        # Add id column if not present (required for compatibility with other POIs)
        if 'id' not in poi_df.columns:
            poi_df['id'] = poi_df['ID'].astype(str)

        return poi_gdf
    
    except Exception as e:
        print(f"Error loading custom POIs: {e}")
        # Return empty GeoDataFrame with correct structure
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=crs)
    
def parse_poi_types(poi_types_str):
    """
    Parse the POI types string into individual type-specific filters in a format suitable for pyrosm.
    
    Expects input in the format: "category1:[value1,value2]+category2:[value3,value4]" or just "category"
    For example: 
    - "amenity:[cafe,pub,restaurant]+shop:[bakery,supermarket]" (specific values)
    - "amenity+shop:[bakery,supermarket]" (all amenities, specific shops)
    - "amenity" (all amenities)
    
    Parameters:
    -----------
    poi_types_str : str
        Plus-separated list of POI types in the grouped format
        
    Returns:
    --------
    dict
        Dictionary of POI filters suitable for pyrosm.get_pois
    """
    poi_filters = {}
    
    # Split by '+' to get each category group
    for category_group in poi_types_str.split('+'):
        # Skip empty entries
        if not category_group.strip():
            continue
            
        # Check if the category has specified values
        if ':' in category_group:
            # Extract the category and values
            try:
                # Split at the first ':' to separate category from values
                category, values_str = category_group.split(':', 1)
                category = category.strip()
                
                # Extract values inside the square brackets
                values = values_str.strip()
                if values.startswith('[') and values.endswith(']'):
                    values = values[1:-1]  # Remove the brackets
                    
                    # Split by comma to get individual values
                    for value in values.split(','):
                        value = value.strip()
                        if value:
                            # Create a filter entry for each value
                            filter_key = f"{category}:{value}"
                            poi_filters[filter_key] = {category: [value]}
                else:
                    print(f"Warning: Expected '[value1,value2,...]' format for {category}, got {values_str}")
                    # Try to use the single value anyway
                    if values:
                        filter_key = f"{category}:{values}"
                        poi_filters[filter_key] = {category: [values]}
            except ValueError:
                print(f"Warning: Could not parse category group '{category_group}', skipping")
        else:
            # If just a category is provided without specific values, include all values for that category
            category = category_group.strip()
            # Use a special key for this case
            filter_key = f"{category}:all"
            # For pyrosm, using None or an empty list means "get all values for this key"
            poi_filters[filter_key] = {category: None}
            print(f"Will extract all POIs with the key '{category}'")
    
    return poi_filters

def extract_networks_and_pois(osm, boundary, poi_types_str, custom_pois_gdf=None):
    """
    Extract road networks and points of interest from OSM data within the area boundary.
    
    Parameters:
    -----------
    osm : pyrosm.OSM
        PyROSM object containing loaded OpenStreetMap data
    boundary : geopandas.GeoDataFrame
        GeoDataFrame containing the area boundary polygon
    poi_types_str : str
        Comma-separated list of POI types to extract
    custom_pois_gdf : geopandas.GeoDataFrame, optional
        GeoDataFrame containing custom POIs (default: None)
        
    Returns:
    --------
    tuple
        (area_roads, major_roads, pois) where:
        - area_roads: GeoDataFrame containing all road segments within the area
        - major_roads: GeoDataFrame containing only major road segments
        - pois: Dictionary of GeoDataFrames containing points of interest by category
    """
    print("\nExtracting road network... (this may take a while)")
    road_network = osm.get_network(network_type="driving")
    
    print("Classifying roads...")
    highway_types = {
        'motorway': 'major',
        'trunk': 'major',
        'primary': 'major',
        'secondary': 'major',
        'tertiary': 'minor',
        'residential': 'minor',
        'unclassified': 'minor',
        'service': 'service',
    }
    road_network['road_type'] = road_network['highway'].map(highway_types).fillna('other')
    
    # Clip to boundary
    print("Clipping road network to area boundary...")
    try:
        area_roads = gpd.clip(road_network, boundary)
        major_roads = area_roads[area_roads['road_type'] == 'major']
    except Exception as e:
        print(f"Error clipping road network: {e}")
        print("Using spatial join as fallback method...")
        # Fallback to spatial join if clipping fails
        area_roads = gpd.sjoin(road_network, boundary, predicate='intersects')
        major_roads = area_roads[area_roads['road_type'] == 'major']
    
    print(f"Extracted {len(area_roads)} road segments ({len(major_roads)} major roads)")
    
    print("\nExtracting POIs...")
    
    # Initialize POIs dictionary
    pois = {}
    
    # Add custom POIs if provided
    if custom_pois_gdf is not None and not custom_pois_gdf.empty:
        print(f"Adding {len(custom_pois_gdf)} custom POIs from CSV")
        pois["custom"] = custom_pois_gdf
    
    # Parse and extract OSM POIs
    poi_filters = parse_poi_types(poi_types_str)
    
    for filter_key, custom_filter in poi_filters.items():
        print(f"  Extracting POIs for {filter_key}...")
        try:
            poi_gdf = osm.get_pois(custom_filter=custom_filter)
            if poi_gdf is not None and not poi_gdf.empty:
                # Clip to boundary
                try:
                    area_poi = gpd.clip(poi_gdf, boundary)
                except Exception as e:
                    print(f"    Error clipping {filter_key}: {e}")
                    print("    Using spatial join as fallback method...")
                    # Fallback to spatial join if clipping fails
                    area_poi = gpd.sjoin(poi_gdf, boundary, predicate='intersects')
                
                pois[filter_key] = area_poi
                print(f"    Found {len(area_poi)} {filter_key} POIs")

            else:
                print(f"    No {filter_key} POIs found")
                pois[filter_key] = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=road_network.crs)
        except Exception as e:
            print(f"    Error extracting {filter_key}: {e}")
            pois[filter_key] = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=road_network.crs)
    
    return area_roads, major_roads, pois

def get_basemap_provider(provider_name):
    """
    Get a contextily basemap provider based on name.
    
    Parameters:
    -----------
    provider_name : str
        Name of the basemap provider
        
    Returns:
    --------
    contextily.providers
        Basemap provider object
    """
    provider_parts = provider_name.split('.')
    
    if len(provider_parts) != 2:
        print(f"Warning: Invalid basemap provider format '{provider_name}'. Using CartoDB.Positron as fallback.")
        return ctx.providers.CartoDB.Positron
    
    provider_name, style_name = provider_parts
    
    # Get the provider
    try:
        provider = getattr(ctx.providers, provider_name)
        # Get the style
        try:
            style = getattr(provider, style_name)
            return style
        except AttributeError:
            print(f"Warning: Style '{style_name}' not found for provider '{provider_name}'. Using CartoDB.Positron as fallback.")
            return ctx.providers.CartoDB.Positron
    except AttributeError:
        print(f"Warning: Provider '{provider_name}' not found. Using CartoDB.Positron as fallback.")
        return ctx.providers.CartoDB.Positron

def draw_icon_legend(ax, entries, start_y=0.95, y_step=0.05, icon_zoom=0.5):
    """
    Draw a custom legend using icons and text outside the plot area.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Axes to draw the legend on
    entries : list of tuples
        Each entry should be (label, icon_path)
    start_y : float
        Y-coordinate in axes fraction to start drawing
    y_step : float
        Vertical spacing between legend items
    icon_zoom : float
        Zoom factor for the icons
    """
    for i, (label, icon_path) in enumerate(entries):
        y = start_y - i * y_step
        try:
            img = mpimg.imread(icon_path)
            imagebox = OffsetImage(img, zoom=icon_zoom)
            ab = AnnotationBbox(imagebox, (1.01, y),
                                xycoords='axes fraction',
                                frameon=False)
            ax.add_artist(ab)

            ax.text(1.05, y, label, transform=ax.transAxes,
                    fontsize=9, va='center', ha='left')
        except Exception as e:
            print(f"Could not render legend icon for {label}: {e}")

def create_map(boundary, roads, pois, fsa_code, output_path, basemap_provider_name='CartoDB.Voyager'):
    """
    Create and save a map visualization of the transportation infrastructure for a specified area.
    
    This function creates a map showing the area boundary, different road types (major, minor, service),
    and points of interest with OSM-style icons when available. It adds a basemap for geographic context 
    and saves the map to a PNG file.
    
    Parameters:
    -----------
    boundary : geopandas.GeoDataFrame
        GeoDataFrame containing the area boundary polygon
    roads : geopandas.GeoDataFrame
        GeoDataFrame containing road segments within the area
    pois : dict
        Dictionary of GeoDataFrames containing points of interest by category
    fsa_code : str
        The Forward Sortation Area code for the map title
    output_path : str
        Path to save the output map image
    basemap_provider_name : str, optional
        Name of the basemap provider to use (default: 'CartoDB.Voyager')
        
    Returns:
    --------
    bool
        True if the map was created and saved successfully, False otherwise
    """
    try:
        import matplotlib.image as mpimg
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        
        print(f"\nCreating map for {fsa_code}...")
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Convert everything to Web Mercator (EPSG:3857) for basemap compatibility
        boundary_web_merc = boundary.to_crs('EPSG:3857')
        
        # Plot the FSA boundary
        boundary_web_merc.plot(ax=ax, color='none', edgecolor='black', linewidth=2)
        
        # Plot the road network by type
        if not roads.empty:
            # Convert roads to Web Mercator
            roads_web_merc = roads.to_crs('EPSG:3857')
            
            roads_major = roads_web_merc[roads_web_merc['road_type'] == 'major']
            roads_minor = roads_web_merc[roads_web_merc['road_type'] == 'minor']
            roads_service = roads_web_merc[roads_web_merc['road_type'] == 'service']
            
            if not roads_major.empty:
                roads_major.plot(ax=ax, color='red', linewidth=2, label='Major roads')
            if not roads_minor.empty:
                roads_minor.plot(ax=ax, color='orange', linewidth=1, label='Minor roads')
            if not roads_service.empty:
                roads_service.plot(ax=ax, color='grey', linewidth=0.5, label='Service roads')
        
        # Define a color palette for POIs when icons are not available
        poi_colors = [
            'lime', 'blue', 'purple', 'hotpink', 'cyan', 
            'darkgreen', 'darkblue', 'darkred', 'darkorange', 'magenta'
        ]

        ICON_NAME_MAP = {
            "pub": "bar",
            "fuel": "gas-station",
            "parking_entrance": "parking",
            "clinic": "hospital-building",
        #    "veterinary": "veterinary",
            "waste_basket": "trash",
            "recycling": "recycle",
            "vending_machine": "vending-machine",
            "supermarket": "grocery",
        #    "coffee": "coffee",
            "alcohol": "liquor",
            "general": "market",
            "convenience": "conveniencestore",
            "department_store": "departmentstore",
        #    "mall": "mall",
            "musical_instrument": "music",
            "games": "game",
        #    "video": "video",
            "video_games": "videogames",
            "books": "book-store",
        #    "stationery": "stationery",
            "ticket": "ticket_office2.png",
        #    "music": "music",
        #    "bench": "bench",
            "toilets": "toilets_inclusive",
            "cannabis": "marijuana",
            "pet": "pets",

        }
        
        # Function to fetch and cache icons from mapicons.mapsmarker.com
        def get_osm_icon(category, subcategory):

            icon_key = ICON_NAME_MAP.get(subcategory, subcategory)  # remap if needed
            icon_file = f"{icon_key}.png"

            icon_cache_path = Path("icon_cache")

            # Search all subdirectories for a matching icon file
            matching_icons = list(icon_cache_path.rglob(icon_file))

            if matching_icons:
                return str(matching_icons[0])  # use the first match found
            else:
                print(f"    Warning: Icon not found for {category}:{subcategory} → {icon_file}")
                return None
        
        # Track used colors for POIs without icons
        color_index = 0
        
        # Plot each POI category
        for poi_name, poi_gdf in pois.items():
            if poi_gdf is not None and not poi_gdf.empty:
                # Convert POIs to Web Mercator
                poi_gdf_web_merc = poi_gdf.to_crs('EPSG:3857')
                
                # Format the POI name for display
                if ':' in poi_name:
                    category, subcategory = poi_name.split(':', 1)
                    display_name = subcategory.replace('_', ' ')  # Use the subcategory name without the category
    
                    # Try to get an OSM icon
                    icon_path = get_osm_icon(category, subcategory)
                else:
                    display_name = poi_name.replace('_', ' ')  # Keep lowercase, just remove underscores
                    icon_path = None  # No icon for generic categories
                
                # The special case for "all" values of a category
                if display_name == "all":
                    display_name = f"{category}"  # e.g., "amenity" 

            #    display_name = display_name.strip().replace('_', ' ')

                # Determine if we'll use an icon or a colored dot
                use_icon = icon_path is not None and os.path.exists(icon_path)
                
                # Get color for this POI type (for dots or when icons not available)
                poi_color = poi_colors[color_index % len(poi_colors)]
                color_index += 1
                
                # Always plot POIs with a semi-transparent color for visibility

                # Plot polygons (e.g., schools, parking lots) with fill color
                # and points (e.g., benches, ATMs) with dots or icons

                # 1. Plot polygons (areas)
                if poi_gdf_web_merc.geom_type.isin(['Polygon', 'MultiPolygon']).any():
                    polygons = poi_gdf_web_merc[poi_gdf_web_merc.geom_type.isin(['Polygon', 'MultiPolygon'])]
                    polygons.plot(
                        ax=ax,
                        facecolor=poi_color,
                        edgecolor='black',
                        linewidth=0.5,
                        alpha=0.4,
                        label=None
                    )

                # 2. Plot points (dots)
                if poi_gdf_web_merc.geom_type.isin(['Point']).any():
                    points = poi_gdf_web_merc[poi_gdf_web_merc.geom_type == 'Point']
                    points.plot(
                        ax=ax, 
                        color=poi_color, 
                        markersize=50 if not use_icon else 30,
                        alpha=0.7,
                        label=None
                    )

                    # Load the logo image
                    credit_img = mpimg.imread("icon_cache/mapicons-credit.gif")
                    imagebox = OffsetImage(credit_img, zoom=0.5)  # adjust zoom as needed

                    # Place the logo just under the legend
                    # Coordinates are in axes fraction (0 to 1), so tweak (1.01, 0.05) as needed
                    ab = AnnotationBbox(
                        imagebox,
                        (1.01, 0.05),  # slightly right of plot, low on the side
                        xycoords='axes fraction',
                        frameon=False
                    )
                    ax.add_artist(ab)
                    ax.text(1.01, 0.01, "Icons by Map Icons Collection", transform=ax.transAxes,
                                fontsize=8, ha='left', va='bottom')

                num_polygons = poi_gdf_web_merc.geom_type.isin(['Polygon', 'MultiPolygon']).sum()
                num_points = poi_gdf_web_merc.geom_type.isin(['Point']).sum()
                print(f"    Plotted {num_polygons} {display_name} polygons and {num_points} points")


        # Add basemap
        try:
            basemap_provider = get_basemap_provider(basemap_provider_name)
            ctx.add_basemap(
                ax, 
                source=basemap_provider,
                zoom=12,
                attribution_size=8
            )
            print(f"Added {basemap_provider_name} basemap.")
        except Exception as e:
            print(f"Could not add basemap: {e}")
        
        # Convert tick labels from Web Mercator to lat/lon
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{mercator_to_lon(x):.2f}°"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"{mercator_to_lat(y):.2f}°"))
        
        # Add title and labels
        plt.title(f'Points of Interest - {fsa_code}', fontsize=15)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Collect entries with found icons
        legend_entries = []
        for poi_name, poi_gdf in pois.items():
            if poi_gdf is not None and not poi_gdf.empty:
                if ':' in poi_name:
                    category, subcategory = poi_name.split(':', 1)
                    display_name = subcategory.replace('_', ' ')
                    icon_path = get_osm_icon(category, subcategory)
                    if icon_path and os.path.exists(icon_path):
                        legend_entries.append((display_name, icon_path))

        # Draw icon-based legend
        draw_icon_legend(ax, legend_entries)

        # Save the map
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nMap saved to {output_path}")
        return True
    
    except Exception as e:
        print(f"Error creating map: {e}")
        return False

def main():
    """
    Main function to process a single area and create a map.
    
    This function orchestrates the entire map generation process:
    1. Parses command line arguments
    2. Checks if required input files exist
    3. Loads the area boundary from the GeoPackage file
    4. Loads custom POIs if specified
    5. Calculates a bounding box with buffer for the area
    6. Loads OpenStreetMap data for the bounding box
    7. Extracts road networks and points of interest
    8. Creates and saves the map visualization
    9. Reports timing information
    
    The function doesn't take any parameters or return any values.
    It exits with code 1 if any step fails.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Start timing
    start_time = time.time()
    print(f"\nScript started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if required files exist
    check_files_exist(args.osm, args.boundaries, args.poi_csv)
    
    # Load FSA boundary
    fsa_boundary = load_fsa_boundary(args.boundaries, args.fsa_code)
    
    # Load custom POIs if specified
    custom_pois = None
    if args.poi_csv:
        custom_pois = load_custom_pois(args.poi_csv, fsa_boundary)
    
    # Get bounding box with buffer (default: 1km)
    bbox = get_bounding_box(fsa_boundary.geometry, buffer_km=args.buffer)
    print(f"\nUsing bounding box with {args.buffer}km buffer: {bbox}")
    
    # Load OSM data for the bounding box
    print(f"Loading OSM data from {args.osm}...")
    try:
        # Convert NumPy array to list for PyROSM
        bbox_list = bbox.tolist()
        print(f"Using bounding box (as list): {bbox_list}")
        osm = pyrosm.OSM(args.osm, bounding_box=bbox_list)
        print("OSM data loaded successfully with bounding box")
    except Exception as e:
        print(f"Error loading OSM data with bounding box: {e}")
        print("Trying to load without bounding box (this may take longer)...")
        try:
            osm = pyrosm.OSM(args.osm)
            print("OSM data loaded successfully without bounding box")
        except Exception as e:
            print(f"Error loading OSM data: {e}")
            sys.exit(1)
    
    # Extract networks and POIs
    roads, major_roads, pois = extract_networks_and_pois(osm, fsa_boundary, args.poi_types, custom_pois)
    
    # Create map
    success = create_map(fsa_boundary, roads, pois, args.fsa_code, args.output, args.base_map)
    
    # Report timing
    elapsed_time = time.time() - start_time
    elapsed_min = elapsed_time / 60
    print(f"Process completed in {elapsed_min:.2f} minutes")
    
    if success:
        print(f"Map for area {args.fsa_code} created successfully!")
    else:
        print(f"Failed to create map for area {args.fsa_code}")
        sys.exit(1)

if __name__ == "__main__":
    main()
