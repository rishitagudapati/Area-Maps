"""
This module processes geographic data for Forward Sortation Areas (FSAs) for any Canadian province.

It loads FSA boundary shapefiles, filters for the specified province based on FSA prefixes,
standardizes the data format, and creates visualization maps with basemaps showing natural features.
The processed geodata is saved for further analysis and visualization purposes.

Usage:
    python draw_boundaries.py [province]

Where [province] can be:
    - Full province name (e.g., "Ontario", "British Columbia") 
    - Province abbreviation (e.g., "ON", "BC")

The processed data is saved to "../data/{province}_fsas.gpkg".
Visualizations are saved to "../plots".
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import os
import sys
import re
import contextily as ctx

# Define mappings for provinces to FSA prefixes
PROVINCE_TO_PREFIX_MAP = {
    # Full names (lowercase for case-insensitive matching)
    "alberta": ["T"],
    "british columbia": ["V"],
    "manitoba": ["R"],
    "new brunswick": ["E"],
    "newfoundland and labrador": ["A"],
    "nova scotia": ["B"],
    "ontario": ["K", "L", "M", "N", "P"],
    "prince edward island": ["C"],
    "quebec": ["G", "H", "J"],
    "saskatchewan": ["S"],
    "northwest territories": ["X"],
    "nunavut": ["X"],
    "yukon": ["Y"],
    
    # Standard abbreviations
    "ab": ["T"],
    "bc": ["V"],
    "mb": ["R"],
    "nb": ["E"],
    "nl": ["A"],
    "ns": ["B"],
    "on": ["K", "L", "M", "N", "P"],
    "pe": ["C"],
    "qc": ["G", "H", "J"],
    "sk": ["S"],
    "nt": ["X"],
    "nu": ["X"],
    "yt": ["Y"],
}

# Define clean display names for provinces
PROVINCE_DISPLAY_NAMES = {
    # Full names
    "alberta": "Alberta",
    "british columbia": "British Columbia",
    "manitoba": "Manitoba",
    "new brunswick": "New Brunswick",
    "newfoundland and labrador": "Newfoundland and Labrador",
    "nova scotia": "Nova Scotia",
    "ontario": "Ontario",
    "prince edward island": "Prince Edward Island",
    "quebec": "Quebec",
    "saskatchewan": "Saskatchewan",
    "northwest territories": "Northwest Territories",
    "nunavut": "Nunavut",
    "yukon": "Yukon",
    
    # Standard abbreviations
    "ab": "Alberta",
    "bc": "British Columbia",
    "mb": "Manitoba",
    "nb": "New Brunswick",
    "nl": "Newfoundland and Labrador",
    "ns": "Nova Scotia",
    "on": "Ontario",
    "pe": "Prince Edward Island",
    "qc": "Quebec",
    "sk": "Saskatchewan",
    "nt": "Northwest Territories",
    "nu": "Nunavut",
    "yt": "Yukon",
}

def normalize_province_input(province):
    """
    Normalize province input by converting to lowercase and handling special cases.
    """
    if province is None:
        return None
    
    # Convert to lowercase and remove extra spaces
    normalized = province.lower().strip()
    
    # Handle special cases like abbreviated forms with periods (e.g., "B.C." -> "bc")
    normalized = re.sub(r'\.', '', normalized)
    
    # Handle "PEI" and "NWT" special cases
    if normalized == "pei":
        normalized = "pe"

    elif normalized == "nwt":
        normalized = "nt"
    
    return normalized

def get_province_info(province_input):
    """
    Get province information from the input.
    
    Args:
        province_input: The province name or abbreviation
        
    Returns:
        tuple: (prefixes, display_name, slug)
        
    Raises:
        ValueError: If the province is not recognized
    """
    normalized = normalize_province_input(province_input)
    
    if normalized not in PROVINCE_TO_PREFIX_MAP:
        valid_options = list(PROVINCE_TO_PREFIX_MAP.keys())
        raise ValueError(
            f"Province '{province_input}' not recognized. "
            f"Valid options include: {', '.join(sorted(set(PROVINCE_DISPLAY_NAMES.values())))}"
        )
    
    prefixes = PROVINCE_TO_PREFIX_MAP[normalized]
    display_name = PROVINCE_DISPLAY_NAMES[normalized]
    
    # Create a slug version of the province name for filenames
    slug = display_name.lower().replace(' ', '_')
    
    return prefixes, display_name, slug

def filter_province(province_input):
    """
    Process geographic data for the specified province.
    
    Args:
        province_input: The province name or abbreviation
    """
    # Get province information
    try:
        prefixes, display_name, slug = get_province_info(province_input)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Processing FSA boundaries for {display_name}")
    print(f"Using FSA prefixes: {', '.join(prefixes)}")
    
    # Create directories for processed data and visualization output
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../plots', exist_ok=True)
    
    # Load the FSA boundaries Shapefile
    try:
        fsa_boundaries = gpd.read_file('../data/raw/lfsa000a21a_e.shp')
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        print("Make sure the shapefile exists at ../data/raw/lfsa000a21a_e.shp")
        sys.exit(1)
    
    # Explore and analyze the loaded data structure
    print(f"Columns in the Shapefile: {fsa_boundaries.columns.tolist()}")
    print(f"Total FSAs: {len(fsa_boundaries)}")
    print(f"Coordinate Reference System: {fsa_boundaries.crs}")
    
    # Examine a sample of the data to understand its structure
    print("\nSample records:")
    print(fsa_boundaries.head())
    
    # Automatically identify the FSA code column in the dataset
    fsa_column = None
    for column in fsa_boundaries.columns:
        # Check the first value to see if it matches FSA pattern
        sample_value = str(fsa_boundaries[column].iloc[0])
        if len(sample_value) == 3 and sample_value[0].isalpha() and sample_value[1].isdigit() and sample_value[2].isalpha():
            fsa_column = column
            print(f"Found FSA code column: {fsa_column}")
            break
    
    if not fsa_column:
        print("Could not automatically detect FSA column. Please check the columns and values above.")
        fsa_column = 'CFSAUID'  # Default guess, may need to be changed
    
    # Filter to keep only the specified province's FSAs based on their first letter
    province_fsas = fsa_boundaries[fsa_boundaries[fsa_column].str[0].isin(prefixes)]
    print(f"\nFiltered to {len(province_fsas)} {display_name} FSAs")
    
    if len(province_fsas) == 0:
        print(f"No FSAs found for {display_name} with prefixes {', '.join(prefixes)}.")
        print("Please check the data and prefixes.")
        sys.exit(1)
    
    # Rename the FSA column to a standard name for consistency
    province_fsas = province_fsas.rename(columns={fsa_column: 'FSA'})
    
    # Convert coordinate system to WGS84 (EPSG:4326) if needed
    if province_fsas.crs != 'EPSG:4326':
        print(f"Converting from {province_fsas.crs} to WGS84 (EPSG:4326)...")
        province_fsas = province_fsas.to_crs('EPSG:4326')
    
    # Save the processed province FSA boundaries to a GeoPackage file
    output_file = f'../data/{slug}_fsas.gpkg'
    province_fsas.to_file(output_file, driver='GPKG')
    print(f"Saved processed {display_name} FSA boundaries to {output_file}")
    
    # Create visualizations
    create_maps(province_fsas, display_name, slug, prefixes)
    
    print(f"\nFSA boundaries for {display_name} are now ready for use with OSMnx and other geospatial analyses!")

def create_maps(province_fsas, display_name, slug, prefixes):
    """
    Create and save map visualizations of province FSAs.
    
    Args:
        province_fsas: GeoDataFrame with the province's FSAs
        display_name: Display name of the province
        slug: Slug version of the province name for filenames
        prefixes: List of FSA prefixes for the province
    """
    create_fsa_map(province_fsas, display_name, slug)
    create_prefix_map(province_fsas, display_name, slug, prefixes)

def create_fsa_map(province_fsas, display_name, slug):
    """
    Create and save a map visualization of all province FSAs with white fill, black borders,
    and a light basemap.
    
    Args:
        province_fsas: GeoDataFrame with the province's FSAs
        display_name: Display name of the province
        slug: Slug version of the province name for filenames
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a copy and convert only the copy to Web Mercator for the map
    # This doesn't affect the original data or the saved geopackage file
    plot_data = province_fsas.copy()
    plot_data = plot_data.to_crs('EPSG:3857')
    
    # Plot the FSA boundaries with thin black borders and semi-transparent white fill
    plot_data.plot(
        ax=ax,
        color='white',        # Fixed white fill for all areas
        edgecolor='black',    # Black boundary lines
        linewidth=0.3,        # Thin boundary lines
        alpha=0.7             # Semi-transparent to see the basemap
    )
    
    # Add the CartoDB Voyager basemap
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.Voyager,
        attribution_size=8
    )
    
    plt.title(f'{display_name} Forward Sortation Areas (FSAs)', fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    
    output_file = f'../plots/{slug}_fsas_map.png'
    plt.savefig(output_file, dpi=300)
    print(f"Created map visualization at {output_file}")
    plt.close()

def create_prefix_map(province_fsas, display_name, slug, prefixes):
    """
    Create and save a map visualization of province FSAs colored by their prefix letter,
    with a light basemap.
    
    Args:
        province_fsas: GeoDataFrame with the province's FSAs
        display_name: Display name of the province
        slug: Slug version of the province name for filenames
        prefixes: List of FSA prefixes for the province
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a copy and convert only the copy to Web Mercator for the map
    plot_data = province_fsas.copy()
    plot_data = plot_data.to_crs('EPSG:3857')
    
    # Add a new column with just the first letter of each FSA
    plot_data['FSA_prefix'] = plot_data['FSA'].str[0]
    
    # Plot the FSAs colored by their prefix letter
    plot_data.plot(
        ax=ax, 
        column='FSA_prefix',
        legend=True,
        cmap='tab10',
        edgecolor='black',
        linewidth=0.3,
        alpha=0.7,
        legend_kwds={'loc': 'upper right', 'title': 'FSA Prefix'}
    )
    
    # Add the CartoDB Voyager basemap
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.Voyager,
        attribution_size=8
    )
    
    prefix_str = ', '.join(prefixes)
    plt.title(f'{display_name} FSAs by Prefix ({prefix_str})', fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    
    output_file = f'../plots/{slug}_prefix.png'
    plt.savefig(output_file, dpi=300)
    print(f"Created FSA prefix map at {output_file}")
    plt.close()

def main():
    """Main function to process FSA boundaries for a specified province."""
    # Get province from command line argument or prompt user
    if len(sys.argv) > 1:
        province = sys.argv[1]
    else:
        province = input("Enter province name or abbreviation: ")
    
    filter_province(province)

if __name__ == "__main__":
    main()
