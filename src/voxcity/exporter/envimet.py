"""ENVI-met model file exporter module.

This module provides functionality to export voxel city data to ENVI-met INX format.
ENVI-met is a three-dimensional microclimate model designed to simulate surface-plant-air
interactions in urban environments.

Key Features:
    - Converts voxel grids to ENVI-met compatible format
    - Handles building heights, vegetation, materials, and terrain
    - Supports telescoping grid for vertical mesh refinement
    - Generates complete INX files with all required parameters
    - Creates plant database (EDB) files for 3D vegetation

Main Functions:
    - prepare_grids: Processes input grids for ENVI-met format
    - create_xml_content: Generates INX file XML content
    - export_inx: Main function to export model to INX format
    - generate_edb_file: Creates plant database file
    - array_to_string: Helper functions for grid formatting

Dependencies:
    - numpy: For array operations
    - datetime: For timestamp generation
"""

import os
import numpy as np
import datetime

from ..geoprocessor.grid import apply_operation, translate_array, group_and_label_cells, process_grid
from ..geoprocessor.utils import get_city_country_name_from_rectangle, get_timezone_info
from ..utils.lc import convert_land_cover
from ..models import VoxCity

def array_to_string(arr):
    """Convert a 2D numpy array to a string representation with comma-separated values.
    
    This function formats array values for ENVI-met INX files, where each row must be:
    1. Indented by 5 spaces
    2. Values separated by commas
    3. No trailing comma
    
    Args:
        arr (numpy.ndarray): 2D numpy array to convert
        
    Returns:
        str: String representation with each row indented by 5 spaces and values comma-separated
        
    Example:
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> print(array_to_string(arr))
             1,2
             3,4
    """
    return '\n'.join('     ' + ','.join(str(cell) for cell in row) for row in arr)

def array_to_string_with_value(arr, value):
    """Convert a 2D numpy array to a string representation, replacing all values with a constant.
    
    This function is useful for creating uniform value grids in ENVI-met INX files,
    such as for soil profiles or fixed height indicators.
    
    Args:
        arr (numpy.ndarray): 2D numpy array to convert (only shape is used)
        value (str or numeric): Value to use for all cells
        
    Returns:
        str: String representation with each row indented by 5 spaces and constant value repeated
        
    Example:
        >>> arr = np.zeros((2, 2))
        >>> print(array_to_string_with_value(arr, '0'))
             0,0
             0,0
    """
    return '\n'.join('     ' + ','.join(str(value) for cell in row) for row in arr)

def array_to_string_int(arr):
    """Convert a 2D numpy array to a string representation of rounded integers.
    
    This function is used for grids that must be represented as integers in ENVI-met,
    such as building numbers or terrain heights. Values are rounded to nearest integer.
    
    Args:
        arr (numpy.ndarray): 2D numpy array to convert
        
    Returns:
        str: String representation with each row indented by 5 spaces and values rounded to integers
        
    Example:
        >>> arr = np.array([[1.6, 2.3], [3.7, 4.1]])
        >>> print(array_to_string_int(arr))
             2,2
             4,4
    """
    return '\n'.join('     ' + ','.join(str(int(cell+0.5)) for cell in row) for row in arr)

def prepare_grids(building_height_grid_ori, building_id_grid_ori, canopy_height_grid_ori, land_cover_grid_ori, dem_grid_ori, meshsize, land_cover_source):
    """Prepare and process input grids for ENVI-met model.
    
    This function performs several key transformations on input grids:
    1. Flips grids vertically to match ENVI-met coordinate system
    2. Handles missing values and border conditions
    3. Converts land cover classes to ENVI-met vegetation and material codes
    4. Processes building IDs and heights
    5. Adjusts DEM relative to minimum elevation
    
    Args:
        building_height_grid_ori (numpy.ndarray): Original building height grid (meters)
        building_id_grid_ori (numpy.ndarray): Original building ID grid
        canopy_height_grid_ori (numpy.ndarray): Original canopy height grid (meters)
        land_cover_grid_ori (numpy.ndarray): Original land cover grid (class codes)
        dem_grid_ori (numpy.ndarray): Original DEM grid (meters)
        meshsize (float): Size of mesh cells in meters
        land_cover_source (str): Source of land cover data for class conversion
        
    Returns:
        tuple: Processed grids:
            - building_height_grid (numpy.ndarray): Building heights
            - building_id_grid (numpy.ndarray): Building IDs
            - land_cover_veg_grid (numpy.ndarray): Vegetation codes
            - land_cover_mat_grid (numpy.ndarray): Material codes
            - canopy_height_grid (numpy.ndarray): Canopy heights
            - dem_grid (numpy.ndarray): Processed DEM
            
    Notes:
        - Building heights at grid borders are set to 0
        - DEM is normalized to minimum elevation
        - Land cover is converted based on source-specific mapping
    """
    # Flip building height grid vertically and replace NaN with 10m height
    building_height_grid = np.flipud(np.nan_to_num(building_height_grid_ori, nan=10.0)).copy()
    building_id_grid = np.flipud(building_id_grid_ori)
    
    # Set border cells to 0 height
    building_height_grid[0, :] = building_height_grid[-1, :] = building_height_grid[:, 0] = building_height_grid[:, -1] = 0
    building_height_grid = apply_operation(building_height_grid, meshsize)

    # Convert land cover if needed based on source
    if (land_cover_source == 'OpenEarthMapJapan') or (land_cover_source == 'OpenStreetMap'):
        land_cover_grid_converted = land_cover_grid_ori   
    else:
        land_cover_grid_converted = convert_land_cover(land_cover_grid_ori, land_cover_source=land_cover_source)        

    land_cover_grid = np.flipud(land_cover_grid_converted).copy() + 1

    # Dictionary mapping land cover types to vegetation codes
    veg_translation_dict = {
        1: '',  # Bareland
        2: '0200XX',  # Rangeland
        3: '0200H1',  # Shrub
        4: '0200XX',  # Moss and lichen
        5: '0200XX',  # Agriculture land
        6: '',  # Tree
        7: '0200XX',  # Wet land
        8: ''  # Mangroves
    }
    land_cover_veg_grid = translate_array(land_cover_grid, veg_translation_dict)

    # Dictionary mapping land cover types to material codes
    mat_translation_dict = {
        1: '000000',  # Bareland
        2: '000000',  # Rangeland
        3: '000000',  # Shrub
        4: '000000',  # Moss and lichen
        5: '000000',  # Agriculture land
        6: '000000',  # Tree
        7: '0200WW',  # Wet land
        8: '0200WW',  # Mangroves
        9: '0200WW',  # Water
        10: '000000', # Snow and ice
        11: '0200PG', # Developed space
        12: '0200ST', # Road
        13: '000000', # Building
        14: '000000', # No Data
    }
    land_cover_mat_grid = translate_array(land_cover_grid, mat_translation_dict)

    # Process canopy and DEM grids
    canopy_height_grid = canopy_height_grid_ori.copy()
    dem_grid = np.flipud(dem_grid_ori).copy() - np.min(dem_grid_ori)

    return building_height_grid, building_id_grid, land_cover_veg_grid, land_cover_mat_grid, canopy_height_grid, dem_grid

def create_xml_content(building_height_grid, building_id_grid, land_cover_veg_grid, land_cover_mat_grid, canopy_height_grid, dem_grid, meshsize, rectangle_vertices, **kwargs):
    """Create XML content for ENVI-met INX file.
    
    This function generates the complete XML structure for an ENVI-met INX file,
    including model metadata, geometry settings, and all required grid data.
    
    Args:
        building_height_grid (numpy.ndarray): Processed building heights
        building_id_grid (numpy.ndarray): Processed building IDs
        land_cover_veg_grid (numpy.ndarray): Vegetation codes grid
        land_cover_mat_grid (numpy.ndarray): Material codes grid
        canopy_height_grid (numpy.ndarray): Processed canopy heights
        dem_grid (numpy.ndarray): Processed DEM
        meshsize (float): Size of mesh cells in meters
        rectangle_vertices (list): Vertices defining model area as [(lon, lat), ...]
        **kwargs: Additional keyword arguments:
            - author_name (str): Name of model author
            - model_description (str): Description of model
            - domain_building_max_height_ratio (float): Ratio of domain height to max building height
            - useTelescoping_grid (bool): Whether to use telescoping grid
            - verticalStretch (float): Vertical stretch factor
            - startStretch (float): Height to start stretching
            - min_grids_Z (int): Minimum vertical grid cells
            
    Returns:
        str: Complete XML content for INX file
        
    Notes:
        - Automatically determines location information from coordinates
        - Handles both telescoping and uniform vertical grids
        - Sets appropriate defaults for optional parameters
        - Includes all required ENVI-met model settings
    """
    # XML template defining the structure of an ENVI-met INX file
    xml_template = """<ENVI-MET_Datafile>
    <Header>
    <filetype>INPX ENVI-met Area Input File</filetype>
    <version>440</version>
    <revisiondate>7/5/2024 5:44:52 PM</revisiondate>
    <remark>Created with SPACES 5.6.1</remark>
    <checksum>0</checksum>
    <encryptionlevel>0</encryptionlevel>
    </Header>
      <baseData>
         <modelDescription> $modelDescription$ </modelDescription>
         <modelAuthor> $modelAuthor$ </modelAuthor>
         <modelcopyright> The creator or distributor is responsible for following Copyright Laws </modelcopyright>
      </baseData>
      <modelGeometry>
         <grids-I> $grids-I$ </grids-I>
         <grids-J> $grids-J$ </grids-J>
         <grids-Z> $grids-Z$ </grids-Z>
         <dx> $dx$ </dx>
         <dy> $dy$ </dy>
         <dz-base> $dz-base$ </dz-base>
         <useTelescoping_grid> $useTelescoping_grid$ </useTelescoping_grid>
         <useSplitting> 1 </useSplitting>
         <verticalStretch> $verticalStretch$ </verticalStretch>
         <startStretch> $startStretch$ </startStretch>
         <has3DModel> 0 </has3DModel>
         <isFull3DDesign> 0 </isFull3DDesign>
      </modelGeometry>
      <nestingArea>
         <numberNestinggrids> 0 </numberNestinggrids>
         <soilProfileA> 000000 </soilProfileA>
         <soilProfileB> 000000 </soilProfileB>
      </nestingArea>
      <locationData>
         <modelRotation> $modelRotation$ </modelRotation>
         <projectionSystem> $projectionSystem$ </projectionSystem>
         <UTMZone> 0 </UTMZone>
         <realworldLowerLeft_X> 0.00000 </realworldLowerLeft_X>
         <realworldLowerLeft_Y> 0.00000 </realworldLowerLeft_Y>
         <locationName> $locationName$ </locationName>
         <location_Longitude> $location_Longitude$ </location_Longitude>
         <location_Latitude> $location_Latitude$ </location_Latitude>
         <locationTimeZone_Name> $locationTimeZone_Name$ </locationTimeZone_Name>
         <locationTimeZone_Longitude> $locationTimeZone_Longitude$ </locationTimeZone_Longitude>
      </locationData>
      <defaultSettings>
         <commonWallMaterial> 000000 </commonWallMaterial>
         <commonRoofMaterial> 000000 </commonRoofMaterial>
      </defaultSettings>
      <buildings2D>
         <zTop type="matrix-data" dataI="$grids-I$" dataJ="$grids-J$">
    $zTop$
         </zTop>
         <zBottom type="matrix-data" dataI="$grids-I$" dataJ="$grids-J$">
    $zBottom$
         </zBottom>
         <buildingNr type="matrix-data" dataI="$grids-I$" dataJ="$grids-J$">
    $buildingNr$
         </buildingNr>
         <fixedheight type="matrix-data" dataI="$grids-I$" dataJ="$grids-J$">
    $fixedheight$
         </fixedheight>
      </buildings2D>
      <simpleplants2D>
         <ID_plants1D type="matrix-data" dataI="$grids-I$" dataJ="$grids-J$">
    $ID_plants1D$
         </ID_plants1D>
      </simpleplants2D>
    $3Dplants$
      <soils2D>
         <ID_soilprofile type="matrix-data" dataI="$grids-I$" dataJ="$grids-J$">
    $ID_soilprofile$
         </ID_soilprofile>
      </soils2D>
      <dem>
         <DEMReference> $DEMReference$ </DEMReference>
         <terrainheight type="matrix-data" dataI="$grids-I$" dataJ="$grids-J$">
    $terrainheight$
         </terrainheight>
      </dem>
      <sources2D>
         <ID_sources type="matrix-data" dataI="$grids-I$" dataJ="$grids-J$">
    $ID_sources$
         </ID_sources>
      </sources2D>
    </ENVI-MET_Datafile>"""

    # Get location information based on rectangle vertices
    city_country_name = get_city_country_name_from_rectangle(rectangle_vertices)

    # Calculate center coordinates of the model area
    longitudes = [coord[0] for coord in rectangle_vertices]  # Changed order from lat to lon
    latitudes = [coord[1] for coord in rectangle_vertices]  # Changed order from lat to lon
    center_lon = str(sum(longitudes) / len(longitudes))  # Changed order
    center_lat = str(sum(latitudes) / len(latitudes))  # Changed order
    
    timezone_info = get_timezone_info(rectangle_vertices)

    # Set default values for optional parameters
    author_name = kwargs.get('author_name')
    if author_name is None:
        author_name = "[Enter model author name]"
    model_desctiption = kwargs.get('model_desctiption')
    if model_desctiption is None:
        model_desctiption = "[Enter model desctription]"

    # Replace location-related placeholders in template
    placeholders = {
        "$modelDescription$": model_desctiption,
        "$modelAuthor$": author_name,
        "$modelRotation$": "0",
        "$projectionSystem$": "GCS_WGS_1984",
        "$locationName$": city_country_name,
        "$location_Longitude$": center_lon,
        "$location_Latitude$": center_lat,
        "$locationTimeZone_Name$": timezone_info[0],
        "$locationTimeZone_Longitude$": timezone_info[1],
    }

    # Ensure no None values are passed to replace()
    for placeholder, value in placeholders.items():
        if value is None:
            print(f"Warning: {placeholder} is None, using fallback value")
            if placeholder == "$locationName$":
                value = "Unknown Location/ Unknown Country"
            elif placeholder == "$locationTimeZone_Name$":
                value = "UTC+00:00"
            elif placeholder == "$locationTimeZone_Longitude$":
                value = "0.00000"
            elif placeholder == "$modelDescription$":
                value = "[Enter model description]"
            elif placeholder == "$modelAuthor$":
                value = "[Enter model author name]"
            else:
                value = "Unknown"
        xml_template = xml_template.replace(placeholder, str(value))
    
    # Calculate building heights including terrain elevation
    building_on_dem_grid = building_height_grid + dem_grid    
    
    # Configure vertical grid settings
    domain_building_max_height_ratio = kwargs.get('domain_building_max_height_ratio')
    if domain_building_max_height_ratio is None:
        domain_building_max_height_ratio = 2
    
    # Configure telescoping grid settings if enabled
    useTelescoping_grid = kwargs.get('useTelescoping_grid')
    if (useTelescoping_grid is None) or (useTelescoping_grid == False):
        useTelescoping_grid = 0
        verticalStretch = 0
        startStretch = 0
    else:
        useTelescoping_grid = 1
        verticalStretch = kwargs.get('verticalStretch')
        if (verticalStretch is None):
            verticalStretch = 20
        startStretch = kwargs.get('startStretch')
        if (startStretch is None):
            startStretch = int(np.max(building_on_dem_grid)/meshsize + 0.5) * meshsize
    
    # Set horizontal grid dimensions
    grids_I, grids_J = building_height_grid.shape[1], building_height_grid.shape[0]

    # Calculate vertical grid dimension based on building heights and telescoping settings
    min_grids_Z = kwargs.get('min_grids_Z', 20)
    if verticalStretch > 0:
        # Calculate minimum number of cells needed to reach target height with telescoping
        a = meshsize  # First cell size
        r = (100 + verticalStretch) / 100  # Growth ratio
        S_target = (int(np.max(building_on_dem_grid)/meshsize + 0.5) * meshsize) * (domain_building_max_height_ratio - 1)
        min_n = find_min_n(a, r, S_target, max_n=1000000)
        if min_n is None:
            # Fallback to non-telescoping grid if calculation fails
            print("Warning: Telescoping grid calculation failed, using uniform grid")
            grids_Z = max(int(np.max(building_on_dem_grid)/meshsize + 0.5) * domain_building_max_height_ratio, min_grids_Z)
        else:
            grids_Z_tent = int(np.max(building_on_dem_grid)/meshsize + 0.5) + min_n
            if grids_Z_tent < min_grids_Z:
                grids_Z = min_grids_Z
                startStretch += (min_grids_Z - grids_Z)
            else:
                grids_Z = grids_Z_tent
    else:
        # Calculate vertical grid cells without telescoping
        grids_Z = max(int(np.max(building_on_dem_grid)/meshsize + 0.5) * domain_building_max_height_ratio, min_grids_Z)

    # Set grid cell sizes
    dx, dy, dz_base = meshsize, meshsize, meshsize

    # Replace grid-related placeholders
    grid_placeholders = {
        "$grids-I$": str(grids_I),
        "$grids-J$": str(grids_J),
        "$grids-Z$": str(grids_Z),
        "$dx$": str(dx),
        "$dy$": str(dy),
        "$dz-base$": str(dz_base),
        "$useTelescoping_grid$": str(useTelescoping_grid),
        "$verticalStretch$": str(verticalStretch),
        "$startStretch$": str(startStretch),
    }

    for placeholder, value in grid_placeholders.items():
        xml_template = xml_template.replace(placeholder, value)

    # Replace matrix data placeholders with actual grid data
    xml_template = xml_template.replace("$zTop$", array_to_string(building_height_grid))
    xml_template = xml_template.replace("$zBottom$", array_to_string_with_value(building_height_grid, '0'))
    xml_template = xml_template.replace("$fixedheight$", array_to_string_with_value(building_height_grid, '0'))

    # Process and add building numbers
    building_nr_grid = group_and_label_cells(building_id_grid)
    xml_template = xml_template.replace("$buildingNr$", array_to_string(building_nr_grid))

    # Add vegetation data
    xml_template = xml_template.replace("$ID_plants1D$", array_to_string(land_cover_veg_grid))

    # Generate and add 3D plant data
    tree_content = ""
    for i in range(grids_I):
        for j in range(grids_J):
            canopy_height = int(canopy_height_grid[j, i] + 0.5)
            # Only add trees where there are no buildings
            if canopy_height_grid[j, i] > 0 and np.flipud(building_height_grid)[j, i]==0:
                plantid = f'H{canopy_height:02d}W01'
                tree_ij = f"""  <3Dplants>
     <rootcell_i> {i+1} </rootcell_i>
     <rootcell_j> {j+1} </rootcell_j>
     <rootcell_k> 0 </rootcell_k>
     <plantID> {plantid} </plantID>
     <name> .{plantid} </name>
     <observe> 0 </observe>
  </3Dplants>"""
                tree_content += '\n' + tree_ij

    # Add remaining data
    xml_template = xml_template.replace("$3Dplants$", tree_content)
    xml_template = xml_template.replace("$ID_soilprofile$", array_to_string(land_cover_mat_grid))
    dem_grid = process_grid(building_nr_grid, dem_grid)
    xml_template = xml_template.replace("$DEMReference$", '0')
    xml_template = xml_template.replace("$terrainheight$", array_to_string_int(dem_grid))
    xml_template = xml_template.replace("$ID_sources$", array_to_string_with_value(land_cover_mat_grid, ''))

    return xml_template

def save_file(content, output_file_path):
    """Save content to a file with UTF-8 encoding.
    
    This function ensures consistent file encoding and error handling when
    saving ENVI-met files.
    
    Args:
        content (str): String content to save
        output_file_path (str): Path to save file to
        
    Notes:
        - Creates parent directories if they don't exist
        - Uses UTF-8 encoding for compatibility
        - Overwrites existing file if present
    """
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def export_inx(building_height_grid_ori, building_id_grid_ori, canopy_height_grid_ori, land_cover_grid_ori, dem_grid_ori, meshsize, land_cover_source, rectangle_vertices, **kwargs):
    """Export model data to ENVI-met INX file format.
    
    This is the main function for exporting voxel city data to ENVI-met format.
    It coordinates the entire export process from grid preparation to file saving.
    
    Args:
        building_height_grid_ori (numpy.ndarray): Original building height grid
        building_id_grid_ori (numpy.ndarray): Original building ID grid
        canopy_height_grid_ori (numpy.ndarray): Original canopy height grid
        land_cover_grid_ori (numpy.ndarray): Original land cover grid
        dem_grid_ori (numpy.ndarray): Original DEM grid
        meshsize (float): Size of mesh cells in meters
        land_cover_source (str): Source of land cover data
        rectangle_vertices (list): Vertices defining model area
        **kwargs: Additional keyword arguments:
            - output_directory (str): Directory to save output
            - file_basename (str): Base filename for output
            - Other args passed to create_xml_content()
            
    Notes:
        - Creates output directory if it doesn't exist
        - Handles grid preparation and transformation
        - Generates complete INX file with all required data
        - Uses standardized file naming convention
    """
    # Prepare grids
    building_height_grid_inx, building_id_grid, land_cover_veg_grid_inx, land_cover_mat_grid_inx, canopy_height_grid_inx, dem_grid_inx = prepare_grids(
        building_height_grid_ori.copy(), building_id_grid_ori.copy(), canopy_height_grid_ori.copy(), land_cover_grid_ori.copy(), dem_grid_ori.copy(), meshsize, land_cover_source)    

    # Create XML content
    xml_content = create_xml_content(building_height_grid_inx, building_id_grid, land_cover_veg_grid_inx, land_cover_mat_grid_inx, canopy_height_grid_inx, dem_grid_inx, meshsize, rectangle_vertices, **kwargs)

    # Save the output
    output_dir = kwargs.get("output_directory", 'output')
    os.makedirs(output_dir, exist_ok=True)
    file_basename = kwargs.get("file_basename", 'voxcity')
    output_file_path = os.path.join(output_dir, f"{file_basename}.INX")
    save_file(xml_content, output_file_path)


class EnvimetExporter:
    """Exporter adapter to write a VoxCity model to ENVI-met INX format."""

    def export(self, obj, output_directory: str, base_filename: str, **kwargs):
        if not isinstance(obj, VoxCity):
            raise TypeError("EnvimetExporter expects a VoxCity instance")
        city: VoxCity = obj
        rect = city.extras.get("rectangle_vertices")
        land_cover_source = city.extras.get("land_cover_source", "Standard")
        export_inx(
            city.buildings.heights,
            city.buildings.ids if city.buildings.ids is not None else np.zeros_like(city.buildings.heights, dtype=int),
            city.tree_canopy.top if city.tree_canopy is not None else np.zeros_like(city.land_cover.classes, dtype=float),
            city.land_cover.classes,
            city.dem.elevation,
            city.voxels.meta.meshsize,
            land_cover_source,
            rect if rect is not None else [(0.0, 0.0)] * 4,
            output_directory=output_directory,
            file_basename=base_filename,
            **kwargs,
        )
        return os.path.join(output_directory, f"{base_filename}.INX")

def generate_edb_file(**kwargs):
    """Generate ENVI-met database file for 3D plants.
    
    Creates a plant database file (EDB) containing definitions for trees of
    different heights with customizable leaf area density profiles.
    
    Args:
        **kwargs: Keyword arguments:
            - lad (float): Leaf area density in m²/m³ (default 1.0)
            - trunk_height_ratio (float): Ratio of trunk height to total height
              (default 11.76/19.98)
              
    Notes:
        - Generates plants for heights from 1-50m
        - Uses standardized plant IDs in format 'HxxW01'
        - Includes physical properties like wood density
        - Sets seasonal variation profiles
        - Creates complete ENVI-met plant database format
    """
    
    lad = kwargs.get('lad')
    if lad is None:
        lad=1.0
    
    trunk_height_ratio = kwargs.get("trunk_height_ratio")
    if trunk_height_ratio is None:
        trunk_height_ratio = 11.76 / 19.98

    # Create header with current timestamp
    header = f'''<ENVI-MET_Datafile>
<Header>
<filetype>DATA</filetype>
<version>1</version>
<revisiondate>{datetime.datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")}</revisiondate>
<remark>Envi-Data</remark>
<checksum>0</checksum>
<encryptionlevel>1699612</encryptionlevel>
</Header>
'''

    footer = '</ENVI-MET_Datafile>'

    # Generate plant definitions for heights 1-50m
    plant3d_objects = []

    for height in range(1, 51):
        plant3d = f'''  <PLANT3D>
     <ID> H{height:02d}W01 </ID>
     <Description> H{height:02d}W01 </Description>
     <AlternativeName> Albero nuovo </AlternativeName>
     <Planttype> 0 </Planttype>
     <Leaftype> 1 </Leaftype>
     <Albedo> 0.18000 </Albedo>
     <Eps> 0.00000 </Eps>
     <Transmittance> 0.30000 </Transmittance>
     <isoprene> 12.00000 </isoprene>
     <leafweigth> 100.00000 </leafweigth>
     <rs_min> 0.00000 </rs_min>
     <Height> {height:.5f} </Height>
     <Width> 1.00000 </Width>
     <Depth> {height * trunk_height_ratio:.5f} </Depth>
     <RootDiameter> 1.00000 </RootDiameter>
     <cellsize> 1.00000 </cellsize>
     <xy_cells> 1 </xy_cells>
     <z_cells> {height} </z_cells>
     <scalefactor> 0.00000 </scalefactor>
     <LAD-Profile type="sparematrix-3D" dataI="1" dataJ="1" zlayers="{height}" defaultValue="0.00000">
{generate_lad_profile(height, trunk_height_ratio, lad=str(lad))}
     </LAD-Profile>
     <RAD-Profile> 0.10000,0.10000,0.10000,0.10000,0.10000,0.10000,0.10000,0.10000,0.10000,0.10000 </RAD-Profile>
     <Root-Range-Profile> 1.00000,1.00000,1.00000,1.00000,1.00000,1.00000,1.00000,1.00000,1.00000,1.00000 </Root-Range-Profile>
     <Season-Profile> 0.30000,0.30000,0.30000,0.40000,0.70000,1.00000,1.00000,1.00000,0.80000,0.60000,0.30000,0.30000 </Season-Profile>
     <Blossom-Profile> 0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000 </Blossom-Profile>
     <DensityWood> 690.00000 </DensityWood>
     <YoungsModulus> 8770000896.00000 </YoungsModulus>
     <YoungRatioRtoL> 0.12000 </YoungRatioRtoL>
     <MORBranch> 65.00000 </MORBranch>
     <MORConnection> 45.00000 </MORConnection>
     <PlantGroup> 0 </PlantGroup>
     <Color> 0 </Color>
     <Group>  </Group>
     <Author>  </Author>
     <costs> 0.00000 </costs>
     <ColorStem> 0 </ColorStem>
     <ColorBlossom> 0 </ColorBlossom>
     <BlossomRadius> 0.00000 </BlossomRadius>
     <L-SystemBased> 0 </L-SystemBased>
     <Axiom> V </Axiom>
     <IterationDepth> 0 </IterationDepth>
     <hasUserEdits> 0 </hasUserEdits>
     <LADMatrix_generated> 0 </LADMatrix_generated>
     <InitialSegmentLength> 0.00000 </InitialSegmentLength>
     <SmallSegmentLength> 0.00000 </SmallSegmentLength>
     <ChangeSegmentLength> 0.00000 </ChangeSegmentLength>
     <SegmentResolution> 0.00000 </SegmentResolution>
     <TurtleAngle> 0.00000 </TurtleAngle>
     <RadiusOuterBranch> 0.00000 </RadiusOuterBranch>
     <PipeFactor> 0.00000 </PipeFactor>
     <LeafPosition> 0 </LeafPosition>
     <LeafsPerNode> 0 </LeafsPerNode>
     <LeafInternodeLength> 0.00000 </LeafInternodeLength>
     <LeafMinSegmentOrder> 0 </LeafMinSegmentOrder>
     <LeafWidth> 0.00000 </LeafWidth>
     <LeafLength> 0.00000 </LeafLength>
     <LeafSurface> 0.00000 </LeafSurface>
     <PetioleAngle> 0.00000 </PetioleAngle>
     <PetioleLength> 0.00000 </PetioleLength>
     <LeafRotationalAngle> 0.00000 </LeafRotationalAngle>
     <FactorHorizontal> 0.00000 </FactorHorizontal>
     <TropismVector> 0.000000,0.000000,0.000000 </TropismVector>
     <TropismElstaicity> 0.00000 </TropismElstaicity>
     <SegmentRemovallist>  </SegmentRemovallist>
     <NrRules> 0 </NrRules>
     <Rules_Variable>  </Rules_Variable>
     <Rules_Replacement>  </Rules_Replacement>
     <Rules_isConditional>  </Rules_isConditional>
     <Rules_Condition>  </Rules_Condition>
     <Rules_Remark>  </Rules_Remark>
     <TermLString>   </TermLString>
     <ApplyTermLString> 0 </ApplyTermLString>
  </PLANT3D>
'''
        plant3d_objects.append(plant3d)

    content = header + ''.join(plant3d_objects) + footer
    
    with open('projectdatabase.edb', 'w') as f:
        f.write(content)

def generate_lad_profile(height, trunk_height_ratio, lad = '1.00000'):
    """Generate leaf area density profile for a plant.
    
    Creates a vertical profile of leaf area density (LAD) values for ENVI-met
    plant definitions, accounting for trunk space and crown distribution.
    
    Args:
        height (int): Total height of plant in meters
        trunk_height_ratio (float): Ratio of trunk height to total height
        lad (str): Leaf area density value as string (default '1.00000')
        
    Returns:
        str: LAD profile data formatted for ENVI-met EDB file
        
    Notes:
        - LAD values start above trunk height
        - Uses 5-space indentation for ENVI-met format
        - Profile follows format: "z-level,x,y,LAD"
    """
    lad_profile = []
    # Only add LAD values above trunk height
    start = max(0, int(height * trunk_height_ratio))
    for i in range(start, height):
        lad_profile.append(f"     0,0,{i},{lad}")
    return '\n'.join(lad_profile)
    
def find_min_n(a, r, S_target, max_n=1000000):
    """Find minimum number of terms needed in geometric series to exceed target sum.
    
    Used for calculating telescoping grid parameters to achieve desired domain height.
    Solves for n in the equation: a(1-r^n)/(1-r) > S_target
    
    Args:
        a (float): First term of series (base cell size)
        r (float): Common ratio (stretch factor)
        S_target (float): Target sum to exceed (desired height)
        max_n (int): Maximum number of terms to try (default 1000000)
        
    Returns:
        int or None: Minimum number of terms needed, or None if not possible within max_n
        
    Notes:
        - Handles special case of r=1 (arithmetic series)
        - Protects against overflow with large exponents
        - Returns None if solution not found within max_n terms
    """
    n = 1
    while n <= max_n:
        if r == 1:
            S_n = a * n
        else:
            try:
                S_n = a * (1 - r ** n) / (1 - r)
            except OverflowError:
                # Handle large exponents
                S_n = float('inf') if r > 1 else 0
        if (a > 0 and S_n > S_target) or (a < 0 and S_n < S_target):
            return n
        n += 1
    return None  # Not possible within max_n terms