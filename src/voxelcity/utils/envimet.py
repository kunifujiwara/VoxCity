import os
import numpy as np
import datetime

from ..geo.grid import apply_operation, translate_array, group_and_label_cells, process_grid
from ..utils.visualization import convert_land_cover

def array_to_string(arr):
    return '\n'.join('     ' + ','.join(str(cell) for cell in row) for row in arr)

def array_to_string_with_value(arr, value):
    return '\n'.join('     ' + ','.join(str(value) for cell in row) for row in arr)

def array_to_string_int(arr):
    return '\n'.join('     ' + ','.join(str(int(cell+0.5)) for cell in row) for row in arr)

def prepare_grids(building_height_grid_ori, canopy_height_grid_ori, land_cover_grid_ori, dem_grid_ori, meshsize, land_cover_source):
    building_height_grid = np.flipud(building_height_grid_ori).copy()
    building_height_grid[0, :] = building_height_grid[-1, :] = building_height_grid[:, 0] = building_height_grid[:, -1] = 0
    building_height_grid = apply_operation(building_height_grid, meshsize)

    if land_cover_source != 'OpenEarthMapJapan':
        land_cover_grid_converted = convert_land_cover(land_cover_grid_ori, land_cover_source=land_cover_source)  
    else:
        land_cover_grid_converted = land_cover_grid_ori    
    land_cover_grid = np.flipud(land_cover_grid_converted).copy() + 1

    veg_translation_dict = {
        1: '',
        2: '0200XX',
        3: '',
        4: '',
        5: '0200XX',
        6: '',
        7: '0200XX',
        8: ''
    }
    land_cover_veg_grid = translate_array(land_cover_grid, veg_translation_dict)

    mat_translation_dict = {
        1: '0200SD',#Bareland
        2: '000000',#Rangeland
        3: '0200PG',#Developed space
        4: '0200ST',#Road
        5: '000000',#Tree
        6: '0200WW',#Water
        7: '000000',#Agriculture land
        8: '000000',#Building
    }
    land_cover_mat_grid = translate_array(land_cover_grid, mat_translation_dict)

    # canopy_height_grid = np.flipud(canopy_height_grid_ori).copy()
    canopy_height_grid = canopy_height_grid_ori.copy()

    dem_grid = np.flipud(dem_grid_ori).copy()

    return building_height_grid, land_cover_veg_grid, land_cover_mat_grid, canopy_height_grid, dem_grid

def create_xml_content(building_height_grid, land_cover_veg_grid, land_cover_mat_grid, canopy_height_grid, dem_grid, meshsize):
    # XML template
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
         <useTelescoping_grid> 0 </useTelescoping_grid>
         <useSplitting> 1 </useSplitting>
         <verticalStretch> 0.00000 </verticalStretch>
         <startStretch> 0.00000 </startStretch>
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

    # Replace placeholders
    placeholders = {
        "$modelDescription$": "A brave new area",
        "$modelAuthor$": "[Enter model author name]",
        "$modelRotation$": "20",
        "$projectionSystem$": "GCS_WGS_1984",
        "$locationName$": "Essen/ Germany",
        "$location_Longitude$": "7.00000",
        "$location_Latitude$": "53.00000",
        "$locationTimeZone_Name$": "CET/ UTC+1",
        "$locationTimeZone_Longitude$": "15.00000",
    }

    for placeholder, value in placeholders.items():
        xml_template = xml_template.replace(placeholder, value)

    # Set grid dimensions
    grids_I, grids_J = building_height_grid.shape[1], building_height_grid.shape[0]
    grids_Z = max(int(100/meshsize), int(np.max(building_height_grid)/meshsize + 0.5) * 3)
    dx, dy, dz_base = meshsize, meshsize, meshsize

    grid_placeholders = {
        "$grids-I$": str(grids_I),
        "$grids-J$": str(grids_J),
        "$grids-Z$": str(grids_Z),
        "$dx$": str(dx),
        "$dy$": str(dy),
        "$dz-base$": str(dz_base),
    }

    for placeholder, value in grid_placeholders.items():
        xml_template = xml_template.replace(placeholder, value)

    # Replace matrix data
    xml_template = xml_template.replace("$zTop$", array_to_string(building_height_grid))
    xml_template = xml_template.replace("$zBottom$", array_to_string_with_value(building_height_grid, '0'))
    xml_template = xml_template.replace("$fixedheight$", array_to_string_with_value(building_height_grid, '0'))

    building_nr_grid = group_and_label_cells(building_height_grid)
    xml_template = xml_template.replace("$buildingNr$", array_to_string(building_nr_grid))

    xml_template = xml_template.replace("$ID_plants1D$", array_to_string(land_cover_veg_grid))

    # Add 3D plants
    tree_content = ""
    for i in range(grids_I):
        for j in range(grids_J):
            canopy_height = int(canopy_height_grid[j, i] + 0.5)
            if canopy_height_grid[j, i] > 0 and np.flipud(building_height_grid)[j, i]==0:
            # if canopy_height > 0 and building_height_grid[j, i]==0:
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

    xml_template = xml_template.replace("$3Dplants$", tree_content)
    xml_template = xml_template.replace("$ID_soilprofile$", array_to_string(land_cover_mat_grid))
    dem_grid = process_grid(building_nr_grid, dem_grid)
    xml_template = xml_template.replace("$DEMReference$", '0')
    xml_template = xml_template.replace("$terrainheight$", array_to_string_int(dem_grid))
    xml_template = xml_template.replace("$ID_sources$", array_to_string_with_value(land_cover_mat_grid, ''))

    return xml_template

def save_file(content, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def export_inx(building_height_grid_ori, canopy_height_grid_ori, land_cover_grid_ori, dem_grid_ori, meshsize, land_cover_source, output_dir="output"):
    # Prepare grids
    building_height_grid_inx, land_cover_veg_grid_inx, land_cover_mat_grid_inx, canopy_height_grid_inx, dem_grid_inx = prepare_grids(
        building_height_grid_ori.copy(), canopy_height_grid_ori.copy(), land_cover_grid_ori.copy(), dem_grid_ori.copy(), meshsize, land_cover_source)

    # Create XML content
    xml_content = create_xml_content(building_height_grid_inx, land_cover_veg_grid_inx, land_cover_mat_grid_inx, canopy_height_grid_inx, dem_grid_inx, meshsize)

    # Save the output
    output_file_path = os.path.join(output_dir, "output.INX")
    save_file(xml_content, output_file_path)

def generate_edb_file(lad='1.00000'):
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
     <Depth> {height * 0.4:.5f} </Depth>
     <RootDiameter> 1.00000 </RootDiameter>
     <cellsize> 1.00000 </cellsize>
     <xy_cells> 1 </xy_cells>
     <z_cells> {height} </z_cells>
     <scalefactor> 0.00000 </scalefactor>
     <LAD-Profile type="sparematrix-3D" dataI="1" dataJ="1" zlayers="{height}" defaultValue="0.00000">
{generate_lad_profile(height, lad=lad)}
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
    
    with open('plant3d_objects.edb', 'w') as f:
        f.write(content)

def generate_lad_profile(height, lad = '1.00000'):
    lad_profile = []
    start = max(0, int(height * 0.4))
    for i in range(start, height):
        lad_profile.append(f"     0,0,{i},{lad}")
    return '\n'.join(lad_profile)
    
