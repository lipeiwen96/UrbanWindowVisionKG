import json
import os
import platform
import time
from dataclasses import dataclass, field
from typing import List

# --- Assume these imports are correct based on your provided code ---
# Adjust paths if your project structure is different
try:
    from map_system.utils.map_structure import MapRoadCenterLine, MapBuilding, MapLot, MapRoadPolygon
    from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement
    from art_public_modules.art_data_structure.shapely.data_element_material import DataElementMaterial
    from art_public_modules.art_data_exchange.data_exchange import ARTShapelyDataExchanger
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure map_system and art_public_modules are correctly installed and accessible.")
    exit()
# --- End of assumed imports ---

# --- Configuration ---
OUTPUT_3DM_FILENAME = "hong_kong_full_map2.3dm"

# Define base path based on OS
if platform.system() == "Windows":
    # Adjust this path if your 'library' or project root is elsewhere
    BASE_MAP_PATH = os.path.abspath(os.path.join("../library", "HK_map", "row_map"))
    OUTPUT_DIR = os.path.abspath(os.path.join("../output"))
else:
    # Adjust this path for your Linux/Mac environment
    BASE_MAP_PATH = "/map_data/HK_map/row_map"
    OUTPUT_DIR = "/output_data" # Example output directory for Linux

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_3DM_PATH = os.path.join(OUTPUT_DIR, OUTPUT_3DM_FILENAME)

# Paths to specific data files
ROAD_CENTERLINE_PATH = os.path.join(BASE_MAP_PATH, "road", "GEO_STREET_CENTRELINE.json")
BUILDING_PATH = os.path.join(BASE_MAP_PATH, "building", "BUILDING_STRUCTURE.json")
LOT_PATH = os.path.join(BASE_MAP_PATH, "lot", "LOT.json")
GLA_PATH = os.path.join(BASE_MAP_PATH, "lot", "GovernmentLandAllocation.json")
ROAD_POLYGON_PATH = os.path.join(BASE_MAP_PATH, "road", "INV_PG.json") # Assuming INV_PG is road polygons

# --- Helper Function ---
def read_map_json(path: str):
    """Reads a GeoJSON-like file and returns the 'features' list."""
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data.get("features", [])
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {path}: {e}")
        return []
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return []

def load_map_data(file_path: str, map_class: type, data_list: list):
    """Reads features from a JSON file and populates a list with map objects."""
    print(f"Reading {map_class.__name__} data from: {file_path}")
    features = read_map_json(file_path)
    count = 0
    error_count = 0
    start_time = time.time()
    for i, feature_data in enumerate(features):
        if i > 10000:
            break
        try:
            map_object = map_class()
            map_object.init(feature_data) # Assumes .init() parses feature_data
            if hasattr(map_object, 'geometry') and map_object.geometry and not map_object.geometry.is_empty:
                 data_list.append(map_object)
                 count += 1
            else:
                 print(f"Warning: Skipped {map_class.__name__} feature {i+1} due to empty or invalid geometry.")
                 error_count += 1

        except Exception as e:
            print(f"Error processing {map_class.__name__} feature {i+1}: {e}. Skipping.")
            error_count += 1
        # Optional: Progress indicator for large files
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(features)} {map_class.__name__} features...")

    end_time = time.time()
    print(f"Finished reading {map_class.__name__}. Loaded: {count}, Errors/Skipped: {error_count}. Time: {end_time - start_time:.2f}s")

# --- Main Export Class ---
@dataclass
class FullMapExporter:
    road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list)
    road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)
    building_list: List[MapBuilding] = field(default_factory=list)
    lot_list: List[MapLot] = field(default_factory=list) # Will hold both LOT and GLA

    model: DataModel = field(default_factory=lambda: DataModel(name="HongKongFullMap"))

    def load_all_data(self):
        """Loads all raw map data from the specified JSON files."""
        print("--- Starting Data Loading ---")
        load_map_data(ROAD_CENTERLINE_PATH, MapRoadCenterLine, self.road_center_line_list)
        load_map_data(BUILDING_PATH, MapBuilding, self.building_list)
        load_map_data(LOT_PATH, MapLot, self.lot_list)
        load_map_data(GLA_PATH, MapLot, self.lot_list) # Append GLA to the same lot list
        load_map_data(ROAD_POLYGON_PATH, MapRoadPolygon, self.road_polygon_list)
        print("--- Data Loading Complete ---")
        print(f"Total Road Centerlines: {len(self.road_center_line_list)}")
        print(f"Total Road Polygons:    {len(self.road_polygon_list)}")
        print(f"Total Buildings:        {len(self.building_list)}")
        print(f"Total Lots (LOT+GLA):   {len(self.lot_list)}")

    def populate_data_model(self):
        """Populates the DataModel with DataElements for 3DM export."""
        print("--- Populating DataModel for 3DM Export ---")
        start_time = time.time()

        # Process Buildings
        print("Processing Buildings...")
        building_mat = DataElementMaterial(color="0xbbbec2") # Gray
        for i, building in enumerate(self.building_list):
            try:
                # Use height/start_height from data if available, else default
                height = getattr(building, 'height', 10) # Default height 10m if not present
                start_height = getattr(building, 'start_height', 0) # Default start 0m
                element = DataElement(geometry=building.geometry,
                                      layer="Buildings",
                                      material=building_mat,
                                      start_height=start_height,
                                      height=height if height and height > 0 else 10) # Ensure positive height
                self.model.insert_element(element)
            except Exception as e:
                print(f"Error creating DataElement for building {i}: {e}")
        print(f"Added {len(self.building_list)} buildings to model.")

        # Process Road Polygons
        print("Processing Road Polygons...")
        road_poly_mat = DataElementMaterial(color="0x404040") # Dark Gray
        for i, road_poly in enumerate(self.road_polygon_list):
             try:
                element = DataElement(geometry=road_poly.geometry,
                                      layer="Roads",
                                      material=road_poly_mat,
                                      start_height=-0.1, # Slightly below zero
                                      height=0.1) # Flat representation
                self.model.insert_element(element)
             except Exception as e:
                print(f"Error creating DataElement for road polygon {i}: {e}")
        print(f"Added {len(self.road_polygon_list)} road polygons to model.")

        # Process Lots (LOT and GLA)
        print("Processing Lots...")
        lot_mat = DataElementMaterial(color="0xced4d6") # Light Gray/Greenish
        gla_mat = DataElementMaterial(color="0xa0d6ce") # Different color for GLA maybe
        for i, lot in enumerate(self.lot_list):
            try:
                mat = gla_mat if getattr(lot, 'is_GLA', False) else lot_mat
                layer = "GLA" if getattr(lot, 'is_GLA', False) else "Lots"
                element = DataElement(geometry=lot.geometry,
                                      layer=layer,
                                      material=mat,
                                      start_height=-0.2, # Below roads
                                      height=0.1) # Flat representation
                self.model.insert_element(element)
            except Exception as e:
                print(f"Error creating DataElement for lot {i}: {e}")
        print(f"Added {len(self.lot_list)} lots (LOT+GLA) to model.")

        # Process Road Center Lines (Optional - represent as curves)
        # print("Processing Road Centerlines...")
        # road_line_mat = DataElementMaterial(color="0xFF0000") # Red
        # for i, road_line in enumerate(self.road_center_line_list):
        #     try:
        #         element = DataElement(geometry=road_line.geometry, # Should be LineString or MultiLineString
        #                               layer="RoadCenterlines",
        #                               material=road_line_mat,
        #                               start_height=0, # At grade
        #                               height=0) # No extrusion for lines
        #         self.model.insert_element(element)
        #     except Exception as e:
        #         print(f"Error creating DataElement for road centerline {i}: {e}")
        # print(f"Added {len(self.road_center_line_list)} road centerlines to model.")


        # Renew model (might be necessary for some internal updates in DataModel)
        self.model.renew()
        end_time = time.time()
        print(f"--- DataModel Population Complete --- Time: {end_time - start_time:.2f}s")


    def export_to_3dm(self, output_path: str):
        """Exports the populated DataModel to a 3DM file."""
        print(f"--- Starting 3DM Export to: {output_path} ---")
        start_time = time.time()
        try:
            # Crucial step: Use the ArtShapelyDataExchanger to write the file
            ARTShapelyDataExchanger.write_rhino_file(output_path, self.model)
            end_time = time.time()
            print(f"--- 3DM Export Successful --- Time: {end_time - start_time:.2f}s")
        except Exception as e:
            end_time = time.time()
            print(f"--- 3DM Export Failed --- Time: {end_time - start_time:.2f}s")
            print(f"Error: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("==============================================")
    print("=== Hong Kong Full Map to 3DM Exporter ===")
    print("==============================================")
    overall_start_time = time.time()

    exporter = FullMapExporter()

    # 1. Load all data from JSON files
    exporter.load_all_data()

    # 2. Populate the DataModel object
    exporter.populate_data_model()

    # 3. Export the DataModel to a 3DM file
    exporter.export_to_3dm(OUTPUT_3DM_PATH)

    overall_end_time = time.time()
    print("==============================================")
    print(f"Total Execution Time: {overall_end_time - overall_start_time:.2f} seconds")
    print("==============================================")