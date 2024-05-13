import ezdxf

# Load the DXF file
doc = ezdxf.readfile('figure_2_0.dxf')

# Extract road width information
road_widths = {}
for entity in doc.modelspace():
    if entity.dxftype() == 'LINE':
        start_point = entity.dxf.start
        end_point = entity.dxf.end
        width = abs(start_point[0] - end_point[0])
        road_widths[entity] = width

# Print road widths
for entity, width in road_widths.items():
    print(f"Road ID: {entity.dxf.handle}, Width: {width}")

# Save the modified DXF file
doc.saveas('output.dxf')