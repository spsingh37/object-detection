import os
import xml.etree.ElementTree as ET

def validate_xml_annotations(xml_dir):
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)
                if xmin >= xmax or ymin >= ymax:
                    print(f"Invalid box in {xml_file}: {xmin}, {ymin}, {xmax}, {ymax}")

validate_xml_annotations("Annotations")

