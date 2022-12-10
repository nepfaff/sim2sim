import numpy as np
import open3d as o3d
import os, subprocess
import json


def preProcessData(type: str, input_path: str, output_path: str, xml_path: str):
    if type is not "metashape":
        os.system(f"ns-process-data {type} --data {input_path} --output-dir {output_path}")
    else:
        os.system(f"ns-process-data metashape --data {input_path} --xml {xml_path} --output-dir {output_path}")


if __name__ == "__main__":
    main()
