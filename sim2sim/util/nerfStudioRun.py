import numpy as np
import open3d as o3d
import os, subprocess
import json
import argparse


def preProcessData(type: str, input_path: str, output_path: str, xml_path: str):
    if type != "metashape":
        os.system(f"ns-process-data {type} --data {input_path} --output-dir {output_path} --num-downscales 1")
    else:
        os.system(f"ns-process-data metashape --data {input_path} --xml {xml_path} --output-dir {output_path}")


def runNerfStudio(algo_type: str, input_path: str):
    os.system(f"ns-train {algo_type} --data {input_path}")


def generateMesh():
    # ns-export poisson --load-config outputs/data-test_data/nerfacto/2022-12-11_043731/config.yml --output-dir exports/mesh/ --target-num-faces 50000 --num-pixels-per-side 2048 --normal-output-name normals --num-points 1000000 --remove-outliers True --use-bounding-box True --bounding-box-min -1 -1 -1 --bounding-box-max 1 1 1
    pass


def generaatePointCloud():
    # ns-export pointcloud --load-config outputs/data-test_data/nerfacto/2022-12-11_043731/config.yml --output-dir exports/pcd/ --num-points 1000000 --remove-outliers True --estimate-normals False --use-bounding-box True --bounding-box-min -1 -1 -1 --bounding-box-max 1 1 1
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run NerfStudio to generate meshes", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-p", "--preprocess", action="store_true", help="preprocess data to NeRF Studio format")
    parser.add_argument(
        "-t", "--type", type=str, help="data type for preprocess (video,images,polycam,insta360,record3d)"
    )
    parser.add_argument("-s", "--source", type=str, help="source folder path for preprocess")
    parser.add_argument("-d", "--dest", type=str, help="destination folder path for preprocess")
    parser.add_argument("-x", "--xml", type=str, help="xml file for metashape preprocessing")
    parser.add_argument(
        "-a",
        "--algo",
        type=str,
        help="algo for NeRF Generation (nerfacto,instant-ngp,mipnerf,semantic-nerfw,vanilla-nerf,tensorf,dnerf,phototourism)",
    )
    parser.add_argument("-i", "--input", help="input data Spath for NeRF Generation")
    args = parser.parse_args()
    config = vars(args)

    if config["preprocess"]:
        preProcessData(config["type"], config["source"], config["dest"], config["xml"])

    runNerfStudio(config["algo"], config["input"])
