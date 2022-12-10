import numpy as np
import open3d as o3d
import os, subprocess
import json


def preProcessImages(input_path, output_path):
    os.system(f"ns-process-data images --data {input_path} --output-dir {output_path}")


def preProcessVideo(input_path, output_path):
    os.system(f"ns-process-data video --data {input_path} --output-dir {output_path}")


def preProcessPolycam(input_path, output_path):
    os.system(f"ns-process-data polycam --data {input_path} --output-dir {output_path}")


def preProcessInsta360(input_path, output_path):
    os.system(f"ns-process-data insta360 --data {input_path} --output-dir {output_path}")


def preProcessRecord3d(input_path, output_path):
    os.system(f"ns-process-data record3d --data {input_path} --output-dir {output_path}")


def preProcessMetashape(input_path, output_path, xml_path):
    os.system(f"ns-process-data record3d --data {input_path} --xml {xml_path} --output-dir {output_path}")


if __name__ == "__main__":
    main()
