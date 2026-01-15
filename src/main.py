import logging
import os
from read_file import generate_matrix
from segmentation import cellpose_segmentation, compare_spectral_methods, stardist_segmentation
from utils import config_logging

config_logging()
logger = logging.getLogger(__name__)

file = "corte1"

file_path = os.path.join("data", f"{file}.tif")
tabela_celulas = generate_matrix(file_path, output_filename=f"{file}_matriz.csv")

cellpose_segmentation(file_path)
compare_spectral_methods(file_path)