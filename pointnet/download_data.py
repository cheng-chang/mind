import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "data")
MODELNET40_URL = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
ZIP_FILE = os.path.basename(MODELNET40_URL)

CMD = [
  "wget {}".format(MODELNET40_URL),
  "unzip {}".format(ZIP_FILE),
  "mv {} {}".format(ZIP_FILE[:-4], DATA_DIR),
  "rm {}".format(ZIP_FILE)
]

if not os.path.exists(DATA_DIR):
  os.mkdir(DATA_DIR)

for cmd in CMD:
  os.system(cmd)
