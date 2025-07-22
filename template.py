import os
from pathlib import Path
import logging

# create logging string
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(levelname)s - %(message)s",
    handlers= [
        logging.FileHandler("Domain_Name.log"),
        logging.StreamHandler()
    ]
)

list_of_files = [
    ".github/workflows/.gitkeep",
    "data/.gitkeep",
    "api/.gitkeep",
    "evaluation/.gitkeep",
    "models/.gitkeep",
    "notebooks/.gitkeep",
    "reports/.gitkeep",
    "README.md",
    "requirements.txt"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty files: {filepath}")
    else:
        logging.info(f"{filename}: exists")