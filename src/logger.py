import logging
import os 
from datetime import datetime

# definig formate of the file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# crating a path from current working dir to main folder (logs) to inner folder (LOG_FILE)
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# now creating the dir with the path
os.makedirs(logs_path, exist_ok=True)

# now joing the path of dir with file 
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# defining log configration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)