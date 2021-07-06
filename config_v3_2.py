import os
INITIAL_FRAME_RATE = 10
JPEG_QUALITY = 20


EXPLORE_INTERVAL=1
PIPELINE_UPDATE_PRECISION=0.1

Q_READ_TIMEOUT = 10000
INPUT_VIDEO = 0#"test_video40.avi"#0#"/Users/davide/Google Drive/University/PhD/19_sigcom_w/GOPR9840.MP4"#0#"test_video40.avi"
MAX_IMG_QUEUE = 10
SAVE_IMG = False
SHOW_IMG = False
DAEMONIAC_THREADS = True
CONNECTION_STRING = "/dev/ttyUSB0"#"127.0.0.1:14550"

# Variables for detection -> movement
MAX_IMG_QUEUE = 3
X_CUTOFF = 0.1
Y_CUTOFF = 0.1
AREA_CUTOFF = 1000
FORWARD = 1
BACKWARD = -1
RIGHT = 1
LEFT = -1
UP = 1
DOWN = -1
TARGET_AREA = 10000

ALPHA = 0.1
MOVEMENT_DURATION = 0.4

BUFF_SIZE = 1024
MODEL_PATH = "models"

MESSAGE = "Oh bella ciao"
Q_READ_TIMEOUT = 10000
DEVICE_STATE_UPDATE = 0.5

UDP_DISCOVERY_PORT = 5006
INITIAL_TCP_PORT = 12001
BUFF_SIZE = 1024
JPEG = True

LOGS_PATH = "logs"
BOOKKEEPER_path = os.path.join(LOGS_PATH, "_tenboom.csv")
FLIGHTLOG_NAME_path = os.path.join(LOGS_PATH, "FlightLog_{}.csv")
DEVICE_LOG_path = os.path.join(LOGS_PATH, "bubi_{}.csv")
BOOKKEEPER = "_tenboom.csv"
FLIGHTLOG_NAME = "FlightLog_{}.csv"
DEVICE_LOG = "bubi_{}.csv"

DELTA_E = 0.3
DELTA_L = 0.7

models = ["ssd_mobilenet_v1_coco_2018_01_28",
          "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03",
          "ssd_mobilenet_v2_coco_2018_03_29",
          "ssdlite_mobilenet_v2_coco_2018_05_09",
          "ssdlite_mobilenet_v2_coco_FP32_50_trt.pb"]

DECISION_POLICY = "energy_saving"
ENERGY_SAVING_THR = 0.25
VERBOSE = False