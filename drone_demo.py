import time
from queue import LifoQueue as Queue
import data_types
from blocks_v3_3 import Logger, ImageProducer, ConsumeImageProduceFeatDet, ConsumeFeatProduceAction, ConsumeAction, Advertizer, TegraLogger, NetLogger
from config_v3_2 import *
import dronekit
# from vehicle_class_file import MyVehicle
from FakeDrone import FakeDrone
from SuperDrone import *
from FlightLog import *
from threading import Thread
import argparse
import numpy as np
import copy


class Device(Thread):
  def __init__(self, iid, start_time=time.time(), connection_string=None, input_pics="zombies", solo_edge=False, adaptive_fr = False, adaptive_pipes=True, is_nano=False, model="", ismac=False):
    Thread.__init__(self)
    self.id = iid
    self.start_time = start_time
    self.modules = {}
    self.outfile = open(os.path.join(LOGS_PATH, DEVICE_LOG.format(start_time)), "w+")
    self.queues = {"img": Queue(maxsize=1), "feat": Queue(maxsize=1), "action": Queue(maxsize=1)}
    self.state = {
      "info": Logger(name="timing", outfile = self.outfile, category="flux_point", start=self.start_time, cats=["id", "from/to", "type", "auth"]),
      "dev_id": self.id,
      "is_running": True,
      "mode": "explore",
      "start_time": self.start_time,
      "sample_options": {"camera": {"intra_frame": 1/float(INITIAL_FRAME_RATE), "prec": 0.1}},
      "active_pipelines": {"": True, self.id: True},
      "pipelines_log": {self.id: []},
      "queues": self.queues,
      "img_out_q": [self.queues["img"], ],
      "adapt_fr": adaptive_fr,
      "adapt_pipes": adaptive_pipes,
      "input": input_pics,
      "ismac": ismac,
      "model": model if len(model) > 0 else "ssd_mobilenet_v1_coco_2018_01_28",
      }
    self.modules["advert"] = Advertizer(state=self.state)
    if is_nano:
       print("starting tegrastats")
       self.modules["tegrastats"] = TegraLogger(state=self.state, freq=10)
       print("starting netstats")
       self.modules["netstats"] = NetLogger(state=self.state, freq=10)
    self.modules["camera"] = ImageProducer(out_q=self.state["img_out_q"], state=self.state)
    if not solo_edge:
      self.modules["img2feat"] = ConsumeImageProduceFeatDet(in_q=self.queues["img"], 
                        out_q=[self.queues["feat"], ], state=self.state, pipe=self.id)
      self.modules["feat2act"] = ConsumeFeatProduceAction(in_q=self.queues["feat"], 
                        out_q=[self.queues["action"], ], state=self.state, pipe=self.id)
    if connection_string:  
      self.drone = dronekit.connect(connection_string, baud=57600, wait_ready=True, vehicle_class=SuperDrone)
    else:
      self.drone = FakeDrone()
    self.modules["act"] = ConsumeAction(in_q=self.queues["action"], drone=self.drone, state=self.state)
    # self.modules["keyboard"] = Keyboard(idrone=drone, istate=s, state=self.state)
    self.start_explore = time.time()
    self.last_counted = {}

  def run(self):
    print("IN RUN")
    count = 0
    self.drone.arm_and_takeoff(10)
    for t_name in self.modules:
      print("Starting {}".format(t_name))
      self.modules[t_name].setDaemon(True)
      self.modules[t_name].start()
      print("Started " + t_name)
    print("Device Started")
    self.state["mode"] = "explore"
    self.activate_edges()
    self.activate_local()
    while self.state["is_running"]:
      # if count%5 == 4:
      tmp = [len(self.state["pipelines_log"][k]) for k in self.state["pipelines_log"]]
      self.clean_logs(since=time.time() - 1)
      tmp1 = [len(self.state["pipelines_log"][k]) for k in self.state["pipelines_log"]]
      if VERBOSE:
        print(tmp, '\n', tmp1)
      self.pipeline_update()
      # if "explore" in self.state["mode"]:
      #   if self.start_explore + EXPLORE_INTERVAL < time.time() and self.state["adapt_pipes"]:
      #     self.pipeline_update()
      #   else:
      #     pass
      # else:
      #   if self.state["adapt_pipes"]:
      #     self.pipeline_update()
      time.sleep(PIPELINE_UPDATE_PRECISION)
      # print("STATE: {}".format(self.state["mode"]))
      count += 1
      
  def pipeline_update(self):
    if self.state["adapt_pipes"]:
      #print("START UPDATE PIPELINES: {}".format(self.state["mode"]))
      #print([e for e in self.state["active_pipelines"]])
      if self.state["mode"] == self.id:
        self.mode["mode"] = "explore"
        self.activate_edges()
        return
      else:
        curr_means, curr_max = [], []
        curr_max_dict = {}
        ## no need to reset it: it keeps last K examples
        pipelines_log = copy.deepcopy(self.state["pipelines_log"])
        for k in pipelines_log:
          if k not in self.last_counted:
            self.last_counted[k] = {"time": 0, "value": 0}
          if pipelines_log[k] and k != self.id:
            curr_means.append((np.mean([pipelines_log[k][i][1] for i in range(len(pipelines_log[k]))]), k))
            curr_max.append((np.max([pipelines_log[k][i][1] for i in range(len(pipelines_log[k]))]), k))
            curr_max_dict[k] = np.max([pipelines_log[k][i][1] for i in range(len(pipelines_log[k]))])
            for i in range(len(pipelines_log[k])):
              if pipelines_log[k][i][0] > self.last_counted[k]["time"]:
                if pipelines_log[k][i][1] < ENERGY_SAVING_THR:
                  self.last_counted[k]["value"] = 0
                else:
                  self.last_counted[k]["value"] += 1
          else:
            curr_means.append((float("inf"), k))
            curr_max.append((float("inf"), k))
        if DECISION_POLICY == "double_thr":
          min_max = min(curr_max)
          curr_best = min(curr_means)
          #print(pipelines_log)
          if min_max[0] <= DELTA_E:
            self.state["mode"] = "performance"
            self.deactivate_local()
            self.deactivate_edges_but(curr_best[1])
          elif DELTA_E < min_max[0] <= DELTA_L:
            self.state["mode"] = "explore_edges"
            self.deactivate_local()
            self.activate_edges()
            self.start_explore = time.time()
          elif min_max[0] > DELTA_E:
            self.state["mode"] = "explore"
            self.activate_local()
            self.activate_edges()
            self.start_explore = time.time()
        
        elif DECISION_POLICY == "energy_saving":
          if VERBOSE:
            print(curr_means)
          curr_best = min(curr_means)
          if curr_best[0] > 1:
            self.activate_local()
            self.activate_edges()
            self.state["mode"] = "explore"
            return
          curr_best_max = curr_max_dict[curr_best[1]]
          if VERBOSE:
            print("curr_max_dict")
          for k in curr_max_dict:
            if VERBOSE:
              print("{} : {}".format(k, curr_max_dict[k]))
          if curr_best_max < ENERGY_SAVING_THR:
            self.deactivate_local()
            self.deactivate_edges_but(curr_best[1])
            self.state["mode"] = "performance"
          else:
            self.activate_edges()
            self.state["mode"] = "explore_edges"
            if self.last_counted[curr_best[1]]["value"] > 5:
              self.activate_local()
              self.activate_edges()
              self.state["mode"] = "explore"
    else:
      pass

  def activate_edges(self):
    for k in self.state["active_pipelines"]:
      if "EDGE" in k:
        self.state["active_pipelines"][k] = True

  def deactivate_edges(self):
    for k in self.state["active_pipelines"]:
      if "EDGE" in k:
        self.state["active_pipelines"][k] = False

  def deactivate_edges_but(self, pipe):
    for k in self.state["active_pipelines"]:
      if "EDGE" in k and k != pipe:
        self.state["active_pipelines"][k] = False
  
  def activate_local(self):
    self.state["active_pipelines"][self.id] = True

  def deactivate_local(self):
    self.state["active_pipelines"][self.id] = False

  def clean_logs(self, number = -1, since= -1):
    if since > 0: 
      for p in self.state["pipelines_log"]:
        for i in range(len(self.state["pipelines_log"][p])):
          try:
            if self.state["pipelines_log"][p][i][0] < since:
              del self.state["pipelines_log"][p][i]
          except:
            print(self.state["pipelines_log"])
          # if self.state["pipelines_log"][p][len(self.state["pipelines_log"][p]) - i -1][0] < since:
            # self.state["pipelines_log"][p] = self.state["pipelines_log"][p][len(self.state["pipelines_log"][p]) - i:]
      return
    if number > 0:
      for p in self.state["pipelines_log"]:
        self.state["pipelines_log"][p] = self.state["pipelines_log"][p][number:]
      return
    raise ValueError("Cleaning logs needs a rule")

  def stop_module(self, module_name):
    self.modules[module_name].stop()

  def is_running(self,):
    return self.state["is_running"]
  
  def stop(self):
    self.state["info"].save()
    self.outfile.close()
    for m in self.modules:
      self.stop_module(m)

def get_argparse():
  parser = argparse.ArgumentParser()
  parser.add_argument("--c", help="connection string to connect to the drone", 
        default=None)
  parser.add_argument("--r", help="", default=1)
  parser.add_argument("--m", help="", default=10)
  parser.add_argument("--s", help="", default=1)
  # parser.add_argument("--real", help="Deactivates all the debugging conveniencies", action="store_true")
  parser.add_argument("--solo_edge", action="store_true")
  parser.add_argument("--tegra", action="store_true")
  parser.add_argument("--fly", help="Is it actually connected to a drone?", action="store_true")
  parser.add_argument("--name", help="", default="UAV01")
  parser.add_argument("--info", help="Should characterize the experiment", default="Just another experiment")
  parser.add_argument("--adaptive_fr", "-afr", action="store_true")
  parser.add_argument("--adaptive_pipes", "-apip", action="store_true")
  parser.add_argument("--move", action="store_true")
  parser.add_argument("--model", default=2)
  parser.add_argument("--mac", action="store_true")
  parser.add_argument("--input", default=None, help="Input source: 0 - camera, zombies - image")


  # parser.add_argument("port")
  args = parser.parse_args()
  return args
if __name__ == "__main__":
  args = get_argparse()
  try:
    d = Device("UAV01", connection_string=args.c, input_pics=0)
    d.start()
    while True:
      time.sleep(DEVICE_STATE_UPDATE)
  except Exception as e:
    raise(e)
  finally:
    d.stop()
