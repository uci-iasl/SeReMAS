import time
import cv2
from queue import LifoQueue as Queue
from queue import Empty
import data_types
import datetime
from threading import Thread
from inspect import getframeinfo, stack
from config_v3_2 import *
import socket
import pickle
import _thread
import net_p3
import os
from blocks_detection import Model
import numpy as np
import subprocess
from statistics import mean
import tensorflow as tf

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
# emergency_outfile = open("logs/emergency{}.csv".format(time.time()), "w")

class Module(Thread):
  def __init__(self, in_q, out_q, state, pipe=""):
    Thread.__init__(self)
    self.in_q = in_q
    self.out_q = out_q
    self.pipe=pipe
    self.dev_id = state["dev_id"]
    # self.log = state["monitor"]
    self.info = state["info"]
    self.dev_is_running = state["is_running"]
    self.is_running = True
    self.state = state
    self.pipe = pipe
    self.last_put = {}

  def stop(self):
    self.is_running = False


class ImageProducer(Module):
  def __init__(self, out_q, state, pipe=""):
    Module.__init__(self, None, out_q, state, pipe)
    if state["input"] == "zombies":
      self.capture = cv2.imread("data/6zombies_small.jpg")
    else:
      self.capture = cv2.VideoCapture(INPUT_VIDEO)
      self.capture.set(3,400)
      self.capture.set(4,300)
      self.last_taken = self.capture.read()[1]
    # self.capture.set(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO, 0)
    self.setName("Capturer")

  def run(self):
    # self.log.add(category="img_prod", message="{}".format(time.time()))
    count = 0
    self.is_running = True
    while self.is_running: # and self.dev_is_running: # and self.pipe in self.state["active_pipelines"]:
      time_frame_taken = time.time()
      if self.state["input"] == "zombies" or isinstance(self.capture, np.ndarray):
        ret, image = True, self.capture
      else:
        ret, image = self.capture.read()
      if ret:
        # print("Img taken")
        for q in self.state["img_out_q"]:
          while q.qsize() >= MAX_IMG_QUEUE:
            q.get()
            q.task_done()
        st = time.time()
        # image = cv2.resize(image, (400, 300))
        if JPEG:
          result, image = cv2.imencode('.jpg', image, encode_param)
        img_taken = data_types.Image(image=image, t_frame=time.time(), frame_auth=self.dev_id)
        # self.info.add(category="img_prod", fields={"id": img_taken.t_frame, "from/to": "img_prod", "type": "img", "auth": self.dev_id})
        # print("blocks 3.2 line 69")
        self.info.save_action(img_taken)
        if VERBOSE:
          print(self.out_q)
        for q in self.out_q:
          if q not in self.last_put:
            self.last_put[q] = -1
          item = None
          to_put = img_taken
          try:
            item = q.get(False)
            if item:
              q.task_done()
              if to_put.t_frame < item.t_frame:
                to_put = item
          except Empty as e:
            pass
          except Exception as e:
            raise e
          finally:
            if to_put.t_frame > self.last_put[q]:
              q.put(to_put)
              self.last_put[q] = to_put.t_frame
              # print("Img producer {}, {}".format(q, to_put.t_frame))
            else:
              print("IMG producer NO PUT!!!", time.time(), self.last_put , to_put.t_frame)
      # print("t = {0:.2f} : Img taken".format(time.time()-self.state["start_time"]))
      time_before_next_frame = self.state["sample_options"]["camera"]["intra_frame"] + time_frame_taken - time.time()
      time.sleep(max(time_before_next_frame, 0))
    print("Img called stop")
    self.stop()

class ConsumeImageProduceFeatDet(Module):
  def __init__(self, in_q, out_q, state, pipe):
    Module.__init__(self, in_q, out_q, state, pipe)
    self.setName("Data Analysis")
    print("Loading model {}".format(self.state["model"]))
    self.detector = Model(self.state["model"])
    self.detector.predict(self.detector.load_image_into_numpy_array_cv('data/6zombies_small.jpg'))
    self.artificial_delay = 0.6 if "UAV" in self.dev_id else 0

  def run(self):
    while self.is_running and self.dev_is_running:
      try:
        image = self.in_q.get(timeout=Q_READ_TIMEOUT)
        if VERBOSE:
          print(time.time(), "PICK UP DDN", image.t_frame)
        while self.in_q.qsize() >= MAX_IMG_QUEUE:
          tmp = self.in_q.get()
          if VERBOSE:
            print(time.time(), "PICK UP DDN", image.t_frame)
          if (image.t_frame < tmp.t_frame):
            image = tmp
          self.in_q.task_done()
      except Empty as e:
        if VERBOSE:
          (self.getName() + " - Timeout expired")
          print(e)
        self.is_running = False
      else:
        self.in_q.task_done()
        if self.state["active_pipelines"][self.pipe] or not self.state["adapt_pipes"]:
          image_exp = image
          if JPEG:
            image_exp = cv2.imdecode(image.get_value(), 1)
          bbox = self.detector.predict(np.expand_dims(image_exp, axis=0), 'person')
          time.sleep(self.artificial_delay)
          if bbox:
            x1, y1, x2, y2 = bbox[0][0] #first is example, second is bbox
            pos_img = ((x1+x2)/2)-0.5, ((y1+y2)/2)-0.5, ((x2-x1)*(y2-y1))
          else:
            pos_img = 0.5,0.5,0
          feat_inst = data_types.Detection(x=pos_img[0], y=pos_img[1], area=pos_img[2], t_frame=image.get_time(),
                                           frame_auth=image.get_auth(), t_det=time.time(), det_auth=self.dev_id)
          print(feat_inst)
        for q in self.out_q:
          if q not in self.last_put:
            self.last_put[q] = -1
          item = None
          to_put = feat_inst
          try:
            item = q.get(False)
            if item:
              q.task_done()
              if to_put.t_frame < item.t_frame:
                to_put = item
          except:
            pass
          finally:
            if to_put.t_frame > self.last_put[q]:
              if VERBOSE:
                print(time.time(), "PUT DOWN DNN", to_put.t_frame)
              q.put(to_put)
              self.last_put[q] = to_put.t_frame
              if VERBOSE:
                print("Feat production PUT")
            else:
              if VERBOSE:
                print(time.time(), self.last_put , to_put.t_frame)


class ConsumeFeatProduceAction(Module):
  def __init__(self, in_q, out_q, state, pipe):
    Module.__init__(self, in_q, out_q, state, pipe)
    self.setName("Decision Maker")
    print("Consume Feat")

  def run(self):
    while self.dev_is_running:
      try:
        if VERBOSE:
          print("GET in Prod action")
        feat = self.in_q.get(timeout=Q_READ_TIMEOUT)

        if VERBOSE:
          print(time.time(), "PICK UP FEAT", feat.t_frame)
        # self.info.save_action(feat)
      except Empty as e:
        print(self.getName() + " - Timeout expired")
        print(e)
        self.stop()
      else:
        self.in_q.task_done()
        action = self.detection2movement(feat)
        print(action)
        if VERBOSE:
          print("action produced")
        if self.is_running:
          if VERBOSE:
            print(self.out_q)
          for q in self.out_q:
            if q not in self.last_put:
              self.last_put[q] = -1
            item = None
            to_put = action
            try:
              item = q.get(False)
              if item:
                if VERBOSE:
                  print("DISCARD ACT")
                q.task_done()
                if to_put.t_frame < item.t_frame:
                  to_put = item
            except:
              if VERBOSE:
                print(q)
            finally:
              if VERBOSE:
                print("in finally line 204")
              if to_put.t_frame > self.last_put[q]:
                if VERBOSE:
                  print(time.time(), "PUT DOWN ACT", feat.t_frame)
                q.put(to_put)
                self.last_put[q] = to_put.t_frame
                if VERBOSE:
                  print("Produce action PUT")
              else:
                if VERBOSE:
                  print("DISCARD ACT")
                  print(time.time(), self.last_put , to_put.t_frame)

  ## Note: here we convert from 2D picture, to the 3D world.
  ## In this case, we pass LR to x, FB to y, UD to z                  # ",".join([str(pos) for pos in [feat.x, feat.y]]) + '\n')
  def detection2movement(self, d):
    lr_action, ud_action, fb_action = 0, 0, 0
    if abs(d.x) > X_CUTOFF:  # does it make sense to move?
        if d.x > 0:  # target on my left?
            lr_action = RIGHT
        else:
            lr_action = LEFT
    else:
        pass  # not worth of it
    if abs(d.y) > Y_CUTOFF:
        if d.y > 0:  # target up?
            ud_action = UP
        else:
            ud_action = DOWN
    else:
        pass  # not worth of it
    if abs((d.area - TARGET_AREA) / TARGET_AREA) > AREA_CUTOFF:
        if (d.area - TARGET_AREA) > 0:
            fb_action = BACKWARD
        else:
            fb_action = FORWARD
    else: pass  # not worth moving

    ##DISABLE Z AXIS!!!!#########
    ud_action = 0
    fb_action = 0
    ##DISABLE Z AXIS!!!!#########
    
    action = data_types.Action(x_act=lr_action, y_act=fb_action, z_act=ud_action, t_frame=d.t_frame,frame_auth=d.frame_auth, t_det=d.t_det, det_auth=d.det_auth, t_act=time.time(), act_auth=self.dev_id)
    return action


class ConsumeAction(Module):
  def __init__(self, in_q, state, drone, pipe=""):
    Module.__init__(self, in_q, [], state, pipe)
    self.setName("Operator")
    self.last_act = None
    self.drone = drone
    self.mov_dur=MOVEMENT_DURATION

  def run(self):
    while self.is_running and self.dev_is_running:
      action_list = []
      action2take = None
      try:
        action_list.append(self.in_q.get(timeout=Q_READ_TIMEOUT))
        self.in_q.task_done()
        while not self.in_q.empty():
          action_list.append(self.in_q.get(timeout=Q_READ_TIMEOUT))
          self.in_q.task_done()
      except Empty as e:
        if VERBOSE:
          print(self.getName() + " - Timeout expired")
          print(e)
          print("consume action stopping")
        self.stop()
      else:
        action2take = self.choose_action(action_list, self.last_act)
        for e in action_list:
          self.state["pipelines_log"][e.det_auth].append((e.t_frame, time.time() - e.t_frame))
          if e is action2take:
            self.info.save_action(e, True, time.time())
          else:
            self.info.save_action(e, False, time.time())

      if action2take is not None:
        action2take.t_act = time.time()
        info2print = [str(e) for e in [action2take.det_auth, action2take.t_frame, action2take.t_act - action2take.t_frame]]
        info2print.append(str("-")) if self.last_act is None else info2print.append(str(action2take.t_act - self.last_act.t_act))
        self.last_act = action2take
        if self.state["adapt_fr"]:
          self.state["sample_options"]["camera"]["intra_frame"] = (time.time() - action2take.t_frame)*ALPHA + (1-ALPHA)*self.state["sample_options"]["camera"]["intra_frame"]
          self.mov_dur = self.mov_dur
        if VERBOSE:
          print("MOVE {} {}".format(action2take.det_auth, action2take.t_act - action2take.t_frame))
        self.drone.move(action2take.x_act, action2take.y_act, action2take.z_act, duration=self.mov_dur) #self.state["sample_options"]["camera"]["intra_frame"]/2

  def choose_action(self, list_of_actions, last_action_taken):
    ret = None
    s = "List of actions: \n"
    for e in list_of_actions:
      if VERBOSE:
        print("ACTION: {} {}  ---only lab--  {}".format(e.det_auth, e.t_act - e.t_frame, e.t_det - e.t_frame))
      s += str(e) + '\n'
      if ret is not None:
          if ret.t_frame < e.t_frame:
              ret = e
      else:
          ret = e
    if last_action_taken is not None:
      if not (int(last_action_taken.t_frame*100) < int(ret.t_frame*100)):
        return None
    s += "The chosen one\n" + str(ret)
    # print(s)
    # self.info.add(category='action', message="{}".format(s))
    return ret


class Logger:
  def __init__(self, name, outfile=None, category=None, start=time.time(), cats=["message", ], s = None):
    self.name = name
    self.info = []
    self.start_time = start
    if not os.path.isdir(LOG_PATH):
      os.mkdir(LOG_PATH)
    if outfile is None:
      outfile = open(os.path.join(LOGS_PATH, "logger_outfile_NONE_{}.csv".format(start)), "w")
    self.outfile = outfile
    if category is None:
      category = "category"
    self.header = ["time", "lineno", category] + cats + ["taken, capt-to-action"]
    # print("self.header", self.name, self.header)
    self.cats = cats
    self.state = s

  def add(self, category, fields):
    caller = getframeinfo(stack()[1][0])
    # print("fields", fields)
    self.info.append([time.time()-self.start_time, caller.lineno, category] + [fields[e] for e in self.cats])

  def save_action(self, act, taken=False, time_taken=None):
    if VERBOSE:
      print("writing")
    extras = ", {}, {}".format(1 if taken else 0, time.time()-act.t_frame if time_taken is None else time_taken-act.t_frame)
    # print(act.nice_str() + extras)
    self.outfile.write(act.nice_str() + extras + "\n")
    self.outfile.flush()
  
  def __str__(self):
    s = ",".join(self.header) + "\n"
    for lst in self.info:
      s += ",".join([str(e) for e in lst]) + "\n"
    return s

  def save(self, filename=None):
    if filename is None:
      filename = os.path.join(LOGS_PATH, "logger_{}_{:0.2f}.csv".format(self.name, self.start_time))
    with open(filename, "w") as f:
      f.write(self.__str__())
      f.close()


class Advertizer(Thread):
  def __init__(self, state):
    Thread.__init__(self)
    self.state=state
    self.ips = []
    if state["ismac"]:
      self.ips = ['127.0.0.1']
    else:
      self.ips = (subprocess.check_output(['hostname', '--all-ip-addresses'])).decode().strip().split()
    self.discovery_socks = []
    for e in self.ips:
      tmp = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
      tmp.bind((e,0))
      tmp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
      self.discovery_socks.append(tmp)
    print("Initiated sockets on {}".format([e.getsockname() for e in self.discovery_socks])) 

  def run(self):
    # while True:
    while True:
      for sock in self.discovery_socks:
        try:
          if self.state["ismac"]:
            sock.sendto(pickle.dumps(MESSAGE), ("", UDP_DISCOVERY_PORT))
          else:
            sock.sendto(pickle.dumps(MESSAGE), ("<broadcast>", UDP_DISCOVERY_PORT))
          sock.settimeout(2)
          data, addr = sock.recvfrom(1024)
          rcvd_obj = pickle.loads(data)
          if VERBOSE:
            print("client received - {}".format(rcvd_obj))
          if rcvd_obj["name"] not in self.state["active_pipelines"]:
            self.add_edge_pipeline(rcvd_obj["name"])
            c = Connector(ip=rcvd_obj["ip"], port=rcvd_obj["port"], pipe=rcvd_obj["name"], state=self.state)
            c.start()
        except socket.timeout as e:
          if VERBOSE:
            print("Socket on {} timed out".format(sock.getsockname()[0]))

  def stop(self):
    for sock in self.discovery_socks:
      sock.close()

  def update_ips(self):
    # Keep in self.ips the ones for which we already have a socket open
    # Open new sockets for the new ones
    if state["ismac"]:
      current_ips = ['127.0.0.1']
    else:
      current_ips = (subprocess.check_output(['hostname', '--all-ip-addresses'])).decode().strip().split()
    self.discovery_socks = []
    new_ips = [ip for ip in current_ips if ip not in self.ips]
    for e in ips:
      tmp = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
      tmp.bind((e,0))
      tmp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
      self.discovery_socks.append(tmp)
    print("Initiated sockets on {}".format([e.getsockname() for e in self.discovery_socks])) 

  def add_edge_pipeline(self, pipe):
    self.state["active_pipelines"][pipe] = True
    self.state["pipelines_log"][pipe] = []
    self.state["queues"]["{}_in".format(pipe)] = Queue()
    self.state["queues"]["{}_out".format(pipe)] = Queue()
    self.state["img_out_q"].append(self.state["queues"]["{}_in".format(pipe)])
    # print("-> Added q to self.state[img_out_q]")
    # for e in self.state["img_out_q"]:
    #   print(e.qsize())


class Connector(Module):
  def __init__(self, ip, port, state, pipe, in_q=None, out_q=None):
    Module.__init__(self, in_q=state["queues"]["{}_in".format(pipe)], out_q=state["queues"]["action"], state=state, pipe=pipe)
    # Create a TCP/IP socket
    self.ip = ip
    # if pipe in "active_pipelines":
      # raise ValueError("{} already an active pipeline".format((name, self)))
    # self.state["active_pipelines"].append((pipe, None, None))
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.buff_size = BUFF_SIZE
    # Connect the socket to the port where the server is listening
    server_address = (ip, port)
    # print('CLIENT ---> connecting to {} port {}'.format(*server_address))
    time.sleep(0.1)
    self.sock.connect(server_address)
    self.pipe = pipe
    # print("After connection")
    _thread.start_new_thread(self.ready_to_reply, ())

  def run(self):
    rem = b""
    while True:
      try:
        if VERBOSE:
          print("Waiting to receive a message")
        t1, t2, data, rem = net_p3.receive_fat_msg(self.sock, rem)
        # print("Received stuff - putting it in the q")
        if VERBOSE:
          print(time.time(), "Message received")
        if self.state["active_pipelines"][self.pipe] or not self.state["adapt_pipes"]:
          if self.state["queues"]["action"] not in self.last_put:
            self.last_put[self.state["queues"]["action"]] = -1
          tmp = pickle.loads(data)
          if VERBOSE:
            print(time.time(), "Message received", tmp.t_frame, self.sock.getsockname())
          item = None
          to_put = tmp
          try:
            item = self.state["queues"]["action"].get(False)
            if item:
              self.state["queues"]["action"].task_done()
              if to_put.t_frame < item.t_frame:
                to_put = item
            if item:
              self.state["queues"]["action"].task_done()
          except:
            # print("_"*80)
            pass
          finally:
            if VERBOSE:
              print("Connector PUT")
            if to_put.t_frame > self.last_put[self.state["queues"]["action"]]:
              self.state["queues"]["action"].put(to_put)
              self.last_put[self.state["queues"]["action"]] = to_put.t_frame
            else:
              if VERBOSE:
                print(time.time(), self.last_put , to_put.t_frame)
            self.info.add(category="put_out_q", fields={"id": to_put.t_frame, "from/to": "connector", "type": "action", "auth": self.dev_id})
      except Exception as e:
        if VERBOSE:
          print("_"*80)
        del self.state["active_pipelines"][self.pipe]
        del self.state["pipelines_log"][self.pipe]
        raise e

  def ready_to_reply(self):
    while self.is_running:
      try:
        self.sock.setblocking(1)
        if VERBOSE:
          print("Waiting get at ready_to_reply")
        image = self.in_q.get(timeout=Q_READ_TIMEOUT)
        if VERBOSE:
          print("Got at ready_to_reply")
          print(time.time(), "Gotten Image ", image.t_frame, [str(e) for e in self.sock.getsockname()])
        # print(self.sock.getsockname())
        while self.in_q.qsize() > MAX_IMG_QUEUE:
          tmp = self.in_q.get()
          if (image.t_frame < tmp.t_frame):
            image = tmp
          self.in_q.task_done()
        self.info.add(category="get_in_q", fields={"id": image.t_frame, "from/to": "replying", "type": "img", "auth": self.dev_id})
      except Empty as e:
        if VERBOSE:
          print(self.getName() + " - Timeout expired")
          print(e)
      else:
        self.in_q.task_done()
        if self.state["active_pipelines"][self.pipe] or not self.state["adapt_pipes"]:
          # print("sending stuff over")
          if VERBOSE:
            print(time.time(), " sending ", image.type, image.t_frame)
          net_p3.send_fat_msg(self.sock, pickle.dumps(image), 0., 0.)

"""
TODO: 1. extend to different format
      2. recognize format automatically
"""
class TegraLogger(Thread):
  def __init__(self, state, log_file_path=None, freq=10):
    """
    @:arg - log file name
    @:arg - sampling frequency
    """
    Thread.__init__(self)
    cmds = ["tegrastats", "--interval",  str(int(1000/freq))]
    if log_file_path is None:
      log_file_path = "logs/{}_parsed_tegrastats.csv".format(int(state["start_time"]))
    self.p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    self.log_file = open(log_file_path, 'w+')
    self.state = state
  
  def run(self):
      stats, others = {}, {}
      text = None
      while self.state["is_running"]:
        current_stat = self.p.stdout.readline().decode().strip()
        if current_stat == '':
          raise ValueError("Tegrastats error detected")
        fields = current_stat.split(" ")
        stats["ram_used"], stats["ram_tot"] = [int(e) for e in fields[1][:-2].split("/")]
        others["cpu_perc_freq"] = [[int(e.split("%@")[0]), int(e.split("%@")[1])] for e in fields[5][1:-1].split(",")]
        stats["avg_used"] = mean([perc/100. for perc, freq in others["cpu_perc_freq"]])
        stats["avg_freq"] = mean([freq for perc, freq in others["cpu_perc_freq"]])
        # Weighted average frequency
        stats["w_avg_freq"] = mean([perc/100. *freq for perc, freq in others["cpu_perc_freq"]])
        # External Memory Control Frequency percentage used
        stats["emc"] = int(fields[7][:-1]) 
        stats["gpu_used"] = int(fields[9][:-1])
        for i in [16, 18, 20]:
          stats["pom_5v_{}".format(fields[i].split("_")[-1])] = int(fields[i+1].split("/")[0])
        if text is None:
          text = str("time") + "," + ",".join([str(e) for e in sorted(stats)])
          self.log_file.write(text + '\n')
        
        text = str(time.time()) + "," + ",".join([str(stats[e]) for e in sorted(stats)])
        self.log_file.write(text + '\n')
        self.log_file.flush()
  
  def stop(self):
    self.log_file.close()


class NetLogger(Thread):
  def __init__(self, state=None, log_file_path=None, freq=10, passwd='123', iface='wlan0'):
    """
    @:arg - log file name
    @:arg - sampling frequency
    """
    Thread.__init__(self)
    if log_file_path is None:
      log_file_path = "logs/{}_parsed_netstats.csv".format(int(state["start_time"]))
    self.log_file = open(log_file_path, 'w+')
    self.password = passwd
    self.interface = iface
    self.state = state
    self.interval = float(1./freq) # interval in milliseconds

  def run(self):
    stats, others = {}, {}
    text = None
    while self.state["is_running"]:
      start = time.time()
      stats = self.get_dict(self.password, self.interface)
      if text is None:
        text = str("time") + "," + ",".join([str(e) for e in sorted(stats)])
        self.log_file.write(text + '\n')
        self.log_file.flush()
      text = str(time.time()) + "," + ",".join([str(stats[e]) for e in sorted(stats)])
      self.log_file.write(text + '\n')
      self.log_file.flush()
      tmp = start - time.time() + self.interval
      if tmp > 0:
        time.sleep(start - time.time() + self.interval)
      else:
        print("logs are late")

  def stop(self):
    self.log_file.close()

  def get_dict(self, pswd, iface):
    list_fields=["Total_IP_pkts_received", "Total_IP_pkts_delivered", "IP_outgoing_pkts_dropped", "TCP_segments_received", "TCP_segments_sent", "TCP_segments_retransmitted", "UDP_pkts_received", "UDP_pkts_sent", "TCP_num_fast_retransmission", "TCP_num_retransmission_slow_start", "Total_bytes_received_at_IP", "Total_bytes_sent_at_IP", "tx retries", "tx failed", "signal", "signal avg", "beacon signal avg", "tx bitrate", "rx bitrate", "expected throughput"]
    d = {}
    myCmd = os.popen('netstat -s').read()
    #print(myCmd)
    ##Converting statistics into an array
    stat_array=myCmd.splitlines()


    ## Parse Information Under IP from NetStat Command
    if "Ip:" in stat_array:
      Ip_Stat_Array=stat_array[0:stat_array.index("Icmp:")]
      for line in Ip_Stat_Array:
        if "total packets received" in line:
          ipl = line.split()
          d['Total_IP_pkts_received'] = ipl[0]
        if "incoming packets delivered" in line:
          ipl = line.split()
          d['Total_IP_pkts_delivered'] = ipl[0]
        if "outgoing packets dropped" in line:
          ipl = line.split()
          d['IP_outgoing_pkts_dropped'] = ipl[0]
    

    ## Parse Information Under Icmp from NetStat Command
    if "Icmp:" in stat_array:
      Icmp_Stat_Array=stat_array[stat_array.index("Icmp:"):stat_array.index("IcmpMsg:")]
    

    ## Parse Information Under IcmpMsg from NetStat Command
    if "IcmpMsg:" in stat_array:
      IcmpMsg_Stat_Array=stat_array[stat_array.index("IcmpMsg:"):stat_array.index("Tcp:")]
    

    ## Parse Information Under Tcp from NetStat Command
    if "Tcp:" in stat_array:
      Tcp_Stat_Array=stat_array[stat_array.index("Tcp:"):stat_array.index("Udp:")]
      for line in Tcp_Stat_Array:
        if "segments received" in line:
          tcpl=line.split()
          d["TCP_segments_received"] = tcpl[0]
        if "segments sent out" in line:
          tcpl=line.split()
          d["TCP_segments_sent"] = tcpl[0]
        if "segments retransmitted" in line:
          tcpl=line.split()
          d["TCP_segments_retransmitted"] = tcpl[0]
    

    ## Parse Information Under Udp from NetStat Command
    if "Udp:" in stat_array:
      Udp_Stat_Array=stat_array[stat_array.index("Udp:"):stat_array.index("UdpLite:")]
      for line in Udp_Stat_Array:
        if "packets received" in line:
          udpl=line.split()
          d["UDP_pkts_received"] = udpl[0]
        if "packets sent" in line:
          udpl=line.split()
          d["UDP_pkts_sent"] = udpl[0]


    ## Parse Information Under TcpExt from NetStat Command
    if "TcpExt:" in stat_array:
      TcpExt_Stat_Array=stat_array[stat_array.index("TcpExt:"):stat_array.index("IpExt:")]
      for line in TcpExt_Stat_Array:
        if "fast retransmits" in line:
          tcpextl=line.split()
          d["TCP_num_fast_retransmission"] = tcpextl[0]
        if "retransmits in slow start" in line:
          tcpextl=line.split()
          d["TCP_num_retransmission_slow_start"] = tcpextl[0]


    ## Parse Information Under IpExt from NetStat Command
    if "IpExt:" in stat_array:
      IpExt_Stat_Array=stat_array[stat_array.index("IpExt:"):]
      for line in IpExt_Stat_Array:
        if "InOctets" in line:
          ipextl=line.split()
          d['Total_bytes_received_at_IP'] = ipextl[1]
        if "OutOctets" in line:
          ipextl=line.split()
          d['Total_bytes_sent_at_IP'] = ipextl[1]


    ########### WiFi Statistics ##########
    command = "echo {} | sudo -S iw dev {} station dump | grep 'signal\|signal avg\|tx bitrate\|rx bitrate\|expected throughput\|tx retries\|tx failed' | sed 's/://g'".format(pswd, iface)
    #command = "echo {} | sudo -S ./getWirelessStats.sh {}".format(PASSWD, W_IFACE)
    #print(command)
    myCmd = os.popen(command).read()
    ##Converting statistics into an array
    stat_array=myCmd.split('\t')
    #print(stat_array)

    for i in range(8):
      if stat_array[(2*i)+1] == 'expected throughput':
        d[stat_array[(2*i)+1] + " (Mbps)"] = stat_array[(2*i)+2].strip('\n').split()[0][:-4]
      else:
        d[stat_array[(2*i)+1]] = stat_array[(2*i)+2].strip('\n').split()[0]

    return d

