# import zmq
import time
import subprocess
import copy
# from shared_vars import *

START = "@@@"
INTER = "***"
bSTART = b"@@@"
bINTER = b"***"

#Message types
HI = 0
ACK = 1
DATA = 2
PROBE = 3
FINISH = 9
def myrange():
    i = 0
    while True:
        yield i
        i += 1%10000

SEQ_edge = myrange()

POS_FORMAT = "{:05.1f}"
SEQ_N_FORMAT = "{:05d}"
LEN_FORMAT = "{:010d}"
TSTAMP_FORMAT = "{:016.5f}"
DATA_BUFFER_SIZE = 15
LOG_PREFX = "---->"
BUFFER_SIZE = DATA_BUFFER_SIZE+100
NUM_TSTAMPS = 4

class Message(object):
    def __init__(self, aseq=None, atype=None, atotal_bytes=None, aNet=None, aparam1=None, aparam2=None):
        self.seq = aseq
        self.type = atype
        self.total_bytes = atotal_bytes
        self.net = aNet
        self.p1 = aparam1
        self.p2 = aparam2

    def get_type(self, atype):
        return self.type

    def get_net(self, anet):
        return self.net

    def get_p1(self, ap1):
        return self.p1

    def get_p2(self, ap2):
        return self.p2

    def set_type(self, atype):
        self.type = atype

    def set_net(self, anet):
        self.net = anet

    def set_p1(self, ap1):
        self.p1 = ap1

    def set_p2(self, ap2):
        self.p2 = ap2

    def compose(self):
        return START+INTER.join([str(e) for e in [self.type, self.net, self.p1, self.p2]]) + INTER

    def decode(self, pckt):
        start = pckt[:3]
        load = pckt[3:-3]
        finish = pckt[-3:]
        if start != START or finish != INTER:
            raise ValueError("Packet formatting is wrong " +start+ " " + finish)
        fields = load.split(INTER)
        self.type, self.net = fields[:2]
        if fields[2] == "None":
            self.p1 = None
        else:
            self.p1 = fields[2]
        if fields[3] == "None":
            self.p2 = None
        else:
            self.p2 = fields[3]

    def __str__(self):
        return self.compose()


def compose_data_message(sender="U_000", pos=None, sq_n=None, data=None, ns3=True):
    p = [POS_FORMAT.format(e) for e in pos]
    s_n = SEQ_N_FORMAT.format(sq_n)
    return compose_message([sender, s_n, p[0], p[1], str(DATA), str(data)], ns3)


def compose_probe_message(sender, pos, sq_n, data, ns3=True):
    p = [POS_FORMAT.format(e) for e in pos]
    s_n = SEQ_N_FORMAT.format(sq_n)
    return compose_message(["U_000", s_n, p[0], p[1], str(PROBE), str(data)], ns3)


def compose_message(fields, ns3=True, verbose=False):
    if not ns3:
        fields += [time.time()-1, time.time(), 500]
    for i,e in enumerate(fields):
        if verbose:
            if len(str(e)) > 10:
                print(i, str(e)[:10])
            else:
                print(i,e)
    return START + INTER.join([str(e) for e in fields]) + INTER

def send_big_data(sock, data, max_data_msg = DATA_BUFFER_SIZE):
    pck_num = int(len(data)/DATA_BUFFER_SIZE)
    if len(data) % DATA_BUFFER_SIZE != 0:
        pck_num += 1
    print("Sending {} pkts".format(pck_num))
    for i in range(pck_num):
        s_n = SEQ_N_FORMAT.format(next(SEQ_edge))
        portion4msg = data[DATA_BUFFER_SIZE*i:DATA_BUFFER_SIZE*(i+1)]
        msg = compose_message([s_n, i, pck_num, portion4msg], ns3=True).encode()
        print(type(msg))
        sock.send(msg)
    return True

def send_fat_msg(sock, data, timestamp=0, timestamp2=0):
    # print(LOG_PREFX + "send message data")
    # print(len(data))
    data_len = len(data)
    msg = START + SEQ_N_FORMAT.format(next(SEQ_edge)) + INTER + LEN_FORMAT.format(data_len) + INTER + TSTAMP_FORMAT.format(timestamp) + INTER + TSTAMP_FORMAT.format(timestamp2) + INTER
    sock.send(msg.encode())
    sock.send(data)
    sock.send(INTER.encode())
    # print("END OF SEND", len(data), data_len)

def receive_fat_msg(sock, reminder=b''):
    # print(LOG_PREFX, "recv message data")
    # print("receiving")
    len_sq_n = len(SEQ_N_FORMAT.format(next(SEQ_edge)))
    len_len = len(LEN_FORMAT.format(1))
    head_len = sum([len(START), len_sq_n, len(INTER), len_len, len(INTER)])
    # print("head_len", head_len)
    data_rcv = reminder
    # print(type(reminder))
    # print(type(data_rcv))
    # print("receiving2")
    count_empty = 0
    while(len(data_rcv) < head_len):
        rcv_ = sock.recv(BUFFER_SIZE)
        if len(rcv_) < 1:
          count_empty += 1
        else:
          count_empty = 0
        if count_empty > 10:
          raise(ValueError("Connection terminated by peer"))
        # print("RCV", rcv_)
        data_rcv += rcv_
    # print(data_rcv)
    messages = data_rcv.split(bSTART)
    ## SELECT curr-ent message and parsing the header
    # print(messages)
    curr = messages[1]
    # print("curr", curr)
    sq_n = curr[:len_sq_n]
    # print("sq_n", sq_n)
    data_len = (curr[len_sq_n+len(INTER):len_sq_n+len(INTER)+len_len])
    # print("len(data_len)", len(data_len))
    data_len = int(data_len)
    # print("data_len", data_len)
    curr = curr[len_sq_n+len(INTER)+len_len:]

    # Check if I have the full message. If not more than one messages have been received, then be sure I 
    #  receive until end of current message
    # print("receiving3")
    if len(messages) > 2:
        reminder = START.join(messages[2:])
    else:
        reminder = ""        
        while len(curr) < data_len + len(INTER) + len("1544075326.46168") + len(INTER) + len("1544075326.46168") + len(INTER):
            curr += sock.recv(BUFFER_SIZE)
        reminder = curr[data_len+len(INTER)+ len("1544075326.46168") + len(INTER) + len("1544075326.46168") + len(INTER):]
    # print("receiving4")
    
    timestamp_rcv = curr.split(bINTER)[1]
    curr = curr[len(INTER) + len("1544075326.46168"):]
    timestap2_rcv = curr.split(bINTER)[1]
    data = curr.split(bINTER)[2]
    idx = 2
    while len(data) < data_len:
        # print("adding data")
        data += bINTER
        idx += 1
        data += curr.split(bINTER)[idx]
    tmp = copy.deepcopy(data)
    print(len(data), data_len)
    
    # print(timestamp_rcv)
    # print(LOG_PREFX, "recv message data", len(data), data_len)
    
    return timestamp_rcv, timestap2_rcv, data, reminder


def send_tstamps(sock, tstamps):
    if len(tstamps) != NUM_TSTAMPS:
        raise ValueError("tstamps are {} instead of {}".format(len(tstamps), NUM_TSTAMPS))
    # print(LOG_PREFX + "send message tstamps")
    msg = START + INTER.join([TSTAMP_FORMAT.format(float(t)) for t in tstamps]) + INTER
    # print(msg)
    # print(len(msg))
    sock.send(msg.encode())


def recv_tstamps(sock, reminder):
    # print(LOG_PREFX, "receive message tstamps")
    fake = TSTAMP_FORMAT.format(1544075326.46168)
    msg_len = len(START) + NUM_TSTAMPS*(len(fake) + len(INTER))
    # print(msg_len)
    data_rcv = reminder
    while len(data_rcv) < msg_len:
        rcv_ = sock.recv(BUFFER_SIZE)
        data_rcv += rcv_
    if len(data_rcv) > msg_len:
        tmp = data_rcv.split(bSTART)
        curr = tmp[1].decode()
        del tmp[1]
        reminder = bSTART.join(tmp)
    else:
        tmp = data_rcv.split(bSTART)
        curr = tmp[1].decode()
        reminder = b''
    # print(curr.split(INTER))

    res = [float(e) for e in curr.split(INTER) if len(e) > 1]
    # print("tstamps received")
    # print(", ".join([str(e) for e in res]))
    return res, reminder


'''
    Returns the ping estimation to computer with ip address ip. Runs interations times.
'''
def get_ping_est(ip="localhost", iterations=5):
    result = subprocess.run(['ping', ip, '-n', '-i 0.2', '-c 5'], stdout=subprocess.PIPE)
    return float(result.stdout.split(b"=")[-1].split(b"/")[1])/1000.


def receiver_big_data(sock):
  so_far = ""
  finished = False
  reminder = ""
  one = two = ""
  while not finished:
    data_rcv = sock.recv(BUFFER_SIZE).decode()
    pkts = data_rcv.split(START)
    while len(two.split(START)) > 1:
      one = two + pkts[0]
      two = data_rcv[len(START)+len(one):]
      s_n, i, pck_num, portion4msg, gne = one.split(INTER)
      portion4msg = portion4msg[2:-1]
      so_far += portion4msg
      print("{}/{}".format(i, pck_num))
      if int(pck_num) == int(i)+1:
        finished = True 

    while not finished and False:
      # print(finished)
      data_rcv = sock.recv(BUFFER_SIZE).decode()
      print(data_rcv)
      pkts = data_rcv.split(START)
      one = two = None
      if len(pkts) > 3:
          print(data_rcv)
      elif len(pkts) == 3:
          one = reminder + pkts[0]
          two = pkts[1]
          reminder = pkts[2]
      elif len(pkts) == 2:
          one = reminder + pkts[0]
          reminder = pkts[1]
      elif len(pkts) == 1:
          reminder += pkts[0]

      if one is not None:
          if len(one) == 0:
              continue
          try:
              s_n, i, pck_num, portion4msg, gne = one.split(INTER)
          except Exception as e:
              print("--------")
              print(one)
              print("--------")
              print(len(one.split(INTER)))
              raise e
          portion4msg = portion4msg[2:-1]
          so_far += portion4msg
          print("{}/{}".format(i, pck_num))
          if int(pck_num) == int(i)+1:
              finished = True
      if two is not None:
          try:
              s_n, i, pck_num, portion4msg, gne = two.split(INTER)
          except Exception as e:
              print("--------")
              print(two)
              print("--------")
              print(len(two.split(INTER)))
              raise e
          portion4msg = portion4msg[2:-1]
          so_far += portion4msg
          print("{}/{}".format(i, pck_num))
          if int(pck_num) == int(i)+1:
              finished = True
      if reminder[-len(INTER):] == INTER:
          try:
              s_n, i, pck_num, portion4msg, gne = reminder.split(INTER)
          except Exception as e:
              print("--------")
              print(reminder)
              print("--------")
              print(len(reminder.split(INTER)))
              raise e
          portion4msg = portion4msg[2:-1]
          so_far += portion4msg
          reminder = ""
          print("{}/{}".format(i, pck_num))
          if int(pck_num) == int(i)+1:
              finished = True
    print(len(so_far))
    return so_far 

def decode_message(pckt):
    start = pckt[:3]
    load = pckt[3:-3]
    finish = pckt[-3:]
    if start != START or finish != INTER:
        raise ValueError("Packet formatting is wrong " + start + " " + finish)
    fields = load.split(INTER)
    for idx, e in enumerate(fields):
        if len(e) <= 0:
            fields[idx] = ''
        elif e[0] == 'u':
            fields[idx] = e[1:]
    return fields


class TimeoutError(Exception):
    pass


class SocketWrapper(object):
    def __init__(self, socket):
        self.socket = socket

    def __getattr__(self, item):
        return getattr(self.socket, item)

    def recv(self, timeout=10000):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        msg = dict(poller.poll(timeout))
        if len(msg) > 0:
            return self.socket.recv()
        raise TimeoutError()
