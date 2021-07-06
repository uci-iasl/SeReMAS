'''
Created on Dec 14, 2017

@author: davide
'''

import cv2
import base64
import numpy as np
import time



TYPE_IMAGE = 1
TYPE_DETECTION = 2
TYPE_ACTION = 3

verbose = True
STR_REPR_DELIM = '#$!@#'
DEFAULT_MSG_DELIM = '***'
NEW_DELIM = "$lol0l$"
DMD = DEFAULT_MSG_DELIM
HEADER_SIZE = len("Type" + DMD + "0" + DMD + "Size" + DMD + "0000000" + DMD)

send_msg_count = 1
frame_count = 0
SEQ_N_FORMAT = "{:05d}"
MAX_DATA_PCK = 900


def image_decoding(image_string):
    # print(image_string)
    jpg_original = base64.b64decode(image_string)
    nparr = np.fromstring(jpg_original, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         img = a.reshape( h, w, nb_planes )
    if verbose:
        print("frame_received")
    return img


def image_encoding(iimg):
    iretval, ibuffer = cv2.imencode('.jpg', iimg)
    return base64.b64encode(ibuffer)


class Image:
    def __init__(self, image=None, t_frame=None, frame_auth=None):
        self.image = image
        self.t_frame = t_frame
        self.frame_auth = frame_auth
        self.type = TYPE_IMAGE

    def get_value(self):
        return self.image

    def get_auth(self):
        return self.frame_auth

    def get_time(self):
        return self.t_frame

    def to_list(self):
        return [self.image, self.t_frame, self.frame_auth]

    def __str__(self):
        list_of_out = [image_encoding(self.image), self.t_frame, self.frame_auth]
        return STR_REPR_DELIM.join(str(e) for e in list_of_out) + STR_REPR_DELIM

    def from_str(self, input_as_str):
        outcomes = input_as_str.split(STR_REPR_DELIM)
        print(len(outcomes))
        img_as_str, self.t_frame, self.frame_auth, _ = outcomes
        self.t_frame = float(self.t_frame)
        # print(img_as_str)
        self.image = image_decoding(img_as_str)
        # cv2.imwrite("im_{}.jpg".format(time.time()), self.image)

    def nice_str(self, delim=','):
        return delim.join(str(e) for e in [self.t_frame, self.frame_auth, self.type])

    def __lt__(self, other):
        try:
            if self.t_frame < other.t_frame:
                return True
            return False
        except Exception as e:
            raise TypeError(str(type(other)) + " has not .t_frame element\n" + str(e))


class Detection:
    def __init__(self, x=None, y=None, area=None, t_frame=None, frame_auth=None, t_det=None, det_auth=None):
        self.x = x
        self.y = y
        self.area = area
        self.t_frame = t_frame
        self.frame_auth = frame_auth
        self.t_det = t_det
        self.det_auth = det_auth
        self.type = TYPE_DETECTION

    def get_auth(self):
        return self.act_auth

    def get_value(self):
        return self.x, self.y, self.area

    def to_list(self):
        return [self.x, self.y, self.area, self.t_frame, self.frame_auth, self.t_det, self.det_auth]

    def __str__(self):
        return STR_REPR_DELIM.join(str(e) for e in self.to_list())

    def nice_str(self, delim=','):
        return delim.join(str(e) for e in self.to_list())

    def from_str(self, input_as_str):
        self.x, self.y, self.area, self.t_frame, self.frame_auth, self.t_det, self.det_auth = input_as_str.split(STR_REPR_DELIM)
        self.x, self.y, self.area, self.t_frame, self.t_det = [float(e) for e in
                                                                        [self.x, self.y, self.area,
                                                                         self.t_frame, self.t_det]]

    def __lt__(self, other):
        try:
            if self.t_frame < other.t_frame:
                return True
            return False
        except Exception as e:
            raise TypeError(str(type(other)) + " has not .t_frame element\n" + str(e))


class Action:
    def __init__(self, x_act=None, y_act=None, z_act=None, t_frame=None, frame_auth=None, t_det=None, det_auth=None, t_act=None, act_auth=None):
        self.x_act = x_act
        self.y_act = y_act
        self.z_act = z_act
        self.t_frame = t_frame
        self.frame_auth = frame_auth
        self.t_det = t_det
        self.det_auth = det_auth
        self.t_act = t_act
        self.act_auth = act_auth
        self.type = TYPE_ACTION

    def get_value(self):
        return self.x_act, self.y_act, self.z_act

    def get_auth(self):
        return self.act_auth

    def to_list(self):
        return [self.x_act, self.y_act, self.z_act, self.t_frame, self.frame_auth, self.t_det, self.det_auth, self.t_act, self.act_auth]

    def __str__(self):
        return STR_REPR_DELIM.join(str(e) for e in self.to_list())

    def nice_str(self, delim=","):
        return delim.join(str(e) for e in self.to_list())

    def from_str(self, input_as_str):
        self.x_act, self.y_act, self.z_act, self.t_frame, self.frame_auth, self.t_det, self.det_auth, self.t_act, self.act_auth = input_as_str.split(STR_REPR_DELIM)
        self.x_act, self.y_act, self.z_act, self.t_frame, self.t_det, self.t_act = [float(e) for e in
                                                                                    [self.x_act, self.y_act, self.z_act,
                                                                                     self.t_frame, self.t_det, self.t_act]]

    def __lt__(self, other):
        try:
            if self.t_frame < other.t_frame:
                return True
            return False
        except Exception as e:
            raise TypeError(str(type(other)) + " has not .t_frame element\n" + str(e))


def decode_message(msg_str, delim = DEFAULT_MSG_DELIM):
    pieces = msg_str.split(delim)
    # print(pieces[-1])
    TYPE, type_str, SIZE, size_str, payload, _ = pieces
    ret_ = None
    if TYPE != "Type" or SIZE != "Size":
        raise ValueError("Message received hasn't Type and/or Size in the right places: " + pieces[0] + "," + pieces[2])
    if int(size_str) != len(payload):
        raise ValueError("Length of payload does not match the one expected {0} != {1}".format(int(pieces[3]), str(len(pieces[4]))))
    type_int = int(pieces[1])
    if type_int == TYPE_IMAGE:
        ret_ = Image()
    elif type_int == TYPE_DETECTION:
        ret_ = Detection()
    elif type_int == TYPE_ACTION:
        ret_ = Action()
    else:
        raise ValueError("Type " + str(type_int) + " unknown")

    ret_.from_str(payload)
    return ret_


def decode_message_ICS50(msg_str, delim=DEFAULT_MSG_DELIM, uav=False):
    if not uav:
        m = msg_str.split(delim)
        if m[1] == "DISTANCE":
            return None, None, None, None
        # print([(i, m[i]) for i in range(len(m))])
        frame_number, chunck, last_chunck, payload = m[3].split(NEW_DELIM)
        frame_number, chunck, last_chunck = [int(e) for e in [frame_number, chunck, last_chunck]]
        num_chuncks = int(last_chunck) + 1
        # print("Payload len", len(payload))
        # print(payload)
        return frame_number, chunck, last_chunck, payload
    else:
        tmp = msg_str.split(DEFAULT_MSG_DELIM)[3].split(NEW_DELIM)[3]
        print(tmp)
        return tmp


def encode_message_ICS50(msg, delim = DEFAULT_MSG_DELIM, edge=False):
    if not edge:
        global send_msg_count
        global frame_count
        if delim == STR_REPR_DELIM:
            raise ValueError(delim + " is a delimiter used in another layer of the protocol!")
        pic = str(msg)
        # print(pic)
        pckts = int(len(pic) / MAX_DATA_PCK)
        ret = []
        if len(pic) % 1100 > 0:
            pckts += 1
        for i in range(pckts):
            msg_str = DEFAULT_MSG_DELIM.join([str(e) for e in
                                  ['@@@U_000', SEQ_N_FORMAT.format(send_msg_count), str(time.time()),
                                  NEW_DELIM.join([str(e) for e in [frame_count, i, pckts - 1, pic[i * MAX_DATA_PCK: (i + 1) * MAX_DATA_PCK]]])]]) + DEFAULT_MSG_DELIM
            ret.append(msg_str)
            send_msg_count += 1
        frame_count += 1
    if edge:
        ret = DEFAULT_MSG_DELIM.join([str(e) for e in
                                  ['@@@G_000', SEQ_N_FORMAT.format(send_msg_count), str(time.time()), 
                                    str(NEW_DELIM.join([str(e) for e in [1, 1, 1, msg]]))]])+ DEFAULT_MSG_DELIM
        print(ret)
    return ret

DEFAULT_TIME = -1
class time_logs:
    def __init__(self):
      self.start = DEFAULT_TIME
      self.f_taken = DEFAULT_TIME
      self.f_start_tx = DEFAULT_TIME
      self.f_finish_tx = DEFAULT_TIME
      self.f_start_rx = DEFAULT_TIME
      self.f_finish_rx = DEFAULT_TIME
      self.in_i2dq = DEFAULT_TIME
      self.start_det = DEFAULT_TIME
      self.in_det2pil = DEFAULT_TIME
      self.start_pil = DEFAULT_TIME
      self.in_pil2opq = DEFAULT_TIME


def encode_message(obj2send, delim = DEFAULT_MSG_DELIM):
    if delim == STR_REPR_DELIM:
        raise ValueError(delim + " is a delimiter used in another layer of the protocol!")
    obj2send_str = str(obj2send)
    msg_str = "Type" + delim + str(obj2send.type) + \
              delim + "Size" + delim + str(len(obj2send_str)) + delim + obj2send_str + delim
    return msg_str

if __name__ == '__main__':
    import time
    # capt = cv2.VideoCapture(0)
    for _ in range(1000):
        # ret, image = capt.read()
        image = cv2.imread('tommy.jpg')
        im_test = Image(image, time.time(), 'TEST')
        msg = encode_message(im_test)
        im_dec = decode_message(msg)
        print(im_dec.get_time(), im_dec.get_auth())
        cv2.imshow('Frame', im_dec.get_value())
        cv2.waitKey(42)
