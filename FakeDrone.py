'''
Created on Dec 15, 2017

@author: dcallega
'''
import threading
import time

class FakeDrone(threading.Thread):
    '''
    classdocs
    '''
    def __init__(self, connection_string = 'tcp:127.0.0.1:5763'):
      '''
      Constructor
      '''
      threading.Thread.__init__(self)
      self.connection_string = connection_string
      print("Connected with " + self.connection_string)
        
    def set_movement(self, lr, ud, fb, duration=0.5, speed=200):
      # print("Moving " + " ".join([str(e) for e in [lr, ud, fb, duration, speed]]))
      time.sleep(duration)

    def move(self, *args, **kargs):
      pass
      # print("You called Move")
        
    def set_takeoff(self, really=True):
      print("Set takeoff " + str(really))
        
    def set_up(self, really=True):
      print("Set up " + str(really))
        
    def set_mode(self, mode):
      print("Set mode " + mode)
    
    def set_land(self, really=True):
      print("Set land " + str(really))
        
    def stop(self):
      print("STOP")

    def get_gps(self):
      return (0.000001, -0.000012)