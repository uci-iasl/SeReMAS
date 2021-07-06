import os.path
import time
from threading import Thread
from config_v3_2 import *

class FlightLog:
    """
    FlightLog works to intelligently capture flight data in a CSV file
    
    :param vehicle: Vehicle instance to log
    :param logtime: Number of seconds between logging periods (defaults to .1)
    :param position: True or false to log GPS position (default: True)
    :param attitude: True or false to log vehicle attitude (default: True)
    :param gyro: True or false to log gyroscope (default: True)
    :param accel: True or false to log accelerometer (default: True)
    :param barom: True or false to log barometer (default: True)
    :param battery: True or false to log battery information (default: True)
    :param sys: True or false to log misc system information (default: False)
    """
    
    def __init__(self,vehicle, logtime=.5, position = True,attitude = True,gyro = True,accel = True,barom = True,battery = True,sys = False, start_time=None):
        """
        Initialize a FlightLog instance based on a SuperVehicle
        """
        
        if vehicle is None:
            raise ValueError("Must have a vehicle instance! Cannot log an nonexistant item!")
        self._vehicle = vehicle
        
        self._logtime = logtime
        if(self._logtime > 5):
            raise ValueError("Cannot have a logtime over 5 seconds!")
        
        self._position = position
        self._attitude = attitude
        self._gyro = gyro
        self._accel = accel
        self._barom = barom
        self._battery = battery
        self._sys = sys
        
        #reset last known timestamps
        self._gyrots = 0
        self._accelts = 0
        self._baromts = 0
        self._batteryts = 0
        self._systs = 0
        self._modets = None
        self._armedts = None
        self._poslast = ""
        self._attlast = ""
        self._linelast = ""
        self._start_time = start_time
        
        #set internal thread tracking values
        self._run = False
        self._stopped = False

        #set flag message
        self._flag = None
        
    def _logfunction(self):
        while self._run:
            line = self._logline() #get the log line 
            if line is None:
                print("[FlightLog]: Error occured on logging thread!")
            else:           #only write a line if there are things to write
                self._file.write(line + os.linesep)
            time.sleep(self._logtime)
                
        self._stopped = True
        self._closefile()
        
    def start(self):
        """
        Starts the FlightLog instance. Must be called by the user.
        """
        print("entering openfile")
        self._file = self._openfile()
        print("Logging file opened: " + self._filename)
        self._file.write(self._header() + os.linesep)
        self._t = Thread(target = self._logfunction)
        self._run = True
        self._t.start()
        
    def stop(self):
        """
        Stop the logging thread and close the file. Function will block until thread is
        properly stopped
        """
        self._run = False
        while not self._stopped:
            time.sleep(1)

    def flag(self, message):
        """
        Sets a custom 'flag' message to that logging frame for the next log line

        Useful for marking a specific location or area of data to analyze. or commenting on a
        data point.

        Multiple calls to flag before a logging period will be concatenated with a semicolon delimiter

        :param message: Message to flag the log line with
        """
        if self._flag is not None:
            self._flag += "; " + message.replace(',','') #remove commas to not corrupt CSV file
        else:
            self._flag = message.replace(',','') #remove commas to not corrupt CSV file

    def _openfile(self):
        """
        Safely opens an instance a test file,but does not write anything to it
        No file will be overwritten,instead a number will be appended to the file.
        The method will use the current date
        and time for the filename. (e.g. 2018-09-12 11:23:11)
        
        :returns: File instance of logging file
        """
        filename = FLIGHTLOG_NAME.format(self._start_time)
         
        # DEACTIVATED: this checks if there is a same named file and appends a counting number.
        #              Not needed in this context since we use a one-time full precision timestamp
        #find a non-used filename
        # number = 1
        # test = filename
        # while(os.path.isfile(test)): #does the file exist?
        #     number += 1
        #     test = filename + "("+ number +")"
            
        #append filetype
        # test += ".csv"
        self._filename = filename
        return open(os.path.join(LOGS_PATH, self._filename), "w")
        
    def _closefile(self):
        """
        Closes the current logging file
        """
        #close the file
        if self._file is not None:
            self._file.close()
            print("Closed logfile " + self._filename)
        else:
            raise ValueError("Closing unopened file?")

        
    def _header(self):
        """
        Prints the file header with corresponding columns
        """
        # generate header based on each tracked attribute
        line  = "Timestamp,Mode,Armed"
        if self._position:
            line += ",North,East,Down"
        if self._attitude:
            line += ",Pitch,Roll,Yaw"
        if self._gyro:
            line += ",Xgyro,Ygyro,Zgyro"
        if self._accel:
            line += ",Xaccel,Yaccel,Zaccel"
        if self._barom:
            line += ",press_abs,press_diff,temp"
        if self._battery:
            line += ",temperature,voltages,current_battery,current_consumed,battery_remaining,time_remaining,charge_state"
        if self._sys:
            line += ",drop_rate_comm,errors_count1,errors_count2,errors_count3,load,time"
        line += ",flag"
        return line
        
    def _logline(self):
        """
        returns a single line of a log from the current time.
        """
        # generate header based on each tracked attribute
        line  = repr(time.time()) + ","
        
        if(self._vehicle.mode != self._modets):#only append mode if changed
            line += str(self._vehicle.mode)
            self._modets = self._vehicle.mode

        line += ","
        if(self._vehicle.armed is not self._armedts):#only append armed status if changed
            line += str(self._vehicle.armed)
            self._armedts = self._vehicle.armed
        
        if self._position:
            pos = "," + str(self._vehicle.location.local_frame.north) #north
            pos += "," + str(self._vehicle.location.local_frame.east) #east
            pos += "," + str(self._vehicle.location.local_frame.down) #down
            if pos != self._poslast: #only log if position changed
                line +=  pos
                self._poslast = pos
            else:
                line += ",,,"
            
        
        if self._attitude:
            att = "," + str(self._vehicle.attitude.pitch) #pitch
            att += "," + str(self._vehicle.attitude.yaw) #yaw
            att += "," + str(self._vehicle.attitude.roll) #roll
            if att != self._attlast: #only log if attitude changed
                line +=  att
                self._attlast = att
            else:
                line += ",,,"    
                
            
        if self._gyro:
            if self._gyrots is not self._vehicle.state.gyro.timestamp: #gyro has been updated
                line += "," + str(self._vehicle.state.gyro.x) #Xgyro
                line += "," + str(self._vehicle.state.gyro.y) #Ygyro
                line += "," + str(self._vehicle.state.gyro.z) #Zgyro
                self._gyrots = self._vehicle.state.gyro.timestamp
                
            else: #gyro has not been updated,do not re-record
                line += ",,,"
            
        if self._accel:
            if self._accelts is not self._vehicle.state.accel.timestamp: #accel has been updated
                line += "," + str(self._vehicle.state.accel.x) #Xaccel
                line += "," + str(self._vehicle.state.accel.y) #Yaccel
                line += "," + str(self._vehicle.state.accel.z) #Zaccel
                self._accelts = self._vehicle.state.accel.timestamp
            
            else: #accel has not been updated,do not re-record
                line += ",,,"
                
        if self._barom:
            if self._baromts is not self._vehicle.state.barom.timestamp: #barom has been updated
                line += "," + str(self._vehicle.state.barom.press_abs) #press_abs
                line += "," + str(self._vehicle.state.barom.press_diff) #press_diff
                line += "," + str(self._vehicle.state.barom.temp) #temp
                self._baromts = self._vehicle.state.barom.timestamp
            
            else: #barom has not been updated,do not re-record
                line += ",,,"
                
        if self._battery:
            if self._batteryts is not self._vehicle.state.battery.timestamp: #battery has been updated
                line += "," + str(self._vehicle.state.battery.temperature) #temperature
                line += "," + str(self._vehicle.state.battery.voltages) #voltages
                line += "," + str(self._vehicle.state.battery.current_battery) #current_battery
                line += "," + str(self._vehicle.state.battery.current_consumed) #current_consumed
                line += "," + str(self._vehicle.state.battery.battery_remaining) #battery_remaining
                line += "," + str(self._vehicle.state.battery.time_remaining) #time_remaining
                line += "," + str(self._vehicle.state.battery.charge_state) #charge_state
            
                self._batteryts = self._vehicle.state.battery.timestamp
            
            else: #battery has not been updated,do not re-record
                line += ",,,,,,,,"
        if self._sys:
            if self._systs is not self._vehicle.state.sys.time: #sys data has been updated
                line += "," + str(self._vehicle.state.sys.drop_rate_comm) #drop_rate_com
                line += "," + str(self._vehicle.state.sys.errors_count1) #errors_count1
                line += "," + str(self._vehicle.state.sys.errors_count2) #errors_count2
                line += "," + str(self._vehicle.state.sys.errors_count3) #errors_count3
                line += "," + str(self._vehicle.state.sys.load) #load
                line += "," + str(self._vehicle.state.sys.time) #time
                
                self._systs = self._vehicle.state.sys.time
                
            else: #do not re-log duplicate data
                line += ",,,,,,"

        if self._flag is not None:
            line += "," + self._flag
            self._flag = None
        else:
            line += ","

        return line
        
        
