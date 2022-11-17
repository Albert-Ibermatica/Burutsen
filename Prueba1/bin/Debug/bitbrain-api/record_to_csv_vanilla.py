from contextlib import ExitStack
import csv
import sys
import time


from bbt import Signal, Device, SensorType, ImpedanceLevel


def try_to(condition, action, tries, message=None):
        t = 0
        while (not condition() and t < tries):
            t += 1
            if message:
                print("{} ({}/{})".format(message, t, tries))
            action()
        return condition()


def config_signals(device):
    signals = device.get_signals()            
    for s in signals:
        s.set_mode(1)

def csv_filename(signal_number, signal_type):
    return f"signal_{signal_number}({signal_type}).csv"


def record_one(device, csvs, signals):
    sequence, battery, flags, data = device.read()
    ts = time.time_ns()
    common_data = [ts, sequence, battery, flags]
    data_offset = 0
    for csv, s in zip(csvs, signals):
        n_values = s.channels()*s.samples()        
        signal_data_slice = data[data_offset:data_offset + n_values]
        for i in range(s.samples()):            
            csv.writerow(common_data + signal_data_slice[i::s.samples()])
        data_offset += n_values


def record_data(device, length):
    #create the csv files    
    with ExitStack() as stack:
        active_signals = [s for s in device.get_signals() if s.mode() != 0]
        #open csv files
        files = [stack.enter_context(open(csv_filename(i, s.type()), 'w', newline='')) for i,s in enumerate(active_signals)]

        #write headers
        writers = [csv.writer(f) for f in files]
        for (s, w) in zip(active_signals, writers):
            common_header = ["timestamp", "sequence", "battery", "flags"]
            channels_header = [ f"channel_{i}" for i in range(s.channels())]
            w.writerow(common_header + channels_header)

        #record data
        device.start()
        f = int(device.get_frequency())        
        for i in range(length * f):
            record_one(device, writers, active_signals)  
        device.stop()
    print ("Stopped: ", not device.is_running())


if __name__ == "__main__":

    if (len(sys.argv) <= 1):  
        print("Usage: " + sys.argv[0] + " <device name> [time (s) default = 10]")
        exit(0)
        
    name = sys.argv[1]  
    length = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    with Device.create_bluetooth_device(name) as device:
        if not try_to(device.is_connected, device.connect, 10, "Connecting to {}".format(name)):
            print("unable to connect")
            exit(1)
        print ("Connected")

        print(f"Recording {length} seconds of data into csv files from device {name}")

        config_signals(device)
        record_data(device, length)

        if not try_to(lambda: not device.is_connected(), device.disconnect, 10):
            print("unable to disconnect")
            exit(1)        
        print ("Connected")    
    
        
