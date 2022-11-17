from enum import Enum
import py_bbt_driver as detail

class Signal:
    '''
    The Signal object manages the different signals available from the Device objects.

    This objects are created by the Device methods and should not be created by the user

    Parameters
    ----------
    handle : object
        A reference to the device this signal belongs to

    '''
    def __init__(self, handle):
        self.__handle = handle        

    def __repr__(self):
        return "({}, {}, {}, {})".format(self.type(), self.channels(), self.samples(), self.mode())

    def type(self):
        '''
        Get the type of the signal

        Returns
        -------
        type: str
            A string with the type of the signal (e.g. EEG, ExG_B, ...).  See c++ sdk documentation for details on these signals.
        '''
        buffer = bytearray(5)
        detail.bbt_signal_get_type(self.__handle, buffer)
        return buffer.decode('utf8').strip("\0")

    def channels(self):
        '''
        Get the number of channels of the signal

        Returns
        -------
        channels: int
            The number of channels this signal is composed.
        '''
        return detail.bbt_signal_get_channels(self.__handle)

    def samples(self):
        '''
        Get the number of samples of the signal

        Returns
        -------
        samples: int
            The number of samples of this signal in each data block read from the device
        '''
        return detail.bbt_signal_get_samples(self.__handle)
    
    def mode(self):
        '''
        Get the mode of the signal.  Mode 0 means disabled (not registered). Otherwise the signal is registered and transmitted.

        Returns
        -------
        mode: int
            The mode previously configured with set_mode method
        '''
        return detail.bbt_signal_get_mode(self.__handle)

    def set_mode(self, new_mode):
        '''
        Set the mode of the signal.  Mode 0 means disabled (not registered). Otherwise the signal is registered and transmitted.

        This method might fail for different reasons (communication issues, bad values, etc).
        Please use method mode to check if the configuration has been successful

        Parameters
        ----------
        new_mode: int
            The mode to be set.  Check success calling get_mode afterwards
        '''
        return detail.bbt_signal_set_mode(self.__handle, new_mode)


class SensorType(Enum):
    '''
    The type of sensor used for EEG devices. Can be dry or water based
    '''
    DRY = detail.bbt_dry_eeg_sensor
    WATER = detail.bbt_water_eeg_sensor


class ImpedanceLevel(Enum):
    '''
    The different impedance values for the EEG channels.
    '''
    UNKNOWN = detail.bbt_driver_impedance_unknown
    SATURATED = detail.bbt_driver_impedance_saturated
    BAD = detail.bbt_driver_impedance_bad    
    FAIR = detail.bbt_driver_impedance_fair
    GOOD = detail.bbt_driver_impedance_good


class Device:
    '''
    The Device object manages all the interactions with the Bitbrain devices through bluetooth or usb

    Use class methods create_bluetooth_device and create_usb_device to create instances of this class
    '''
    
    class DeviceType(Enum):
        '''
        Enumeration of the different types of devices available
        '''
        BLUETOOTH = 0
        USB = 1

    def __init__(self, id, device_type, eeg_sensor_type):  
        '''
        Private constructor. Preferred use 

        Parameters
        ----------
        id : str, int
            If the name is a string, it represents the serial number (also bluetooth name) of the device or the mac address (with the format xx:xx:xx:xx:xx:xx where xx is a byte in hexadecimal).
            The mac method is preferred to avoid the discovery phase
	        If the name is an int, it corresponds to the number of COM port where a usb device is connected
        device_type: Device.DeviceType
            The type of device to connect to
        eeg_sensor_type : SensorType
            The type of sensor used. Required to get the correct eeg impedance levels. The values available are [SensorType.DRY, SensorType.WATER]

        '''
        self.eeg_sensor_type = eeg_sensor_type
        self.__device_type = device_type
        self.__id = id        
        self.__handle = None
        self.__hw_version = (0,0)
        self.__fw_version = (0,0)
        self.__frequency = 0
        self.__signals = None
        self.__has_sd_capabilities = False
        self.__folder = None
        self.__file = None

    @classmethod
    def create_bluetooth_device(cls, id, eeg_sensor_type=SensorType.DRY):
        '''
        Create an instance of the class with a bluetooth connection to a device with the specified sensor type

        Parameters
        ----------
        id : str
            It represents the serial number (also bluetooth name) of the device or the mac address (with the format xx:xx:xx:xx:xx:xx where xx is a byte in hexadecimal).
            The mac method is preferred to avoid the discovery phase
	        
        eeg_sensor_type : SensorType
            The type of sensor used. Required to get the correct eeg impedance levels. The values available are [SensorType.DRY, SensorType.WATER]
        '''
        return cls(id, Device.DeviceType.BLUETOOTH, eeg_sensor_type)
    
    @classmethod
    def create_usb_device(cls, port, eeg_sensor_type=SensorType.DRY):
        '''
        Create an instance of the class with a usb connection to a device with the specified sensor type

        Parameters
        ----------
        port : int
            It corresponds to the number of COM port where the usb device is connected

        eeg_sensor_type : SensorType
            The type of sensor used. Required to get the correct eeg impedance levels. The values available are [SensorType.DRY, SensorType.WATER]
        '''
        return cls(port, Device.DeviceType.USB, eeg_sensor_type)

    #connection management
    def connect(self):
        '''
        Try to connect to the device and read all the configuration. Check success with is_connected.

        '''
        detail.bbt_driver_connect(self.__handle)
        if self.is_connected():
            self.__load()


    def disconnect(self):
        '''
        Disconnects from the device. Check success with is_connected.

        '''
        detail.bbt_driver_disconnect(self.__handle)


    def is_connected(self):
        '''
        Get the connection status

        Returns
        -------
        is_connected : boolean
            True if the device is connected to the computer. False otherwise.

        '''
        return detail.bbt_driver_is_connected(self.__handle)


    def reconnect(self):
        '''
        Disconnects and tries to connect back to the same device

        '''
        detail.bbt_driver_reconnect(self.__handle)
        if self.is_connected():
            self.__load()

    #gather information from device
    def get_hw_version(self):
        '''
        Get the hardware version of the device

        Returns
        -------
        hw_version: tuple
            A tuple with two values: The major and the minor version. (0,0) on failure.

        '''
        return self.__hw_version

    def get_fw_version(self):
        '''
        Get the firmware version of the device

        Returns
        -------
        fw_version: tuple
            A tuple with two values: The major and the minor version. (0,0) on failure.
        '''
        return self.__fw_version

    def get_frequency(self):
        '''
        Get the transmission frequency of the device

        Returns
        -------
        frequency: int
            The number of blocks recorded and transmitted every second
        '''
        return self.__frequency


    #device signals
    def get_signals(self):
        '''
        Get the signals available

        Returns
        -------
        signals: list of Signal objects
            The signals available with their current mode configuration
        '''
        return self.__signals


    #sd card management
    def has_sd_card_capability(self):
        '''
        Get the SD Card capabilities of the device

        Returns
        -------
        sd_card_capabilities: boolean
            True if the device can record to an SD Card
        '''
        return self.__has_sd_capabilities


    def is_sd_card_enabled(self):
        '''
        Get the SD Card configuration of the device

        Returns
        -------
        is_sd_card_enabled: boolean
            True if the device is configured to record to an SD Card
        '''
        return detail.bbt_driver_is_sd_card_enabled(self.__handle) == 1


    def enable_sd_card(self, enable=True):
        '''
        Set the SD Card configuration of the device

        This function will fail if the SD Card is not in the socket and might fail from other reasons (bad format, etc).
        Check success with is_sd_card_enabled

        Parameters
        ----------
        enable: boolean
            True to enable the recording in sd card. False to disable
        '''
        return detail.bbt_driver_enable_sd_card(self.__handle, 1 if enable else 0)


    def get_folder(self):
        '''
        Get the folder in the SD Card where the recording will be saved

        Returns
        -------
        folder: str
            The full path of the folder inside the SD Card
        '''
        return self.__folder


    def set_folder(self, folder):
        '''
        Set the folder in the SD Card where the recording will be saved.

        This method might fail for different reasons. Please use method get_folder to verify whether the device has been configured successfully.

        Parameters
        ----------
        folder: str
            The full path of the folder inside the SD Card.
            Each directory name should be no longer than 8 characters and the whole path should be no longer than 256 characters.
            Use / to separate subfolders.
        '''
        if detail.bbt_driver_set_folder(self.__handle, folder):
            self.__folder = self.__get_folder()


    def get_file(self):
        '''
        Get the file name in the SD Card where the recording will be saved

        Returns
        -------
        file: str
            The name of the file where the data will be saved with no folder or path.
        '''
        return self.__file


    def set_file(self, filename):
        '''
        Set the file name in the SD Card where the recording will be saved.

        This method might fail for different reasons. Please use method get_file to verify whether the device has been configured successfully.

        Parameters
        ----------
        file: str
            The file name of the file inside the SD Card folder.
            The file basename should be shorter than 8 characters (up to 6 is recommended) with up to 3 characters for the extension (after teh dot)
            Any alphanumeric extension is valid (even none)
            No path or folder should be provided. Use method set_folder instead.
        '''
        if detail.bbt_driver_set_file(self.__handle, filename):
            self.__file = self.__get_file()


    def synchronize(self):
        '''
        Starts a NTP like protocol with the remote device to compute an estimation of the time of flight and the offsets between the clocks of the computer and the device.
        This function will block for few seconds until a stable estimation is obtained

        Note that the time of flight correlates with the size of the data transmitted so that the more signals, channels and samples transmitted, the bigger the time of flight.
        NTP protocol messages are generally smaller than the data transmitted during the recording.

        Note also that the offset between clocks is not valid after the device is reset.

        Returns
        -------
        synchronization: tuple of length 2
            The tuple returned is (time of flight, offset between clocks). (-1,0) on failure
        '''
        result = detail.bbt_driver_synchronize(self.__handle)
        return tuple(result[1:]) if result[0] == 1 else (-1,0)


    def start(self):
        '''
        Commands the device to start the data acquisition and transmission

        Returns
        -------
        True on success. False otherwise
        '''
        return detail.bbt_driver_start(self.__handle) == 1


    def stop(self):
        '''
        Commands the device to stop the data acquisition and transmission

        Returns
        -------
        True on success. False otherwise
        '''
        return detail.bbt_driver_stop(self.__handle) == 1


    def is_running(self):
        '''
        Get the running status of the device.

        Notice this status is not changed on disconnections.  A device that is acquiring will keep acquiring (and recording to de SD Card if configured) even if it gets disconnected from the computer.

        Returns
        -------
        True if acquiring. False otherwise
        '''
        return detail.bbt_driver_is_running(self.__handle) == 1


    def read_data_size(self):
        '''
        Get the size of the data gathered by the read method

        Returns
        -------
        size: int
            The number of values obtained from the read method
        '''
        return detail.bbt_driver_read_data_size(self.__handle)


    def read(self):
        '''
        Waits until it receives a new data block from the device

        Returns
        -------
        data: tuple
            A tuple with four elements (sequence, battery level, flags, signals),
            where sequence is the number of sequence of the block,
            battery level indicates the battery of the device,
            flags indicate if some problem happened and
            signals is a list with all the values registered from the sifferent signals of length read_data_size
        '''
        raw = detail.bbt_driver_read(self.__handle)
        raw_battery = raw[-2]
        if not (raw_battery < 2*16):
            raw_battery = -1

        return (raw[-3], raw_battery, raw[-1], raw[:-3])

    def get_eeg_impedance(self, channel):
        '''
        Gets the impedance level from the eeg channel selected.  Valid values are only available after performing a call to the read method

        Parameters
        ----------
        channel: The number of eeg channel to get the impedance from

        Returns
        -------
        impedance level: ImpedanceLevel

        '''
        return ImpedanceLevel(detail.bbt_driver_get_eeg_impedance(self.__handle, channel))

    ########################################################################################
    ## Internal methods for the Device class.
    ## Not intended to be called directly from the outside (private)
    ########################################################################################
    #context management (required for avoiding c++ memory leaks)
    def __enter__(self):        
        if self.__device_type is Device.DeviceType.USB:
            self.__handle = detail.bbt_driver_new_usb(self.__id, self.eeg_sensor_type.value)     
        elif self.__device_type is Device.DeviceType.BLUETOOTH:
            self.__handle = detail.bbt_driver_new_bluetooth(self.__id, self.eeg_sensor_type.value)
        else:
            self.__handle = None
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        detail.bbt_driver_free(self.__handle)


    def __get_hw_version(self):
        hw_version = detail.bbt_driver_get_hw_version(self.__handle)
        return (0,0) if hw_version[0] == 0 else tuple(hw_version[1:])


    def __get_fw_version(self):
        fw_version = detail.bbt_driver_get_fw_version(self.__handle)
        return (0,0) if fw_version[0] == 0 else tuple(fw_version[1:])


    def __get_folder(self):
        folder = bytearray(255)
        detail.bbt_driver_get_folder(self.__handle, folder)
        return folder.decode('utf8').strip("\0")


    def __get_file(self):
        filename = bytearray(255)
        detail.bbt_driver_get_file(self.__handle, filename)
        return filename.decode('utf8').strip("\0")


    def __load(self):
        self.__hw_version = self.__get_hw_version()
        self.__fw_version = self.__get_fw_version()
        self.__frequency = detail.bbt_driver_get_frequency(self.__handle)
        self.__signals = [Signal(detail.bbt_driver_get_signal(self.__handle, i)) for i in range(detail.bbt_driver_get_number_of_signals(self.__handle))]

        self.__has_sd_capabilities = detail.bbt_driver_has_sd_card_capability(self.__handle) == 1
        if self.__has_sd_capabilities:
            self.__folder = self.__get_folder()
            self.__file = self.__get_file()


if __name__ == "__main__":
    import sys

    def try_to(condition, action, tries, message=None):
        t = 0
        while (not condition() and t < tries):
            t += 1
            if message:
                print("{} ({}/{})".format(message, t, tries))
            action()
        return condition()


    if (len(sys.argv) > 1):  
        name = sys.argv[1]              
        with Device.create_bluetooth_device(name) as device:
            if not try_to(device.is_connected, device.connect, 10, "Connecting to {}".format(name)):
                print("unable to connect")
                exit(1)
            print ("Connected")

            print("HW version: ", device.get_hw_version())
            print("FW version: ", device.get_fw_version())
            print("Frequency: ", device.get_frequency())
            signals = device.get_signals()            
            print("Signals: ", signals)
            for s in signals:
                s.set_mode(1)
            print("Signals: ", signals)

            sd_card = device.has_sd_card_capability()                
            print("SD card? ", sd_card)
            if sd_card:
                enabled = device.is_sd_card_enabled()
                print("SD card enabled? ", enabled)
                if not enabled:
                    device.enable_sd_card()
                    print("SD card enabled? ", device.is_sd_card_enabled())
                print("Folder: ", device.get_folder())
                device.set_folder("A/new/folder/path/to/file")
                print("Folder: ", device.get_folder())
                print("File: ", device.get_file())
                device.set_file("afile.sdf")
                print("File: ", device.get_file())

            sync = device.synchronize()
            print ("Synchronization: ", sync)

            device.start()
            print ("Started: ", device.is_running())
            print ("Data size: ", device.read_data_size())
            for i in range(100):
                sequence, battery, flags, data = device.read()
                print (len(data), sequence, battery, flags)
                print ("impedance: {}".format(device.get_eeg_impedance(0)))
            device.stop()
            print ("Stopped: ", not device.is_running())

            if not try_to(lambda: not device.is_connected(), device.disconnect, 10):
                print("unable to disconnect")
                exit(1)
            print("Disconnected")
    else:
        print("Usage: " + sys.argv[0] + " <device name>")
