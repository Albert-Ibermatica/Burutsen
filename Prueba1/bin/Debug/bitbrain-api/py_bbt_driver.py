# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _py_bbt_driver
else:
    import _py_bbt_driver

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)



def bbt_signal_get_type(signal, output_buffer):
    return _py_bbt_driver.bbt_signal_get_type(signal, output_buffer)

def bbt_signal_get_channels(signal):
    return _py_bbt_driver.bbt_signal_get_channels(signal)

def bbt_signal_get_samples(signal):
    return _py_bbt_driver.bbt_signal_get_samples(signal)

def bbt_signal_get_mode(signal):
    return _py_bbt_driver.bbt_signal_get_mode(signal)

def bbt_signal_set_mode(signal, mode):
    return _py_bbt_driver.bbt_signal_set_mode(signal, mode)

def bbt_driver_new_bluetooth(device_name, eeg_sensor_type):
    return _py_bbt_driver.bbt_driver_new_bluetooth(device_name, eeg_sensor_type)

def bbt_driver_new_usb(port, eeg_sensor_type):
    return _py_bbt_driver.bbt_driver_new_usb(port, eeg_sensor_type)

def bbt_driver_new_eeg64(left_device_name, right_device_name):
    return _py_bbt_driver.bbt_driver_new_eeg64(left_device_name, right_device_name)

def bbt_driver_free(driver):
    return _py_bbt_driver.bbt_driver_free(driver)

def bbt_driver_connect(driver):
    return _py_bbt_driver.bbt_driver_connect(driver)

def bbt_driver_disconnect(driver):
    return _py_bbt_driver.bbt_driver_disconnect(driver)

def bbt_driver_reconnect(driver):
    return _py_bbt_driver.bbt_driver_reconnect(driver)

def bbt_driver_is_connected(driver):
    return _py_bbt_driver.bbt_driver_is_connected(driver)

def bbt_driver_get_hw_version(driver):
    return _py_bbt_driver.bbt_driver_get_hw_version(driver)

def bbt_driver_get_fw_version(driver):
    return _py_bbt_driver.bbt_driver_get_fw_version(driver)

def bbt_driver_get_frequency(driver):
    return _py_bbt_driver.bbt_driver_get_frequency(driver)

def bbt_driver_get_number_of_signals(driver):
    return _py_bbt_driver.bbt_driver_get_number_of_signals(driver)

def bbt_driver_get_signal(driver, index):
    return _py_bbt_driver.bbt_driver_get_signal(driver, index)

def bbt_driver_has_sd_card_capability(driver):
    return _py_bbt_driver.bbt_driver_has_sd_card_capability(driver)

def bbt_driver_is_sd_card_enabled(driver):
    return _py_bbt_driver.bbt_driver_is_sd_card_enabled(driver)

def bbt_driver_enable_sd_card(driver, enable):
    return _py_bbt_driver.bbt_driver_enable_sd_card(driver, enable)

def bbt_driver_get_folder(driver, output_buffer):
    return _py_bbt_driver.bbt_driver_get_folder(driver, output_buffer)

def bbt_driver_set_folder(driver, folder):
    return _py_bbt_driver.bbt_driver_set_folder(driver, folder)

def bbt_driver_get_file(driver, output_buffer):
    return _py_bbt_driver.bbt_driver_get_file(driver, output_buffer)

def bbt_driver_set_file(driver, file):
    return _py_bbt_driver.bbt_driver_set_file(driver, file)

def bbt_driver_synchronize(driver):
    return _py_bbt_driver.bbt_driver_synchronize(driver)

def bbt_driver_start(driver):
    return _py_bbt_driver.bbt_driver_start(driver)

def bbt_driver_stop(driver):
    return _py_bbt_driver.bbt_driver_stop(driver)

def bbt_driver_is_running(driver):
    return _py_bbt_driver.bbt_driver_is_running(driver)

def bbt_driver_read_data_size(driver):
    return _py_bbt_driver.bbt_driver_read_data_size(driver)

def bbt_driver_read(driver):
    return _py_bbt_driver.bbt_driver_read(driver)

def bbt_driver_get_eeg_impedance(driver, index):
    return _py_bbt_driver.bbt_driver_get_eeg_impedance(driver, index)

def bbt_driver_import_sd_record(input_folder, input_file, output_folder):
    return _py_bbt_driver.bbt_driver_import_sd_record(input_folder, input_file, output_folder)

def bbt_driver_import_eeg64_sd_record(left_input_folder, right_input_folder, input_file, output_folder):
    return _py_bbt_driver.bbt_driver_import_eeg64_sd_record(left_input_folder, right_input_folder, input_file, output_folder)

cvar = _py_bbt_driver.cvar
bbt_dry_eeg_sensor = cvar.bbt_dry_eeg_sensor
bbt_water_eeg_sensor = cvar.bbt_water_eeg_sensor
bbt_driver_battery_unknown = cvar.bbt_driver_battery_unknown
bbt_driver_battery_charging = cvar.bbt_driver_battery_charging
bbt_driver_flags_ok = cvar.bbt_driver_flags_ok
bbt_driver_flags_sd_card_removed = cvar.bbt_driver_flags_sd_card_removed
bbt_driver_flags_sd_card_access_error = cvar.bbt_driver_flags_sd_card_access_error
bbt_driver_flags_sd_card_folder_error = cvar.bbt_driver_flags_sd_card_folder_error
bbt_driver_flags_sd_card_time_error = cvar.bbt_driver_flags_sd_card_time_error
bbt_driver_flags_sd_card_file_error = cvar.bbt_driver_flags_sd_card_file_error
bbt_driver_flags_sd_card_full_error = cvar.bbt_driver_flags_sd_card_full_error
bbt_driver_flags_eeg64_not_synchronized = cvar.bbt_driver_flags_eeg64_not_synchronized
bbt_driver_flags_bad_anchor_configuration = cvar.bbt_driver_flags_bad_anchor_configuration
bbt_driver_impedance_unknown = cvar.bbt_driver_impedance_unknown
bbt_driver_impedance_saturated = cvar.bbt_driver_impedance_saturated
bbt_driver_impedance_bad = cvar.bbt_driver_impedance_bad
bbt_driver_impedance_fair = cvar.bbt_driver_impedance_fair
bbt_driver_impedance_good = cvar.bbt_driver_impedance_good

