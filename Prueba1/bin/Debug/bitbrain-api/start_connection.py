import sys
import numpy as np
import pandas as pd
from bbt import Signal, Device, SensorType, ImpedanceLevel
from circe.classification.methods import ClassParam
import record_to_csv



name = "BBT-E08-AAB039"

device = Device.create_bluetooth_device(name)

length = int(sys.argv[2]) if len(sys.argv) > 2 else 10
with Device.create_bluetooth_device(name) as device:
    if not record_to_csv.try_to(device.is_connected, device.connect, 10, "Conectando a la diadema: {}".format(name)):
        print("No se ha podido conectar con la diadema: {}".format(name))
        exit(1)
    print("Conexión establecida")

    print(f"Recogiendo {length} segundos de datos en archivos desde el dispositivo: {name}")
    record_to_csv.config_signals(device)
    eeg = record_to_csv.record_data(device, length)
    result = record_to_csv.call_model(eeg=eeg, feats_folder='models/feats01/',
                                    model_folder='models/model01')
    print('Inference result: {}'.format(result.loc[0, 'Target']))

    if not record_to_csv.try_to(lambda: not device.is_connected(), device.disconnect, 10):
        print("no es posible desconectar")
        exit(1)
    print("Conectado")

