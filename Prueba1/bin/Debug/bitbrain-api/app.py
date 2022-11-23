import sys
import pandas as pd
from flask import Flask
from flask_socketio import SocketIO

import record_to_csv
from bbt import Device

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route("/status", methods=["GET"])
def setName():
    return "running!!!"


@socketio.on('mensaje')
def handle_message(data):
    # emitimos "msg" para enviar un mensaje al servidor.
    socketio.emit("msg", "api funcionando en localhost ...")
    socketio.emit("msg","modelo cargado ...")
    print(data)


@socketio.on('connect')
def handle_connection(data):
    print(data)
    #socketio.emit("result", 0)
    #print('conexion establecida')

@socketio.on('is_connected')
def isconnected():
    print("metodo isconnected")
    name = "BBT-E08-AAB039"
    with Device.create_bluetooth_device(name) as device:
        if device.is_connected:
            print("---- isconnected true -----")
            socketio.emit("isconnected_result", "bbt-connected")
        else:
            print("---- isconnected false -----")
            socketio.emit("isconnected_result", "bbt-disconnected")
@socketio.on('get_prediction')
def startPrediction():
    print("servidor-socket start prediction method.")
    name = "BBT-E08-AAB039"
    device = Device.create_bluetooth_device(name)
    length = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    with Device.create_bluetooth_device(name) as device:
        if not record_to_csv.try_to(device.is_connected, device.connect, 10,socketio,
                                    "Conectando a la diadema: {}".format(name)):
            
            socketio.emit("msg", "No se ha podido conectar con la diadema: {}".format(name))
            socketio.emit("error", "no se puede conectar dispositivo:{} revisa que el dispositivo esta conectado a traves de bluetooth".format(name))
            exit(1)
        print("server-Conexión establecida")
        socketio.emit("msg", "Conexion con BBT-E08-AAB039 establecida")
        #print(f"Recogiendo {length} segundos de datos en archivos desde el dispositivo: {name}")
        socketio.emit("msg", f"Recogiendo {length} segundos de datos en archivos desde el dispositivo: {name}")
        socketio.emit("msg", "Configurando señales ...")
        record_to_csv.config_signals(device)
        socketio.emit("msg", "Iniciando grabacion de datos ...")
        eeg = record_to_csv.record_data(device, length)
        result = record_to_csv.call_model(eeg=eeg, feats_folder='C:/Users/Showroom3/Desktop/BuruTsen/Prueba1/bin/Debug/bitbrain-api/models/feats01/',
                                          model_folder='C:/Users/Showroom3/Desktop/BuruTsen/Prueba1/bin/Debug/bitbrain-api/models/model01/')
        print("esto es el print ")
        print(result)
        result_string= pd.DataFrame.to_string(result)
        socketio.emit("table_data","-------- tabla de datos ----------")
        socketio.emit("table_data",result_string)
        print(result.loc[0, 'm10-svc:poly'])
        print('Inference result: {}'.format(result.loc[0, 'm10-svc:poly']))
        socketio.emit("msg", 'Inference result: {}'.format(result.loc[0, 'm10-svc:poly']))
        socketio.emit("msg", result.loc[0, 'm10-svc:poly'])
        socketio.emit("result", result.loc[0, 'm10-svc:poly'])
        print(result.loc[0, 'm10-svc:poly'])
        if not record_to_csv.try_to(lambda: not device.is_connected(), device.disconnect, 10, socketio):
            print("no es posible desconectar")
            socketio.emit("error", "no es posible desconectar")
            socketio.emit("msg", result.loc[0, 'm10-svc:poly'])
            exit(1)
        print("Conectado")
        # No devolvemos nada, las posibilidades estan cubiertas con los msg del socketio. 


if __name__ == '__main__':
        
        socketio.run(app, host="0.0.0.0", port=8080)
