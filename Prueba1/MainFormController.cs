using SocketIOClient;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace Prueba1
{
    public class MainFormController
    {
        private Timer timer1;
        public SocketIO websocket;
        public MainForm form;
        public Process websocket_process;
        public int ws_pid;
        private static List<Process> procesos = new List<Process>();

        public MainFormController()
        {
            try
            {
                runAPI();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
            try
            {
                websocket = new SocketIO("http://127.0.0.1:8080/");
            }
            catch (Exception ex)
            {
                MessageBox.Show("no se ha podido establecer conexion con la API", "Error de conexion con la API", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Console.WriteLine(ex.ToString());
            }

            form = new MainForm();

            form.FormBorderStyle = FormBorderStyle.FixedSingle;
            form.MaximizeBox = false;

            websocket_init();
            btn_inicia_onclickAsync();

            form.Show();

            form.btn_iniciar_prediccion.Click += new EventHandler(btn_info_onclickAsync);
            form.btn_historial.Click += Btn_historial_ClickAsync;
            form.icono_informacion.Click += new EventHandler(Icono_informacion_Click);
            form.icono_reiniciar.Click += Icono_reiniciar_Click;
            form.label_fecha.Text = "";
            form.label_fecha.Text = DateTime.Today.ToString("dd/MM/yyyy");

            form.FormClosed += new System.Windows.Forms.FormClosedEventHandler(onclose);


        }

        public async void Btn_historial_ClickAsync(object sender, EventArgs e)
        {
            // abrimos con notepad el archivo .txt donde esta el historial de predicciones
            // guardar_prediccion("Cacahuete", DateTime.Now);
            abrir_historial();

        }
        public void onclose(object sender, EventArgs e)
        {
            KillProcess(this.ws_pid); 
            Application.Exit();//cerramos programa y procesos hijos

        }

        public async void Icono_reiniciar_Click(object sender, EventArgs e)
        {
            Console.WriteLine("esto es el boton reiniciar");

            // al pulsar el boton de recargar actualizamos la lista de dispositivos
            // bluetooth para ver si la diadema esta conectada
            // comprobamos con el nombre del bluetooth
            form.log_aplicacion.Text = "";
            form.imagen_resultado.Load("..\\..\\icons\\nada-logo.png");
            form.texto_resultado.Text = "Alimento:      --";

        }

        public void Icono_informacion_Click(object sender, EventArgs e)
        {
            //Console.WriteLine("icono informacion click");
            //MessageBox.Show("BuruTsen es una tecnologia que permite analizar los impulsos electricos del cerebro para determinar" +
            //    "el gusto de los alimentos que estas consumiendo, a traves de una recogida de datos con una diadema- del tipo Dry EGG", "Informacion");
            VentanaInfo info = new VentanaInfo();
            info.Show();

        }

        public async void btn_info_onclickAsync(object sender, EventArgs e)

        {
            // limpiar el log
            form.log_aplicacion.Text = "";
            await websocket.EmitAsync("is_connected");
            Console.WriteLine(websocket.Connected.ToString());
            // comprobamos si esta conectado con el servidor si no lo esta,lo volvemos a intentar
            if (!websocket.Connected)
            {
                // intentamos establecer conexion de nuevo
                await websocket.ConnectAsync();
                MessageBox.Show("No se ha detectado conexion con API,reintentando.", "Error de conexion", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);

            }
            try
            {
                await websocket.EmitAsync("get_prediction");
            }
            catch (NullReferenceException err)
            {
                Console.WriteLine("Please check the string str.");
                MessageBox.Show("Error en la ejecucion.", "Error de conexion con la API", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Console.WriteLine(err.ToString());
                Console.WriteLine(err.Message);
            }
            catch (Exception ex)
            {
                Console.WriteLine("{0} Exception caught.", ex);
                MessageBox.Show("Error en la ejecucion.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Console.WriteLine(ex.ToString());

            }


        }

        public async void btn_inicia_onclickAsync()
        {


            websocket.OnConnected += async (send, x) =>
            {
                // Emit a string
                await websocket.EmitAsync("mensaje", "el cliente se ha conectado.");
                //Console.WriteLine("se ha conectado");
                await websocket.EmitAsync("is_connected");

            };
            await websocket.ConnectAsync();

        }

        public void btn_historial(object sender, EventArgs e)
        {
            Console.WriteLine("btn inicia clicado");

            abrir_historial();

        }

        public void websocket_init()
        {

            websocket.On("msg", response =>
            {
                // Console.WriteLine(response);
                String responseText = response.GetValue<String>();
                Console.WriteLine(responseText);
                if (form.log_aplicacion.InvokeRequired)
                {
                    form.log_aplicacion.Invoke(new Action(() =>

                    form.log_aplicacion.Text += responseText + "\n"
                    ));
                }
                else
                {
                    form.log_aplicacion.Text = responseText;
                }


            });

            websocket.On("result", response =>
            {
                Console.WriteLine("result onresponse");
                // aqui es cuando la prediccion ha sido exitosa.
                // int responseCode = response.GetValue<int>();
                var websocket_response = response.GetValue();
                Console.WriteLine(websocket_response);

                // caramelo
                if (websocket_response.ToString() == "0")
                {
                    if (form.imagen_resultado.InvokeRequired)
                    {
                        form.imagen_resultado.Invoke(new Action(() =>
                              form.imagen_resultado.Load("..\\..\\icons\\caramelo.png")
                            ));
                    }
                    if (form.texto_resultado.InvokeRequired)
                    {
                        form.texto_resultado.Invoke(new Action(() =>
                              form.texto_resultado.Text = "Alimento: Caramelo"
                            ));
                    }

                    guardar_prediccion("Caramelo", DateTime.Now);
                }

                // cacahuete
                if (websocket_response.ToString() == "1")
                {
                    if (form.imagen_resultado.InvokeRequired)
                    {
                        form.imagen_resultado.Invoke(new Action(() =>
                              form.imagen_resultado.Load("..\\..\\icons\\cacahuete.png")
                            ));
                    }
                    if (form.texto_resultado.InvokeRequired)
                    {
                        form.texto_resultado.Invoke(new Action(() =>
                              form.texto_resultado.Text = "Alimento: Cacahuete"
                            ));
                    }
                    guardar_prediccion("Cacahuete", DateTime.Now);
                }
            });

            websocket.On("isconnected_result", response =>
            {
                Console.WriteLine("metodo isconnected result");
                // aqui es cuando la prediccion ha sido exitosa.
                // int responseCode = response.GetValue<int>();
                var websocket_response = response.GetValue();

                Console.WriteLine(response.ToString());
                var string_response = websocket_response.ToString();

                if (string_response == "bbt-connected")
                {
                    Console.WriteLine("metodo diadema conectada");

                    if (form.label_estado.InvokeRequired)
                    {
                        form.label_estado.Invoke(new Action(() =>

                          form.label_estado.Text = "Estado: Conectado"

                        ));
                    }
                }
                // cambiamos el texto de conectado segun si esta o no conectado.
                if (string_response == "bbt-disconnected")
                {
                    Console.WriteLine("metodo diadema desconectada");
                    if (form.label_estado.InvokeRequired)
                    {
                        form.label_estado.Invoke(new Action(() =>

                          form.label_estado.Text = "Estado: Desconectado"

                        ));
                    }
                }
            });

            websocket.On("error", response =>
            {
                Console.WriteLine("error on response");
                // cuando el servidor nos devuelve un error.
                //String  responseCode = response.GetValue<String>();

                MessageBox.Show(response.GetValue().ToString(), "Mensaje de la API:", MessageBoxButtons.OK, MessageBoxIcon.Information);



            });
        }

        public void runAPI()
        {
            try
            {
                string filename = "run_bitbrain_api.bat";
                string parameters = $"/k \"{filename}\"";

                ProcessStartInfo startInfo = new ProcessStartInfo("cmd.exe");

                startInfo.WindowStyle = ProcessWindowStyle.Minimized;


                startInfo.Arguments = parameters;

               var websocket_process =  Process.Start(startInfo);
               this.ws_pid = StoreProcess(websocket_process);
                

            }
            catch (Exception e)
            {

                Console.WriteLine(e.ToString());
            }


        }

        public void escanear_bluetooth()
        {

            //BluetoothClient client = new BluetoothClient();
            //List<string> items = new List<string>();
            //BluetoothDeviceInfo[] devices = client.DiscoverDevices();
            //foreach (BluetoothDeviceInfo d in devices)
            //{
            //    items.Add(d.DeviceName);
            //    Console.WriteLine(d.DeviceName.ToString());
            //    // 
            //    if (d.DeviceName.ToString() == "BBT-E08-AAB039" && d.Connected)
            //    {
            //        Console.WriteLine("la diadema esta conectada.");
            //        form.label_estado.Text = "Estado: conectado";
            //    }
            //    if (d.DeviceName.ToString() == "BBT-E08-AAB039" && d.Connected == false)
            //    {
            //        Console.WriteLine("la diadema esta desconectada.");
            //        form.label_estado.Text = "Estado: desconectado";
            //    }

            //}
        }

        public void guardar_prediccion(string resultado, DateTime hora)
        {
            //File.path("C:\\Users\\iberm\\Desktop\\BuruTsen\\Prueba1\\historial.txt");
            // C:\\Users\\iberm\\Desktop\\BuruTsen\\Prueba1\\historial.txt
            string path = @"..\\..\\historial.txt";

            var hora_string = hora.ToString();

            String registro_prediccion = "Resultado: " + resultado + "  Fecha y hora: " + hora_string + " Modelo: Circe 1.1.2 - Tensorflow - Keras API" + Environment.NewLine;


            // This text is always added, making the file longer over time
            // if it is not deleted.
            using (StreamWriter sw = File.AppendText(path))
            {
                sw.WriteLine(registro_prediccion);
            }
            // leemos el archivo de texto.
            using (StreamReader sr = File.OpenText(path))
            {
                string s = "";
                while ((s = sr.ReadLine()) != null)
                {
                    Console.WriteLine(s);
                }
            }
        }

        public void abrir_historial()
        {
            System.Diagnostics.Process.Start("notepad.exe", "..\\..\\historial.txt");

        }

        public static int StoreProcess(Process prc)
        {
            int PID = prc.Id; // Get the process PID and store it in an int called PID
            procesos.Add(prc); // Add this to our list of processes to be kept track of
            return PID; // Return the PID so that the process can be killed/changed at a later time
        }

        public static void KillProcess(int PID)
        {
            // Search through the countless processes we have and try and find our process
            for (int i = 0; i <= procesos.Count; i++)
            {
                if (procesos[i] == null)
                {
                    continue; // This segment of code prevents NullPointerExceptions by checking if the process is null before doing anything with it
                }
                if (procesos[i].Id == PID)
                { // Is this our process?
                    procesos[i].Kill(); // It is! Lets kill it
                    while (!procesos[i].HasExited) { } // Wait until the process exits
                    procesos[i] = null; // Mark this process to be skipped the next time around
                    return;
                }
            }
            // Couldn't find our process!!!
            throw new Exception("Process not found!");
        }

    }
}