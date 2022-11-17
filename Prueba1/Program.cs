using System;
using System.Windows.Forms;

namespace Prueba1
{
    internal static class Program
    {
        /// <summary>
        /// Punto de entrada principal para la aplicación.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            MainFormController mfc = new MainFormController();
            Application.Run();

        }


    }
}
