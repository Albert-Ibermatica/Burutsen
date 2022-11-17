namespace Prueba1
{
    partial class VistaHistorialResultados
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.barra_superior = new System.Windows.Forms.Panel();
            this.label_fecha = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.barra_superior.SuspendLayout();
            this.SuspendLayout();
            // 
            // barra_superior
            // 
            this.barra_superior.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(62)))), ((int)(((byte)(95)))), ((int)(((byte)(107)))));
            this.barra_superior.Controls.Add(this.label_fecha);
            this.barra_superior.Controls.Add(this.label3);
            this.barra_superior.Font = new System.Drawing.Font("Segoe UI", 15F);
            this.barra_superior.ForeColor = System.Drawing.Color.White;
            this.barra_superior.Location = new System.Drawing.Point(0, 0);
            this.barra_superior.Name = "barra_superior";
            this.barra_superior.Size = new System.Drawing.Size(530, 53);
            this.barra_superior.TabIndex = 12;
            // 
            // label_fecha
            // 
            this.label_fecha.AutoSize = true;
            this.label_fecha.Font = new System.Drawing.Font("Segoe UI", 15F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label_fecha.ForeColor = System.Drawing.Color.White;
            this.label_fecha.Location = new System.Drawing.Point(1071, 12);
            this.label_fecha.Name = "label_fecha";
            this.label_fecha.Size = new System.Drawing.Size(51, 28);
            this.label_fecha.TabIndex = 13;
            this.label_fecha.Text = "date";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Segoe UI", 15F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.ForeColor = System.Drawing.Color.White;
            this.label3.Location = new System.Drawing.Point(149, 9);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(210, 28);
            this.label3.TabIndex = 12;
            this.label3.Text = "BuruTSen - Historial ";
            this.label3.Click += new System.EventHandler(this.label3_Click);
            // 
            // VistaHistorialResultados
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(107)))), ((int)(((byte)(164)))), ((int)(((byte)(184)))));
            this.ClientSize = new System.Drawing.Size(531, 465);
            this.Controls.Add(this.barra_superior);
            this.Name = "VistaHistorialResultados";
            this.Text = "VistaHistorialResultados";
            this.Load += new System.EventHandler(this.VistaHistorialResultados_Load);
            this.barra_superior.ResumeLayout(false);
            this.barra_superior.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel barra_superior;
        public System.Windows.Forms.Label label_fecha;
        private System.Windows.Forms.Label label3;
    }
}