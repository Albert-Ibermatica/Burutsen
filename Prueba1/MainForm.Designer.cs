namespace Prueba1
{
    partial class MainForm
    {
        /// <summary>
        /// Variable del diseñador necesaria.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Limpiar los recursos que se estén usando.
        /// </summary>
        /// <param name="disposing">true si los recursos administrados se deben desechar; false en caso contrario.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }
            
        #region Código generado por el Diseñador de Windows Forms

        /// <summary>
        /// Método necesario para admitir el Diseñador. No se puede modificar
        /// el contenido de este método con el editor de código.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            this.btn_iniciar_prediccion = new System.Windows.Forms.Button();
            this.log_aplicacion = new System.Windows.Forms.RichTextBox();
            this.btn_historial = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.texto_resultado = new System.Windows.Forms.Label();
            this.logo_ibermatica = new System.Windows.Forms.PictureBox();
            this.icono_informacion = new System.Windows.Forms.PictureBox();
            this.imagen_resultado = new System.Windows.Forms.PictureBox();
            this.barra_superior = new System.Windows.Forms.Panel();
            this.label_fecha = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label_estado = new System.Windows.Forms.Label();
            this.icono_reiniciar = new System.Windows.Forms.PictureBox();
            this.link_manual = new System.Windows.Forms.LinkLabel();
            ((System.ComponentModel.ISupportInitialize)(this.logo_ibermatica)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.icono_informacion)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.imagen_resultado)).BeginInit();
            this.barra_superior.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.icono_reiniciar)).BeginInit();
            this.SuspendLayout();
            // 
            // btn_iniciar_prediccion
            // 
            this.btn_iniciar_prediccion.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(62)))), ((int)(((byte)(95)))), ((int)(((byte)(107)))));
            this.btn_iniciar_prediccion.FlatAppearance.BorderSize = 0;
            this.btn_iniciar_prediccion.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btn_iniciar_prediccion.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btn_iniciar_prediccion.ForeColor = System.Drawing.Color.WhiteSmoke;
            this.btn_iniciar_prediccion.Location = new System.Drawing.Point(330, 592);
            this.btn_iniciar_prediccion.Margin = new System.Windows.Forms.Padding(4);
            this.btn_iniciar_prediccion.Name = "btn_iniciar_prediccion";
            this.btn_iniciar_prediccion.Size = new System.Drawing.Size(257, 51);
            this.btn_iniciar_prediccion.TabIndex = 0;
            this.btn_iniciar_prediccion.Text = "Iniciar prediccion";
            this.btn_iniciar_prediccion.UseVisualStyleBackColor = false;
            this.btn_iniciar_prediccion.Click += new System.EventHandler(this.button1_Click);
            // 
            // log_aplicacion
            // 
            this.log_aplicacion.BackColor = System.Drawing.Color.Gray;
            this.log_aplicacion.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.log_aplicacion.Font = new System.Drawing.Font("Courier New", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.log_aplicacion.ForeColor = System.Drawing.Color.White;
            this.log_aplicacion.Location = new System.Drawing.Point(330, 475);
            this.log_aplicacion.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.log_aplicacion.Name = "log_aplicacion";
            this.log_aplicacion.Size = new System.Drawing.Size(559, 98);
            this.log_aplicacion.TabIndex = 3;
            this.log_aplicacion.Text = "";
            // 
            // btn_historial
            // 
            this.btn_historial.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(62)))), ((int)(((byte)(95)))), ((int)(((byte)(107)))));
            this.btn_historial.FlatAppearance.BorderSize = 0;
            this.btn_historial.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btn_historial.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btn_historial.ForeColor = System.Drawing.Color.White;
            this.btn_historial.Location = new System.Drawing.Point(623, 592);
            this.btn_historial.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.btn_historial.Name = "btn_historial";
            this.btn_historial.Size = new System.Drawing.Size(266, 51);
            this.btn_historial.TabIndex = 5;
            this.btn_historial.Text = "Historial predicciones";
            this.btn_historial.UseVisualStyleBackColor = false;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Segoe UI", 15F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.ForeColor = System.Drawing.Color.White;
            this.label2.Location = new System.Drawing.Point(492, 133);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(201, 28);
            this.label2.TabIndex = 6;
            this.label2.Text = "Resultado del análisis:";
            this.label2.Click += new System.EventHandler(this.label2_Click);
            // 
            // texto_resultado
            // 
            this.texto_resultado.AutoSize = true;
            this.texto_resultado.Font = new System.Drawing.Font("Segoe UI", 15F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.texto_resultado.ForeColor = System.Drawing.Color.White;
            this.texto_resultado.Location = new System.Drawing.Point(492, 203);
            this.texto_resultado.Name = "texto_resultado";
            this.texto_resultado.Size = new System.Drawing.Size(142, 28);
            this.texto_resultado.TabIndex = 7;
            this.texto_resultado.Text = "Alimento:      --";
            this.texto_resultado.Click += new System.EventHandler(this.texto_resultado_Click);
            // 
            // logo_ibermatica
            // 
            this.logo_ibermatica.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("logo_ibermatica.BackgroundImage")));
            this.logo_ibermatica.BackgroundImageLayout = System.Windows.Forms.ImageLayout.None;
            this.logo_ibermatica.Cursor = System.Windows.Forms.Cursors.SizeNS;
            this.logo_ibermatica.InitialImage = ((System.Drawing.Image)(resources.GetObject("logo_ibermatica.InitialImage")));
            this.logo_ibermatica.Location = new System.Drawing.Point(1052, 653);
            this.logo_ibermatica.Name = "logo_ibermatica";
            this.logo_ibermatica.Size = new System.Drawing.Size(135, 63);
            this.logo_ibermatica.TabIndex = 8;
            this.logo_ibermatica.TabStop = false;
            // 
            // icono_informacion
            // 
            this.icono_informacion.Image = ((System.Drawing.Image)(resources.GetObject("icono_informacion.Image")));
            this.icono_informacion.Location = new System.Drawing.Point(12, 674);
            this.icono_informacion.Name = "icono_informacion";
            this.icono_informacion.Size = new System.Drawing.Size(42, 42);
            this.icono_informacion.SizeMode = System.Windows.Forms.PictureBoxSizeMode.CenterImage;
            this.icono_informacion.TabIndex = 9;
            this.icono_informacion.TabStop = false;
            // 
            // imagen_resultado
            // 
            this.imagen_resultado.Image = ((System.Drawing.Image)(resources.GetObject("imagen_resultado.Image")));
            this.imagen_resultado.Location = new System.Drawing.Point(475, 243);
            this.imagen_resultado.Name = "imagen_resultado";
            this.imagen_resultado.Size = new System.Drawing.Size(244, 184);
            this.imagen_resultado.SizeMode = System.Windows.Forms.PictureBoxSizeMode.CenterImage;
            this.imagen_resultado.TabIndex = 10;
            this.imagen_resultado.TabStop = false;
            // 
            // barra_superior
            // 
            this.barra_superior.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(62)))), ((int)(((byte)(95)))), ((int)(((byte)(107)))));
            this.barra_superior.Controls.Add(this.label_fecha);
            this.barra_superior.Controls.Add(this.label3);
            this.barra_superior.Controls.Add(this.label_estado);
            this.barra_superior.Controls.Add(this.icono_reiniciar);
            this.barra_superior.Font = new System.Drawing.Font("Segoe UI", 15F);
            this.barra_superior.ForeColor = System.Drawing.Color.White;
            this.barra_superior.Location = new System.Drawing.Point(0, 0);
            this.barra_superior.Name = "barra_superior";
            this.barra_superior.Size = new System.Drawing.Size(1198, 53);
            this.barra_superior.TabIndex = 11;
            this.barra_superior.Paint += new System.Windows.Forms.PaintEventHandler(this.barra_superior_Paint);
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
            this.label3.Location = new System.Drawing.Point(493, 12);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(178, 28);
            this.label3.TabIndex = 12;
            this.label3.Text = "BuruTSen - Client";
            this.label3.Click += new System.EventHandler(this.label3_Click);
            // 
            // label_estado
            // 
            this.label_estado.AutoSize = true;
            this.label_estado.Font = new System.Drawing.Font("Segoe UI", 15F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label_estado.ForeColor = System.Drawing.Color.White;
            this.label_estado.Location = new System.Drawing.Point(85, 10);
            this.label_estado.Name = "label_estado";
            this.label_estado.Size = new System.Drawing.Size(202, 28);
            this.label_estado.TabIndex = 11;
            this.label_estado.Text = "Estado: desconectado";
            this.label_estado.Click += new System.EventHandler(this.label1_Click_1);
            // 
            // icono_reiniciar
            // 
            this.icono_reiniciar.Image = ((System.Drawing.Image)(resources.GetObject("icono_reiniciar.Image")));
            this.icono_reiniciar.Location = new System.Drawing.Point(12, 10);
            this.icono_reiniciar.Name = "icono_reiniciar";
            this.icono_reiniciar.Size = new System.Drawing.Size(42, 39);
            this.icono_reiniciar.TabIndex = 10;
            this.icono_reiniciar.TabStop = false;
            // 
            // link_manual
            // 
            this.link_manual.AutoSize = true;
            this.link_manual.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.link_manual.LinkColor = System.Drawing.Color.White;
            this.link_manual.Location = new System.Drawing.Point(60, 687);
            this.link_manual.Name = "link_manual";
            this.link_manual.Size = new System.Drawing.Size(90, 20);
            this.link_manual.TabIndex = 17;
            this.link_manual.TabStop = true;
            this.link_manual.Text = "Ver manual";
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 17F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(107)))), ((int)(((byte)(164)))), ((int)(((byte)(184)))));
            this.ClientSize = new System.Drawing.Size(1199, 728);
            this.Controls.Add(this.link_manual);
            this.Controls.Add(this.barra_superior);
            this.Controls.Add(this.imagen_resultado);
            this.Controls.Add(this.icono_informacion);
            this.Controls.Add(this.logo_ibermatica);
            this.Controls.Add(this.texto_resultado);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.btn_historial);
            this.Controls.Add(this.log_aplicacion);
            this.Controls.Add(this.btn_iniciar_prediccion);
            this.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.Name = "MainForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Bitbrain - Client";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.logo_ibermatica)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.icono_informacion)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.imagen_resultado)).EndInit();
            this.barra_superior.ResumeLayout(false);
            this.barra_superior.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.icono_reiniciar)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        public System.Windows.Forms.Button btn_iniciar_prediccion;
        public System.Windows.Forms.RichTextBox log_aplicacion;
        public System.Windows.Forms.Button btn_historial;
        private System.Windows.Forms.Label label2;
        public System.Windows.Forms.PictureBox logo_ibermatica;
        public System.Windows.Forms.PictureBox imagen_resultado;
        private System.Windows.Forms.Panel barra_superior;
        private System.Windows.Forms.Label label3;
        public System.Windows.Forms.Label texto_resultado;
        public System.Windows.Forms.PictureBox icono_informacion;
        public System.Windows.Forms.PictureBox icono_reiniciar;
        public System.Windows.Forms.Label label_fecha;
        public System.Windows.Forms.Label label_estado;
        public System.Windows.Forms.LinkLabel link_manual;
    }
}

