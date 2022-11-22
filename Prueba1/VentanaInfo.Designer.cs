namespace Prueba1
{
    partial class VentanaInfo
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(VentanaInfo));
            this.label3 = new System.Windows.Forms.Label();
            this.barra_superior = new System.Windows.Forms.Panel();
            this.label_fecha = new System.Windows.Forms.Label();
            this.richTextBox1 = new System.Windows.Forms.RichTextBox();
            this.barra_superior.SuspendLayout();
            this.SuspendLayout();
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Segoe UI", 15F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.ForeColor = System.Drawing.Color.White;
            this.label3.Location = new System.Drawing.Point(156, 9);
            this.label3.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(238, 28);
            this.label3.TabIndex = 13;
            this.label3.Text = "BuruTSen - Información";
            this.label3.Click += new System.EventHandler(this.label3_Click);
            // 
            // barra_superior
            // 
            this.barra_superior.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(62)))), ((int)(((byte)(95)))), ((int)(((byte)(107)))));
            this.barra_superior.Controls.Add(this.label_fecha);
            this.barra_superior.Controls.Add(this.label3);
            this.barra_superior.Font = new System.Drawing.Font("Segoe UI", 15F);
            this.barra_superior.ForeColor = System.Drawing.Color.White;
            this.barra_superior.Location = new System.Drawing.Point(-2, -2);
            this.barra_superior.Margin = new System.Windows.Forms.Padding(2);
            this.barra_superior.Name = "barra_superior";
            this.barra_superior.Size = new System.Drawing.Size(539, 44);
            this.barra_superior.TabIndex = 14;
            // 
            // label_fecha
            // 
            this.label_fecha.AutoSize = true;
            this.label_fecha.Font = new System.Drawing.Font("Segoe UI", 15F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label_fecha.ForeColor = System.Drawing.Color.White;
            this.label_fecha.Location = new System.Drawing.Point(803, 10);
            this.label_fecha.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.label_fecha.Name = "label_fecha";
            this.label_fecha.Size = new System.Drawing.Size(51, 28);
            this.label_fecha.TabIndex = 13;
            this.label_fecha.Text = "date";
            // 
            // richTextBox1
            // 
            this.richTextBox1.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(107)))), ((int)(((byte)(164)))), ((int)(((byte)(184)))));
            this.richTextBox1.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.richTextBox1.Cursor = System.Windows.Forms.Cursors.No;
            this.richTextBox1.Font = new System.Drawing.Font("Segoe UI", 12F);
            this.richTextBox1.ForeColor = System.Drawing.Color.White;
            this.richTextBox1.Location = new System.Drawing.Point(29, 57);
            this.richTextBox1.Margin = new System.Windows.Forms.Padding(2);
            this.richTextBox1.Name = "richTextBox1";
            this.richTextBox1.ReadOnly = true;
            this.richTextBox1.Size = new System.Drawing.Size(501, 318);
            this.richTextBox1.TabIndex = 15;
            this.richTextBox1.Text = resources.GetString("richTextBox1.Text");
            this.richTextBox1.TextChanged += new System.EventHandler(this.richTextBox1_TextChanged);
            // 
            // VentanaInfo
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(107)))), ((int)(((byte)(164)))), ((int)(((byte)(184)))));
            this.ClientSize = new System.Drawing.Size(539, 434);
            this.Controls.Add(this.richTextBox1);
            this.Controls.Add(this.barra_superior);
            this.Margin = new System.Windows.Forms.Padding(2);
            this.Name = "VentanaInfo";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Información";
            this.barra_superior.ResumeLayout(false);
            this.barra_superior.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Panel barra_superior;
        public System.Windows.Forms.Label label_fecha;
        public System.Windows.Forms.RichTextBox richTextBox1;
    }
}