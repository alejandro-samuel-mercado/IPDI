"""
PDI - Tkinter + NumPy + Pillow + Matplotlib

Permite:

- Abrir una imagen.
- Ver la imagen de entrada a la izquierda.
- Ver la imagen procesada a la derecha.
- Aplicar operaciones PDI.
- Pasar la imagen procesada a la izquierda.
- Restaurar la imagen original.
- Guardar la imagen procesada.
- Ver el histograma de la imagen procesada.

La imagen se guarda internamente como NumPy
con valores normalizados entre 0 y 1.
"""


import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

from PIL import Image, ImageTk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# 2. FUNCIONES 
# ============================================================

def escala_grises(imagen):
    """
    Convierte una imagen RGB a escala de grises.

    Cada píxel:
        [R, G, B]

    se transforma en:

        promedio(R, G, B)

    La imagen sigue teniendo 3 canales para poder mostrarla
    fácilmente como RGB.
    """

    gris = imagen.mean(axis=2)

    resultado = np.stack(
        [gris, gris, gris],
        axis=2
    )

    return resultado


def solo_canal(imagen, canal):
    """
    Conserva solamente un canal RGB.

    0 -> Rojo
    1 -> Verde
    2 -> Azul
    """

    resultado = np.zeros_like(imagen)

    resultado[:, :, canal] = imagen[:, :, canal]

    return resultado


# ============================================================
# 3. APLICACIÓN PRINCIPAL
# ============================================================

class AppPDI:

    def __init__(self, ventana):

        self.ventana = ventana

        self.ventana.title(
            "PDI - Procesamiento Digital de Imágenes"
        )

        self.ventana.geometry(
            "1250x850"
        )
        # Imagen que abrió originalmente el usuario.
        self.imagen_original = None

        self.imagen_entrada = None

        # Resultado de la última operación.
        self.imagen_procesada = None

        self.crear_interfaz()


    # ========================================================
    # 4. CREAR INTERFAZ
    # ========================================================

    def crear_interfaz(self):

        # ----------------------------------------------------
        # Barra superior
        # ----------------------------------------------------

        barra = tk.Frame(
            self.ventana
        )

        barra.pack(
            side="top",
            fill="x",
            padx=10,
            pady=10
        )



        boton_abrir = tk.Button(
            barra,
            text="Abrir imagen",
            command=self.abrir_imagen
        )

        boton_abrir.pack(
            side="left",
            padx=5
        )


        boton_restaurar = tk.Button(
            barra,
            text="Restaurar original",
            command=self.restaurar
        )

        boton_restaurar.pack(
            side="left",
            padx=5
        )


        boton_usar_procesada = tk.Button(
            barra,
            text="Usar procesada como entrada",
            command=self.usar_procesada
        )

        boton_usar_procesada.pack(
            side="left",
            padx=5
        )


        boton_guardar = tk.Button(
            barra,
            text="Guardar procesada",
            command=self.guardar_procesada
        )

        boton_guardar.pack(
            side="left",
            padx=5
        )


        # ====================================================
        # ZONA CENTRAL
        # ====================================================

        centro = tk.Frame(
            self.ventana
        )

        centro.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )


        # ----------------------------------------------------
        # PANEL IZQUIERDO - ENTRADA
        # ----------------------------------------------------

        panel_entrada = tk.Frame(
            centro,
            bg="#eeeeee",
            bd=1,
            relief="solid"
        )

        panel_entrada.pack(
            side="left",
            fill="both",
            expand=True,
            padx=5
        )

        titulo_entrada = tk.Label(
            panel_entrada,
            text="IMAGEN DE ENTRADA",
            font=("Arial", 14, "bold"),
            bg="#eeeeee"
        )

        titulo_entrada.pack(
            pady=10
        )


        self.label_entrada = tk.Label(
            panel_entrada,
            text="Abrí una imagen para comenzar",
            bg="#dddddd"
        )

        self.label_entrada.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )


        # ----------------------------------------------------
        # PANEL CENTRAL - CONTROLES
        # ----------------------------------------------------

        controles = tk.Frame(
            centro
        )

        controles.pack(
            side="left",
            fill="y",
            padx=10
        )


        titulo_controles = tk.Label(
            controles,
            text="Operaciones PDI",
            font=("Arial", 14, "bold")
        )

        titulo_controles.pack(
            pady=15
        )


        # ----------------------------------------------------
        # Menú de operaciones
        # ----------------------------------------------------

        self.operacion = tk.StringVar()

        self.operacion.set(
            "Original"
        )


        opciones = [
            "Original",
            "Escala de grises",
            "Solo canal R",
            "Solo canal G",
            "Solo canal B"
        ]


        menu = tk.OptionMenu(
            controles,
            self.operacion,
            *opciones
        )

        menu.config(
            width=18
        )

        menu.pack(
            pady=10
        )

        boton_aplicar = tk.Button(
            controles,
            text="Aplicar operación",
            command=self.aplicar_operacion
        )

        boton_aplicar.pack(
            fill="x",
            pady=10
        )


        tk.Label(
            controles,
            text=""
        ).pack(
            pady=5
        )

        

        # ----------------------------------------------------
        # PANEL DERECHO - PROCESADA
        # ----------------------------------------------------

        panel_procesada = tk.Frame(
            centro,
            bg="#eeeeee",
            bd=1,
            relief="solid"
        )

        panel_procesada.pack(
            side="left",
            fill="both",
            expand=True,
            padx=5
        )


        titulo_procesada = tk.Label(
            panel_procesada,
            text="IMAGEN PROCESADA",
            font=("Arial", 14, "bold"),
            bg="#eeeeee"
        )

        titulo_procesada.pack(
            pady=10
        )


        self.label_procesada = tk.Label(
            panel_procesada,
            text="El resultado aparecerá aquí",
            bg="#dddddd"
        )

        self.label_procesada.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )


        # ====================================================
        # HISTOGRAMA
        # ====================================================

        panel_histograma = tk.Frame(
            self.ventana,
            bd=1,
            relief="solid"
        )

        panel_histograma.pack(
            fill="x",
            padx=15,
            pady=10
        )


        titulo_histograma = tk.Label(
            panel_histograma,
            text="Histograma de la imagen procesada",
            font=("Arial", 12, "bold")
        )

        titulo_histograma.pack(
            pady=5
        )



        self.figura = Figure(
            figsize=(8, 2.5),
            dpi=100
        )

        self.ax = self.figura.add_subplot(111)


        # Canvas que permite insertar Matplotlib dentro
        # de Tkinter.
        self.canvas_histograma = FigureCanvasTkAgg(
            self.figura,
            master=panel_histograma
        )

        self.canvas_histograma.draw()

        self.canvas_histograma.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )


        # ----------------------------------------------------
        # Estado
        # ----------------------------------------------------

        self.estado = tk.Label(
            self.ventana,
            text="Listo.",
            anchor="w"
        )

        self.estado.pack(
            side="bottom",
            fill="x",
            padx=10,
            pady=5
        )


    # ========================================================
    # 5. ABRIR IMAGEN
    # ========================================================

    def abrir_imagen(self):

        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                (
                    "Imágenes",
                    "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"
                )
            ]
        )


        if not ruta:
            return


        try:

            imagen_pil = Image.open(
                ruta
            ).convert("RGB")


            # Pillow -> NumPy
            #
            # Los valores pasan de:
            #
            #     0-255
            #
            # a:
            #
            #     0-1

            imagen = (
                np.array(imagen_pil)
                / 255.0
            )


            # Guardamos tres versiones lógicas:

            self.imagen_original = imagen.copy()

            self.imagen_entrada = imagen.copy()

            self.imagen_procesada = None


            # Mostrar la original como entrada.
            self.mostrar_imagen(
                self.label_entrada,
                self.imagen_entrada
            )


    
            self.estado.config(
                text="Imagen cargada correctamente."
            )


        except Exception as error:

            messagebox.showerror(
                "Error",
                f"No se pudo abrir la imagen.\n\n{error}"
            )


    # ========================================================
    # 6. APLICAR OPERACIÓN
    # ========================================================

    def aplicar_operacion(self):

        if self.imagen_entrada is None:

            messagebox.showwarning(
                "Atención",
                "Primero abrí una imagen."
            )

            return


        # Obtenemos la opción elegida.

        operacion = self.operacion.get()


        # ----------------------------------------------------
        # Imagen original sin cambios
        # ----------------------------------------------------

        if operacion == "Original":

            resultado = self.imagen_entrada.copy()


        # ----------------------------------------------------
        # Escala de grises
        # ----------------------------------------------------

        elif operacion == "Escala de grises":

            resultado = escala_grises(
                self.imagen_entrada
            )


        # ----------------------------------------------------
        # Canal rojo
        # ----------------------------------------------------

        elif operacion == "Solo canal R":

            resultado = solo_canal(
                self.imagen_entrada,
                0
            )


        # ----------------------------------------------------
        # Canal verde
        # ----------------------------------------------------

        elif operacion == "Solo canal G":

            resultado = solo_canal(
                self.imagen_entrada,
                1
            )


        # ----------------------------------------------------
        # Canal azul
        # ----------------------------------------------------

        elif operacion == "Solo canal B":

            resultado = solo_canal(
                self.imagen_entrada,
                2
            )


        # Guardar el resultado.

        self.imagen_procesada = resultado



        self.mostrar_imagen(
            self.label_procesada,
            self.imagen_procesada
        )



        self.actualizar_histograma(
            self.imagen_procesada
        )


        self.estado.config(
            text=f"Operación aplicada: {operacion}"
        )


    # ========================================================
    # 7. USAR LA PROCESADA COMO NUEVA ENTRADA
    # ========================================================

    def usar_procesada(self):

        if self.imagen_procesada is None:

            messagebox.showwarning(
                "Atención",
                "Todavía no hay una imagen procesada."
            )

            return


        # Se crea una copia independiente.
        #
        # A partir de ahora las siguientes operaciones
        # trabajarán sobre esta imagen.

        self.imagen_entrada = (
            self.imagen_procesada.copy()
        )


        self.mostrar_imagen(
            self.label_entrada,
            self.imagen_entrada
        )


        self.estado.config(
            text="La imagen procesada ahora es la imagen de entrada."
        )


    # ========================================================
    # 8. RESTAURAR ORIGINAL
    # ========================================================

    def restaurar(self):

        if self.imagen_original is None:

            messagebox.showwarning(
                "Atención",
                "Primero abrí una imagen."
            )

            return



        self.imagen_entrada = (
            self.imagen_original.copy()
        )


        self.mostrar_imagen(
            self.label_entrada,
            self.imagen_entrada
        )


        self.estado.config(
            text="Imagen original restaurada en la entrada."
        )


    # ========================================================
    # 9. MOSTRAR UNA IMAGEN
    # ========================================================

    def mostrar_imagen(self, label, array_imagen):

        imagen_uint8 = (
            np.clip(
                array_imagen,
                0,
                1
            ) * 255
        ).astype(np.uint8)


        # NumPy -> Pillow

        imagen_pil = Image.fromarray(
            imagen_uint8
        )



        imagen_pil.thumbnail(
            (520, 420)
        )


        # Pillow -> Tkinter

        foto = ImageTk.PhotoImage(
            imagen_pil
        )


        # Guardamos la referencia para que Tkinter
        # mantenga viva la imagen.

        label.foto = foto



        label.config(
            image=foto,
            text=""
        )


    # ========================================================
    # 10. GUARDAR IMAGEN PROCESADA
    # ========================================================

    def guardar_procesada(self):

        if self.imagen_procesada is None:

            messagebox.showwarning(
                "Atención",
                "No hay una imagen procesada para guardar."
            )

            return


        ruta = filedialog.asksaveasfilename(
            title="Guardar imagen procesada",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp")
            ]
        )


        if not ruta:
            return


        try:

            # Convertimos de 0-1 a uint8 0-255.

            imagen_uint8 = (
                np.clip(
                    self.imagen_procesada,
                    0,
                    1
                ) * 255
            ).astype(np.uint8)


            # NumPy -> Pillow

            imagen_pil = Image.fromarray(
                imagen_uint8
            )


            # Guardamos en disco.

            imagen_pil.save(
                ruta
            )


            self.estado.config(
                text=f"Imagen guardada: {ruta}"
            )


        except Exception as error:

            messagebox.showerror(
                "Error",
                f"No se pudo guardar la imagen.\n\n{error}"
            )


    # ========================================================
    # 11. HISTOGRAMA
    # ========================================================

    def actualizar_histograma(self, imagen):

        self.ax.clear()

    
        imagen_uint8 = (
            np.clip(
                imagen,
                0,
                1
            ) * 255
        ).astype(np.uint8)



        if imagen_uint8.ndim == 3:

            # Extraemos los tres canales.
            #
            # [:, :, 0] -> rojo
            # [:, :, 1] -> verde
            # [:, :, 2] -> azul

            rojo = imagen_uint8[:, :, 0].ravel()
            verde = imagen_uint8[:, :, 1].ravel()
            azul = imagen_uint8[:, :, 2].ravel()




            self.ax.hist(
                rojo,
                bins=256,
                range=(0, 255),
                color="red",
                alpha=0.5,
                label="R"
            )


            # Canal verde.

            self.ax.hist(
                verde,
                bins=256,
                range=(0, 255),
                color="green",
                alpha=0.5,
                label="G"
            )


            # Canal azul.

            self.ax.hist(
                azul,
                bins=256,
                range=(0, 255),
                color="blue",
                alpha=0.5,
                label="B"
            )


            self.ax.legend()


        # ----------------------------------------------------
        # Imagen de un solo canal
        # ----------------------------------------------------

        else:

            valores = imagen_uint8.ravel()

            self.ax.hist(
                valores,
                bins=256,
                range=(0, 255),
                color="gray"
            )


        # ----------------------------------------------------
        # Configuración del gráfico
        # ----------------------------------------------------

        self.ax.set_title(
            "Histograma de intensidades"
        )

        self.ax.set_xlabel(
            "Intensidad"
        )

        self.ax.set_ylabel(
            "Frecuencia"
        )

        self.ax.set_xlim(
            0,
            255
        )

        self.ax.grid(
            alpha=0.2
        )


        #
        self.figura.tight_layout()

   

        self.canvas_histograma.draw()



if __name__ == "__main__":

    ventana = tk.Tk()

    app = AppPDI(
        ventana
    )

    ventana.mainloop()