import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

COLOR_FONDO = "#E8EEF5"
COLOR_PANEL = "#F8FAFC"
COLOR_AREA_IMAGEN = "#DCE4EE"
COLOR_CONTROLES = "#EEF3F8"
COLOR_PRIMARIO = "#2563EB"
COLOR_PRIMARIO_HOVER = "#1D4ED8"
COLOR_TEXTO = "#1E293B"
COLOR_TEXTO_SECUNDARIO = "#64748B"
COLOR_BORDE = "#CBD5E1"
COLOR_BLANCO = "#FFFFFF"

# ============================================================
# FUNCIONES
# ============================================================

MATRIZ_RGB_YIQ = np.array(
    [
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312],
    ],
    dtype=np.float64,
)
# Se calcula la inversa a partir de la misma matriz para que la conversión
# de ida y vuelta sea consistente.
MATRIZ_YIQ_RGB = np.linalg.inv(MATRIZ_RGB_YIQ)

def procesar_imagen_rgb(imagen_rgb, ajuste_luminancia, factor_cromaticidad):
    """
    La imagen se mantiene normalizada entre 0 y 1. La luminancia del control
    se expresa en niveles de intensidad de una imagen de 8 bits (-5 a +5),
    por eso el ajuste se divide por 255 antes de sumarlo a Y.
    """
    # Validar incluso si la función se llama con valores externos al slider.
    ajuste_luminancia = float(np.clip(ajuste_luminancia, -5.0, 5.0))
    factor_cromaticidad = float(np.clip(factor_cromaticidad, 0.0, 2.0))
    imagen_rgb = np.asarray(imagen_rgb, dtype=np.float64)
    imagen_rgb = np.clip(imagen_rgb, 0.0, 1.0)
    alto, ancho, canales = imagen_rgb.shape
    if canales != 3:
        raise ValueError("La imagen debe tener exactamente 3 canales RGB.")
    pixeles_rgb = imagen_rgb.reshape(-1, 3)
    pixeles_yiq = pixeles_rgb @ MATRIZ_RGB_YIQ.T
    imagen_yiq = pixeles_yiq.reshape(alto, ancho, 3)

    Y = imagen_yiq[:, :, 0]
    I = imagen_yiq[:, :, 1]
    Q = imagen_yiq[:, :, 2]

    # Ajuste aditivo de brillo: -5..+5 niveles sobre la escala 0..255.
    Y_nuevo = np.clip(Y + ajuste_luminancia / 255.0, 0.0, 1.0)
    # Factor 0 = sin cromaticidad (grises), 1 = original, 2 = reforzada.
    I_nuevo = I * factor_cromaticidad
    Q_nuevo = Q * factor_cromaticidad

    imagen_yiq_ajustada = np.stack((Y_nuevo, I_nuevo, Q_nuevo), axis=2)

    pixeles_yiq_ajustados = imagen_yiq_ajustada.reshape(-1, 3)

    pixeles_rgb_resultado = pixeles_yiq_ajustados @ MATRIZ_YIQ_RGB.T

    imagen_rgb_resultado = pixeles_rgb_resultado.reshape(alto, ancho, 3)


    # Evita underflow/overflow de los canales al volver al gamut RGB válido.
    return np.clip(imagen_rgb_resultado, 0.0, 1.0)

class AppPDI:

    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title(
            "PDI - Procesamiento Digital de Imágenes"
        )
        self.ventana.geometry(
            "1250x850"
        )
        self.ventana.configure(
            bg=COLOR_FONDO
        )

        # ----------------------------------------------------
        # Datos de imágenes
        # ----------------------------------------------------

        self.imagen_original = None
        self.imagen_entrada = None
        self.imagen_procesada = None


        # Espacio de color actual.
        self.espacio_color = "RGB"


        # Valores iniciales.
        self.luminancia = 0
        self.cromaticidad = 0
        self.crear_interfaz()

    def crear_interfaz(self):

        # ====================================================
        # BARRA SUPERIOR
        # ====================================================

        barra = tk.Frame(
            self.ventana,
            bg=COLOR_FONDO
        )
        barra.pack(
            side="top",
            fill="x",
            padx=20,
            pady=(18, 10)
        )
        boton_abrir = tk.Button(
            barra,
            text="Abrir imagen",
            command=self.abrir_imagen,
            bg=COLOR_PRIMARIO,
            fg=COLOR_BLANCO,
            activebackground=COLOR_PRIMARIO_HOVER,
            activeforeground=COLOR_BLANCO,
            font=("Arial", 11, "bold"),
            padx=15,
            pady=8,
            relief="flat",
            bd=0,
            cursor="hand2"
        )
        boton_abrir.pack(
            side="left",
            padx=5
        )


        # Botón Restaurar
        boton_restaurar = tk.Button(
            barra,
            text="Restaurar original",
            command=self.restaurar,
            bg=COLOR_BLANCO,
            fg=COLOR_TEXTO,
            activebackground=COLOR_AREA_IMAGEN,
            activeforeground=COLOR_TEXTO,
            font=("Arial", 11, "bold"),
            padx=15,
            pady=8,
            relief="solid",
            bd=1,
            cursor="hand2"
        )
        boton_restaurar.pack(
            side="left",
            padx=5
        )


        # Botón usar imagen procesada como entrada
        boton_usar_procesada = tk.Button(
            barra,
            text="Usar procesada como entrada",
            command=self.usar_procesada,
            bg=COLOR_BLANCO,
            fg=COLOR_TEXTO,
            activebackground=COLOR_AREA_IMAGEN,
            activeforeground=COLOR_TEXTO,
            font=("Arial", 11, "bold"),
            padx=15,
            pady=8,
            relief="solid",
            bd=1,
            cursor="hand2"
        )
        boton_usar_procesada.pack(
            side="left",
            padx=5
        )


        # Botón guardar
        boton_guardar = tk.Button(
            barra,
            text="Guardar procesada",
            command=self.guardar_procesada,
            bg=COLOR_BLANCO,
            fg=COLOR_TEXTO,
            activebackground=COLOR_AREA_IMAGEN,
            activeforeground=COLOR_TEXTO,
            font=("Arial", 11, "bold"),
            padx=15,
            pady=8,
            relief="solid",
            bd=1,
            cursor="hand2"
        )
        boton_guardar.pack(
            side="left",
            padx=5
        )

        # ====================================================
        # ZONA CENTRAL
        # ====================================================

        centro = tk.Frame(
            self.ventana,
            bg=COLOR_FONDO
        )
        centro.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=10
        )

        # ====================================================
        # PANEL IZQUIERDO - ENTRADA
        # ====================================================

        panel_entrada = tk.Frame(
            centro,
            bg=COLOR_PANEL,
            bd=1,
            relief="solid"
        )
        panel_entrada.pack(
            side="left",
            fill="both",
            expand=True,
            padx=6
        )
        self.titulo_entrada = tk.Label(
            panel_entrada,
            text=(
                f"IMAGEN DE ENTRADA  •  "
                f"Espacio de color: {self.espacio_color}"
            ),
            font=("Arial", 13, "bold"),
            bg=COLOR_PANEL,
            fg=COLOR_TEXTO
        )
        self.titulo_entrada.pack(
            pady=(15, 10)
        )


        # Área de imagen izquierda
        self.label_entrada = tk.Label(
            panel_entrada,
            text="Abrí una imagen para comenzar",
            bg=COLOR_AREA_IMAGEN,
            fg=COLOR_TEXTO_SECUNDARIO,
            font=("Arial", 12),
            bd=0
        )
        self.label_entrada.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=(0, 12)
        )

        # ====================================================
        # PANEL CENTRAL - CONTROLES
        # ====================================================

        controles = tk.Frame(
            centro,
            bg=COLOR_CONTROLES,
            bd=1,
            relief="solid"
        )
        controles.pack(
            side="left",
            fill="y",
            padx=10,
            pady=0
        )
        titulo_controles = tk.Label(
            controles,
            text="Conversión de color",
            font=("Arial", 13, "bold"),
            bg=COLOR_CONTROLES,
            fg=COLOR_TEXTO
        )
        titulo_controles.pack(
            pady=(18, 5)
        )
        self.subtitulo_controles = tk.Label(
            controles,
            text="Ajustes internos en YIQ · salida RGB",
            font=("Arial", 10),
            bg=COLOR_CONTROLES,
            fg=COLOR_TEXTO_SECUNDARIO
        )
        self.subtitulo_controles.pack(
            pady=(0, 15)
        )



        # LUMINOSIDAD
        self.label_luminancia = tk.Label(
            controles,
            text="Luminosidad",
            font=("Arial", 10, "bold"),
            bg=COLOR_CONTROLES,
            fg=COLOR_TEXTO
        )
        self.label_luminancia.pack(
            pady=(8, 3)
        )
        self.luminancia = tk.DoubleVar(
            value=0.0
        )
        self.scale_luminancia = tk.Scale(
            controles,
            from_=-5.0,
            to=5.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.luminancia,
            length=210,
            bg=COLOR_CONTROLES,
            fg=COLOR_TEXTO,
            troughcolor=COLOR_AREA_IMAGEN,
            highlightthickness=0,
            bd=0,
            font=("Arial", 9)
        )
        self.scale_luminancia.pack()


        # CROMATICIDAD
        self.label_cromaticidad = tk.Label(
            controles,
            text="Cromaticidad",
            font=("Arial", 10, "bold"),
            bg=COLOR_CONTROLES,
            fg=COLOR_TEXTO
        )
        self.label_cromaticidad.pack(
            pady=(20, 3)
        )
        self.cromaticidad = tk.DoubleVar(
            value=1.0
        )
        self.scale_cromaticidad = tk.Scale(
            controles,
            from_=0.0,
            to=2.0,
            resolution=0.01,
            orient="horizontal",
            variable=self.cromaticidad,
            length=210,
            bg=COLOR_CONTROLES,
            fg=COLOR_TEXTO,
            troughcolor=COLOR_AREA_IMAGEN,
            highlightthickness=0,
            bd=0,
            font=("Arial", 9)
        )
        self.scale_cromaticidad.pack()


        # BOTÓN DE OPERACIÓN
        texto_opcion = "Aplicar ajustes (YIQ → RGB)"
        self.boton_operacion = tk.Button(
            controles,
            text=texto_opcion,
            command=self.aplicar_operacion,
            bg=COLOR_PRIMARIO,
            fg=COLOR_BLANCO,
            activebackground=COLOR_PRIMARIO_HOVER,
            activeforeground=COLOR_BLANCO,
            font=("Arial", 10, "bold"),
            padx=10,
            pady=9,
            relief="flat",
            bd=0,
            cursor="hand2"
        )
        self.boton_operacion.pack(
            fill="x",
            padx=12,
            pady=(25, 15)
        )

        # ====================================================
        # PANEL DERECHO - PROCESADA
        # ====================================================

        panel_procesada = tk.Frame(
            centro,
            bg=COLOR_PANEL,
            bd=1,
            relief="solid"
        )
        panel_procesada.pack(
            side="left",
            fill="both",
            expand=True,
            padx=6
        )
        self.titulo_procesada = tk.Label(
            panel_procesada,
            text=(
                f"IMAGEN PROCESADA  •  "
                f"Espacio de color: "
                "RGB (procesamiento interno en YIQ)"
            ),
            font=("Arial", 13, "bold"),
            bg=COLOR_PANEL,
            fg=COLOR_TEXTO
        )
        self.titulo_procesada.pack(
            pady=(15, 10)
        )

        # ----------------------------------------------------
        # Área de imagen derecha
        # ----------------------------------------------------

        self.label_procesada = tk.Label(
            panel_procesada,
            text="El resultado aparecerá aquí",
            bg=COLOR_AREA_IMAGEN,
            fg=COLOR_TEXTO_SECUNDARIO,
            font=("Arial", 12),
            bd=0
        )
        self.label_procesada.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=(0, 12)
        )
        self.estado = tk.Label(
            self.ventana,
            text="Listo.",
            font=("Arial", 10),
            bg=COLOR_FONDO,
            fg=COLOR_TEXTO_SECUNDARIO,
            anchor="w"
        )
        self.estado.pack(
            side="bottom",
            fill="x",
            padx=20,
            pady=(2, 12)
        )

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
            imagen = (
                np.array(imagen_pil)
                / 255.0
            )
            self.imagen_original = imagen.copy()
            self.imagen_entrada = imagen.copy()
            self.imagen_procesada = None
            self.mostrar_imagen(
                self.label_entrada,
                self.imagen_entrada
            )
            # Limpiar imagen derecha.
            self.label_procesada.config(
                image="",
                text="Aplicá los ajustes para ver el resultado RGB"
            )
            self.label_procesada.foto = None
            self.estado.config(
                text="Imagen cargada correctamente."
            )
        except Exception as error:
            messagebox.showerror(
                "Error",
                f"No se pudo abrir la imagen.\n\n{error}"
            )

    def aplicar_operacion(self):
        if self.imagen_entrada is None:
            messagebox.showwarning(
                "Atención",
                "Primero abrí una imagen."
            )
            return
        try:
            self.imagen_procesada = procesar_imagen_rgb(
                self.imagen_entrada,
                self.luminancia.get(),
                self.cromaticidad.get(),
            )
            self.titulo_entrada.config(
                text="IMAGEN DE ENTRADA  •  Espacio de color: RGB"
            )
            self.titulo_procesada.config(
                text="IMAGEN PROCESADA  •  Espacio de color: RGB (procesamiento interno en YIQ)"
            )
            # Los controles permanecen habilitados porque cada operación
            # comienza desde una imagen RGB, que se transforma internamente.
            self.scale_luminancia.config(state="normal")
            self.scale_cromaticidad.config(state="normal")
            self.mostrar_imagen(
                self.label_procesada,
                self.imagen_procesada
            )
            self.estado.config(
                text="Ajustes aplicados. Resultado convertido nuevamente a RGB."
            )
        except (ValueError, TypeError, FloatingPointError) as error:
            messagebox.showerror(
                "Error de procesamiento",
                f"No se pudo procesar la imagen.\n\n{error}"
            )

    def usar_procesada(self):
        if self.imagen_procesada is None:
            messagebox.showwarning(
                "Atención",
                "Todavía no hay una imagen procesada."
            )
            return
        self.imagen_entrada = self.imagen_procesada.copy()
        self.titulo_entrada.config(
            text="IMAGEN DE ENTRADA  •  Espacio de color: RGB"
        )
        self.titulo_procesada.config(
            text="IMAGEN PROCESADA  •  Espacio de color: RGB (procesamiento interno en YIQ)"
        )
        self.mostrar_imagen(self.label_entrada, self.imagen_entrada)
        self.estado.config(
            text="La imagen procesada ahora es la imagen de entrada (RGB)."
        )

    def restaurar(self):
        if self.imagen_original is None:
            messagebox.showwarning(
                "Atención",
                "Primero abrí una imagen."
            )
            return
        self.imagen_entrada = self.imagen_original.copy()
        self.imagen_procesada = None
        self.titulo_entrada.config(
            text="IMAGEN DE ENTRADA  •  Espacio de color: RGB"
        )
        self.titulo_procesada.config(
            text="IMAGEN PROCESADA  •  Espacio de color: RGB (procesamiento interno en YIQ)"
        )
        self.subtitulo_controles.config(
            text="Ajustes internos en YIQ · salida RGB"
        )
        self.boton_operacion.config(
            text="Aplicar ajustes (YIQ → RGB)"
        )
        self.scale_luminancia.config(state="normal")
        self.scale_cromaticidad.config(state="normal")
        self.mostrar_imagen(self.label_entrada, self.imagen_entrada)
        self.label_procesada.config(
            image="",
            text="El resultado aparecerá aquí"
        )
        self.label_procesada.foto = None
        self.estado.config(text="Imagen original restaurada.")

    def mostrar_imagen(
        self,
        label,
        array_imagen
    ):
        # NumPy 0-1 -> uint8 0-255
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
        # Conservación de la referencia.
        label.foto = foto
        label.config(
            image=foto,
            text=""
        )

    # ========================================================
    # GUARDAR IMAGEN PROCESADA
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
            # 0-1 -> 0-255
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
            imagen_pil.save(
                ruta
            )
            self.estado.config(
                text=f"Imagen guardada correctamente: {ruta}"
            )
        except Exception as error:
            messagebox.showerror(
                "Error",
                f"No se pudo guardar la imagen.\n\n{error}"
            )

if __name__ == "__main__":
    ventana = tk.Tk()
    app = AppPDI(
        ventana
    )
    ventana.mainloop()
