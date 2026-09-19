"""
Interfaz Gráfica - TP2 Grupo 3 (Tkinter + NumPy + Pillow)
Geupo 3
integrantes:
   FLORES, HERNAN
   GUZMAN, ARMIN
   MERCADO, ALEJANDRO
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

# ---------Funciones de Conversión de Espacio de Color----------------


def rgbAyiq(img):
    """Convierte una imagen de formato RGB a YIQ."""
    # Matriz de conversión estándar RGB a YIQ
    mat_yiq = np.array([
        [0.299,      0.587,      0.114],
        [0.595716, -0.274453, -0.321263],
        [0.211456, -0.522591,  0.311135]
    ])
    # Multiplicación matricial mediante dot product por la transpuesta
    im_yiq = np.dot(img, mat_yiq.T)
    return im_yiq


def yiqArgb(img):
    """Convierte una imagen de formato YIQ a RGB asegurando valores entre 0 y 1."""
    # Matriz inversa estándar YIQ a RGB
    mat_rgb = np.array([
        [1,  0.9663,  0.6210],
        [1, -0.2721, -0.6474],
        [1, -1.1070,  1.7046]
    ])
    # Multiplicación matricial mediante dot product por la transpuesta
    im_recuperada = np.dot(img, mat_rgb.T)
    return im_recuperada

# ---------Funciones de Operaciones----------------


def sumaClipeada(img1, img2):
    """Suma dos imágenes y recorta los valores entre 0 y 1."""
    return np.clip(img1 + img2, 0.0, 1.0)


def sumaPromedio(img1, img2):
    """Suma dos imágenes calculando el promedio aritmético entre ambas."""
    return (img1 + img2) / 2.0


def restaClipeada(img1, img2):
    """Resta la imagen 2 de la imagen 1 y recorta los valores entre 0 y 1 para evitar negativos."""
    return np.clip(img1 - img2, 0.0, 1.0)


def restaPromedio(img1, img2):
    """Resta la imagen 2 de la imagen 1 y calcula el promedio o escala de la diferencia."""
    return (img1 - img2) / 2.0


def restaValorAbsoluto(img1, img2):
    """Calcula el valor absoluto de la diferencia entre dos imágenes."""
    return np.abs(img1 - img2)


def producto(img1, img2):
    """Multiplica dos imágenes píxel a píxel y recorta los valores entre 0 y 1."""
    return np.clip(img1 * img2, 0.0, 1.0)


def cociente(img1, img2):
    """Divide la imagen 1 por la imagen 2 de forma segura evitando divisiones por cero."""
    epsilon = 1e-7
    resultado = img1 / (img2 + epsilon)
    return np.clip(resultado, 0.0, 1.0)


def interpolar_imagenes(img1, img2):
    """
    Realiza la interpolación combinando canales YIQ de dos imágenes.
    Convierte ambas a YIQ, separa canales, combina y retorna la imagen en RGB.
    """
    # Convertir imágenes de RGB a YIQ
    im_yiqA = rgbAyiq(img1)
    im_yiqB = rgbAyiq(img2)

    # Separar canales im_yiqA
    yA, iA, qA = im_yiqA[:, :, 0], im_yiqA[:, :, 1], im_yiqA[:, :, 2]
    # Separar canales im_yiqB
    yB, iB, qB = im_yiqB[:, :, 0], im_yiqB[:, :, 1], im_yiqB[:, :, 2]

    # Por lo general, se usa la mayor luminancia
    yC = np.maximum(yA, yB)

    # 5. Aplicar la fórmula exacta de interpolación para I y Q
    # Añadimos un pequeño epsilon para evitar división por cero si YA + YB = 0
    epsilon = 1e-7
    suma_Y = yA + yB + epsilon

    iC = (yA * iA + yB * iB) / suma_Y
    qC = (yA * qA + yB * qB) / suma_Y

    # Unir canales YIQ interpolados
    im_yiq_interp = np.stack([yC, iC, qC], axis=-1)

    # Convertir de YIQ a RGB y retornar
    im_rgb_interp = yiqArgb(im_yiq_interp)
    return np.clip(im_rgb_interp, 0.0, 1.0)


def if_darker(im_1, im_2):
    yiq_1 = rgbAyiq(im_1)
    yiq_2 = rgbAyiq(im_2)

    yA = yiq_1[:, :, 0]
    yB = yiq_2[:, :, 0]

    # compara liminancia
    mask = yA < yB
    # Utiliza np.where para seleccionar los píxeles según la máscara
    result_rgb = np.where(mask[:, :, np.newaxis], im_1, im_2)
    return np.clip(result_rgb, 0, 1)


def if_lighter(im_1, im_2):
    yiq_1 = rgbAyiq(im_1)
    yiq_2 = rgbAyiq(im_2)

    yA = yiq_1[:, :, 0]
    yB = yiq_2[:, :, 0]

    # compara liminancia
    mask = yA > yB
    # Utiliza np.where para seleccionar los píxeles según la máscara
    result_rgb = np.where(mask[:, :, np.newaxis], im_1, im_2)
    return np.clip(result_rgb, 0, 1)

# ---------------------------------------------------


class AppTP2:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Procesamiento Digital de Imágenes - TP2 Grupo N°3")
        self.ventana.geometry("1250x620")
        self.ventana.config(bg="#f0f4f8")

        # Variables para almacenar las imágenes
        self.imagen1_data = None
        self.imagen2_data = None
        self.imagen_procesada_data = None

        # Construir la interfaz gráfica
        self.crear_interfaz()

    def crear_interfaz(self):
        # ----------------------------------------------------
        # TÍTULO SUPERIOR
        # ----------------------------------------------------
        frame_titulo = tk.Frame(self.ventana, bg="#1e3d59", pady=10)
        frame_titulo.pack(side="top", fill="x")

        lbl_titulo = tk.Label(
            frame_titulo,
            text="Título tp2 grupo 3",
            font=("Arial", 16, "bold"),
            fg="white",
            bg="#1e3d59"
        )
        lbl_titulo.pack()

        # ----------------------------------------------------
        # PANEL CENTRAL (CONTENEDOR DE IMÁGENES FIJOS)
        # ----------------------------------------------------
        frame_imagenes = tk.Frame(self.ventana, bg="#f0f4f8")
        frame_imagenes.pack(fill="both", expand=True, padx=15, pady=15)

        # Centrar los contenedores en el panel principal
        frame_imagenes.columnconfigure(0, weight=1)
        frame_imagenes.columnconfigure(1, weight=1)
        frame_imagenes.columnconfigure(2, weight=1)
        frame_imagenes.rowconfigure(0, weight=1)

        # Contenedor intermedio centrado para evitar deformaciones
        sub_frame = tk.Frame(frame_imagenes, bg="#f0f4f8")
        sub_frame.grid(row=0, column=0, columnspan=3)

        # Dimensiones fijas exactas para que los 3 contenedores sean idénticos y estables
        ANCHO_CAJA = 380
        ALTO_CAJA = 380

        estilo_marco = {
            "bg": "white",
            "bd": 2,
            "relief": "groove",
            "width": ANCHO_CAJA,
            "height": ALTO_CAJA
        }

        # --- Imagen 1 ---
        marco_img1 = tk.Frame(sub_frame, **estilo_marco)
        marco_img1.pack(side="left", padx=12)
        marco_img1.pack_propagate(False)  # Bloquea cambios de tamaño

        # Texto fijo inferior
        tk.Label(marco_img1, text="Imagen 1", bg="white", font=(
            "Arial", 10, "bold"), fg="#333333").pack(side="bottom", fill="x", pady=4)

        self.label_img1 = tk.Label(
            marco_img1, text="imagen 1", bg="#e8edf2", fg="#555555", font=("Arial", 11))
        self.label_img1.pack(fill="both", expand=True, padx=4, pady=4)

        # --- Imagen 2 ---
        marco_img2 = tk.Frame(sub_frame, **estilo_marco)
        marco_img2.pack(side="left", padx=12)
        marco_img2.pack_propagate(False)  # Bloquea cambios de tamaño

        # Texto fijo inferior
        tk.Label(marco_img2, text="Imagen 2", bg="white", font=(
            "Arial", 10, "bold"), fg="#333333").pack(side="bottom", fill="x", pady=4)

        self.label_img2 = tk.Label(
            marco_img2, text="imagen 2", bg="#e8edf2", fg="#555555", font=("Arial", 11))
        self.label_img2.pack(fill="both", expand=True, padx=4, pady=4)

        # --- Imagen Procesada ---
        marco_proc = tk.Frame(sub_frame, **estilo_marco)
        marco_proc.pack(side="left", padx=12)
        marco_proc.pack_propagate(False)  # Bloquea cambios de tamaño

        # Texto fijo inferior
        tk.Label(marco_proc, text="Procesada", bg="white", font=(
            "Arial", 10, "bold"), fg="#333333").pack(side="bottom", fill="x", pady=4)

        self.label_procesada = tk.Label(
            marco_proc, text="procesada", bg="#e8edf2", fg="#555555", font=("Arial", 11))
        self.label_procesada.pack(fill="both", expand=True, padx=4, pady=4)

        # ----------------------------------------------------
        # PANEL INFERIOR (CONTROLES Y ACCIONES)
        # ----------------------------------------------------
        frame_controles = tk.Frame(
            self.ventana, bg="#d8e2dc", pady=15, padx=15)
        frame_controles.pack(side="bottom", fill="x")

        # Columna Izquierda: Botones de Carga
        frame_botones_carga = tk.Frame(frame_controles, bg="#d8e2dc")
        frame_botones_carga.pack(side="left", padx=5)

        btn_cargar1 = tk.Button(
            frame_botones_carga,
            text="cargar imagen1",
            command=lambda: self.cargar_imagen(1),
            bg="#438a5e",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5,
            relief="raised"
        )
        btn_cargar1.pack(fill="x", pady=3)

        btn_cargar2 = tk.Button(
            frame_botones_carga,
            text="cargar imagen2",
            command=lambda: self.cargar_imagen(2),
            bg="#438a5e",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5,
            relief="raised"
        )
        btn_cargar2.pack(fill="x", pady=3)

        # Columna Central: Menús desplegables y botón Procesar
        frame_menus = tk.Frame(frame_controles, bg="#d8e2dc")
        frame_menus.pack(side="left", padx=25)

        # Etiqueta para Operaciones
        lbl_operaciones = tk.Label(frame_menus, text="Operaciones", bg="#d8e2dc", font=(
            "Arial", 10, "bold"), fg="#333333")
        lbl_operaciones.grid(row=0, column=0, padx=5, pady=4, sticky="w")

        # Opcion 1
        self.opcion1_var = tk.StringVar(value="Suma(clip)")
        operaciones1 = [
            "Suma(clip)",
            "Suma (prom)",
            "Resta(clip)",
            "Resta(prom)",
            "Resta(abs)",
            "Producto",
            "Cociente",
            "Interpolar",
            "if-darker",
            "if-lighter"
        ]
        menu_opcion1 = tk.OptionMenu(
            frame_menus, self.opcion1_var, *operaciones1)
        menu_opcion1.config(bg="white", width=12, font=("Arial", 10))
        menu_opcion1.grid(row=0, column=1, padx=5, pady=4, sticky="w")

        # Etiqueta para Formato
        lbl_formato = tk.Label(frame_menus, text="Formato", bg="#d8e2dc", font=(
            "Arial", 10, "bold"), fg="#333333")
        lbl_formato.grid(row=1, column=0, padx=5, pady=4, sticky="w")

        # Opcion 2
        self.opcion2_var = tk.StringVar(value="RGB")
        self.opcion2_var.trace_add(
            "write", lambda *args: self.actualizar_visualizacion_formatos())

        operaciones2 = [
            "RGB",
            "YIQ"
        ]
        menu_opcion2 = tk.OptionMenu(
            frame_menus, self.opcion2_var, *operaciones2)
        menu_opcion2.config(bg="white", width=12, font=("Arial", 10))
        menu_opcion2.grid(row=1, column=1, padx=5, pady=4, sticky="w")

        btn_procesar = tk.Button(
            frame_menus,
            text="procesar",
            command=self.procesar_imagenes,
            bg="#1e3d59",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=8
        )
        btn_procesar.grid(row=0, column=2, rowspan=2, padx=15)

        # Columna Derecha: Guardar y Salir
        frame_acciones = tk.Frame(frame_controles, bg="#d8e2dc")
        frame_acciones.pack(side="right", padx=5)

        btn_guardar = tk.Button(
            frame_acciones,
            text="guardar",
            command=self.guardar_imagen,
            bg="#17b978",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=12,
            pady=6
        )
        btn_guardar.pack(side="left", padx=8)

        btn_salir = tk.Button(
            frame_acciones,
            text="salir",
            command=self.ventana.quit,
            bg="#ff6e40",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=12,
            pady=6
        )
        btn_salir.pack(side="left", padx=8)

        # ----------------------------------------------------
        # BARRA DE ESTADO
        # ----------------------------------------------------
        self.estado = tk.Label(
            self.ventana,
            text="Listo. Cargue las imágenes para comenzar.",
            bd=1,
            relief="sunken",
            anchor="w",
            bg="#e2e8f0",
            font=("Arial", 11)
        )
        self.estado.pack(side="bottom", fill="x")

    # ========================================================
    # MÉTODOS DE LA APLICACIÓN
    # ========================================================

    def actualizar_visualizacion_formatos(self):
        """Actualiza la visualización de la Imagen 1 y Imagen 2 según el formato seleccionado (RGB o YIQ)."""
        formato = self.opcion2_var.get().strip()
        if formato == "YIQ":
            if self.imagen1_data is not None:
                img1_yiq = rgbAyiq(self.imagen1_data)
                self.mostrar_imagen_en_label(
                    yiqArgb(img1_yiq), self.label_img1)
            if self.imagen2_data is not None:
                img2_yiq = rgbAyiq(self.imagen2_data)
                self.mostrar_imagen_en_label(
                    yiqArgb(img2_yiq), self.label_img2)
        elif formato == "RGB":
            if self.imagen1_data is not None:
                self.mostrar_imagen_en_label(
                    self.imagen1_data, self.label_img1)
            if self.imagen2_data is not None:
                self.mostrar_imagen_en_label(
                    self.imagen2_data, self.label_img2)

    def cargar_imagen(self, numero):
        ruta = filedialog.askopenfilename(
            title=f"Seleccionar imagen {numero}",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not ruta:
            return
        imagen_pil = Image.open(ruta).convert("RGB")

        # Si se carga la imagen 1 y ya existe la imagen 2, ajustamos la imagen 1
        # para que tenga exactamente las mismas dimensiones que la imagen 2.
        if numero == 1:
            if self.imagen2_data is not None:
                # obtener las dimensiones exactas (alto y ancho) de imagen2 para usarlas como referencia
                h, w, _ = self.imagen2_data.shape
                imagen_pil = imagen_pil.resize(
                    (w, h), Image.Resampling.LANCZOS)

            imagen_array = np.array(imagen_pil) / 255.0
            self.imagen1_data = imagen_array
            self.actualizar_visualizacion_formatos()
            self.estado.config(text="Imagen 1 cargada correctamente.")

        # Si se carga la imagen 2 y ya existe la imagen 1, ajustamos la imagen 2
        # para que tenga exactamente las mismas dimensiones que la imagen 1.
        elif numero == 2:
            if self.imagen1_data is not None:
                h, w, _ = self.imagen1_data.shape
                imagen_pil = imagen_pil.resize(
                    (w, h), Image.Resampling.LANCZOS)

            imagen_array = np.array(imagen_pil) / 255.0
            self.imagen2_data = imagen_array
            self.actualizar_visualizacion_formatos()
            self.estado.config(text="Imagen 2 cargada correctamente.")

    def mostrar_imagen_en_label(self, array_imagen, label_destino):
        """Muestra la imagen rellenando el contenedor fijo de manera uniforme."""
        if array_imagen is None:
            return

        # Convertimos 0-1 nuevamente a 0-255. uint8
        imagen_uint8 = (np.clip(array_imagen, 0, 1) * 255).astype(np.uint8)

        # Pillow -> Tkinter
        imagen_pil = Image.fromarray(imagen_uint8)

        # Forzar el tamaño al espacio completo del contenedor fijo (350x380 menos márgenes)
        ancho_util = 350
        alto_util = 368

        # Redimensionar ocupando toda el área disponible con alta calidad
        imagen_redimensionada = imagen_pil.resize(
            (ancho_util, alto_util), Image.Resampling.LANCZOS)

        foto = ImageTk.PhotoImage(imagen_redimensionada)

        # Guardamos la referencia.
        label_destino.foto = foto

        # Mostramos la imagen.
        label_destino.config(image=foto, text="")

    def procesar_imagenes(self):
        if self.imagen1_data is None:
            messagebox.showwarning(
                "Atención", "Debe cargar la 'imagen 1' para procesar.")
            return
        if self.imagen2_data is None:
            messagebox.showwarning(
                "Atención", "Debe cargar la 'imagen 2' para procesar.")
            return

        operacion = self.opcion1_var.get().strip()
        formato = self.opcion2_var.get().strip()

        # Validar restricción para Interpolar, if-darker y if-lighter en formato YIQ
        if (operacion == "Interpolar" or operacion == "if-darker" or operacion == "if-lighter") and formato != "RGB":
            self.estado.config(
                text=f"No esta permitida la operacion entre: (Operacion : {operacion}, Formato: {formato})")

            messagebox.showwarning(
                "Atención", f"La operación '{operacion}' solo está permitida seleccionando Formato: RGB.")
            # Limpiar la imagen procesada y su contenedor visual
            self.imagen_procesada_data = None
            self.label_procesada.config(image="", text="procesada")

            return
        resultado = self.imagen1_data.copy()

        # Conversión y procesamiento según formato seleccionado
        if formato == "YIQ":
            img1_yiq = rgbAyiq(self.imagen1_data)
            img2_yiq = rgbAyiq(self.imagen2_data)

            # Forzar la actualización visual de la Imagen 1 y Imagen 2 en formato YIQ al presionar procesar
            self.mostrar_imagen_en_label(img1_yiq, self.label_img1)
            self.mostrar_imagen_en_label(img2_yiq, self.label_img2)

            if operacion == "Suma(clip)":
                resultado_yiq = sumaClipeada(img1_yiq, img2_yiq)
            elif operacion == "Suma (prom)":
                resultado_yiq = sumaPromedio(img1_yiq, img2_yiq)
            elif operacion == "Resta(clip)":
                resultado_yiq = restaClipeada(img1_yiq, img2_yiq)
            elif operacion == "Resta(prom)":
                resultado_yiq = restaPromedio(img1_yiq, img2_yiq)
            elif operacion == "Resta(abs)":
                resultado_yiq = restaValorAbsoluto(img1_yiq, img2_yiq)
            elif operacion == "Producto":
                resultado_yiq = producto(img1_yiq, img2_yiq)
            elif operacion == "Cociente":
                resultado_yiq = cociente(img1_yiq, img2_yiq)

            # Asignamos resultado_yiq a resultado
            resultado = resultado_yiq

        elif formato == "RGB":
            # Restaurar la visualización normal de las entradas en RGB
            self.mostrar_imagen_en_label(self.imagen1_data, self.label_img1)
            self.mostrar_imagen_en_label(self.imagen2_data, self.label_img2)

            if operacion == "Suma(clip)":
                resultado = sumaClipeada(self.imagen1_data, self.imagen2_data)
            elif operacion == "Suma (prom)":
                resultado = sumaPromedio(self.imagen1_data, self.imagen2_data)
            elif operacion == "Resta(clip)":
                resultado = restaClipeada(self.imagen1_data, self.imagen2_data)
            elif operacion == "Resta(prom)":
                resultado = restaPromedio(self.imagen1_data, self.imagen2_data)
            elif operacion == "Resta(abs)":
                resultado = restaValorAbsoluto(
                    self.imagen1_data, self.imagen2_data)
            elif operacion == "Producto":
                resultado = producto(self.imagen1_data, self.imagen2_data)
            elif operacion == "Cociente":
                resultado = cociente(self.imagen1_data, self.imagen2_data)
            elif operacion == "Interpolar":
                resultado = interpolar_imagenes(
                    self.imagen1_data, self.imagen2_data)
            elif operacion == "if-darker":
                resultado = if_darker(self.imagen1_data, self.imagen2_data)
            elif operacion == "if-lighter":
                resultado = if_lighter(self.imagen1_data, self.imagen2_data)

        self.imagen_procesada_data = resultado
        self.mostrar_imagen_en_label(
            self.imagen_procesada_data, self.label_procesada)

        self.estado.config(
            text=f"Procesamiento completado (Operacion : {operacion}, Formato: {formato})")

    def guardar_imagen(self):
        if self.imagen_procesada_data is None:
            messagebox.showwarning(
                "Atención", "No hay ninguna imagen procesada para guardar.")
            return

        ruta = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"),
                       ("JPEG Image", "*.jpg"), ("All Files", "*.*")]
        )

        if not ruta:
            return

        imagen_uint8 = (np.clip(self.imagen_procesada_data,
                        0, 1) * 255).astype(np.uint8)
        imagen_pil = Image.fromarray(imagen_uint8)
        imagen_pil.save(ruta)

        messagebox.showinfo("Éxito", f"Imagen guardada con éxito en:\n{ruta}")
        self.estado.config(text=f"Imagen guardada en: {ruta}")


if __name__ == "__main__":
    ventana = tk.Tk()
    app = AppTP2(ventana)
    ventana.mainloop()
