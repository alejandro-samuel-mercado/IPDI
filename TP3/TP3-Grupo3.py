"""
Interfaz Gráfica - TP3 Grupo 3 
Grupo 3
Integrantes:
   FLORES, HERNAN
   GUZMAN, ARMIN
   MERCADO, ALEJANDRO
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")

# --------- Funciones de Conversión de Espacio de Color ----------------


def rgbAyiq(_im):
    """Convierte una imagen de formato RGB a YIQ."""
    MAT_YIQ = np.array([[0.299,      0.595716,  0.211456],
                        [0.587,     -0.274453, -0.522591],
                        [0.114,     -0.321263,  0.311135]])

    _rgb = _im.reshape((-1, 3))
    _yiq = _rgb @ MAT_YIQ
    _yiq = _yiq.reshape(_im.shape)
    return _yiq


def yiqArgb(_im):
    """Convierte una imagen de formato YIQ a RGB asegurando valores válidos."""
    MAT_RGB = np.array([[1,  0.9663,  0.6210],
                        [1, -0.2721, -0.6474],
                        [1, -1.1070,  1.7046]])

    _yiq = _im.reshape((-1, 3))
    _rgb = _yiq @ MAT_RGB
    _rgb = _rgb.reshape(_im.shape)
    return _rgb

# --------- Función Externa para Dibujar Histogramas -------------------


def op_raiz(img):
    # Lógica para Raíz usando YIQ
    yiq = rgbAyiq(img)
    y, i, q = yiq[:, :, 0], yiq[:, :, 1], yiq[:, :, 2]

    y_raiz = np.sqrt(y)

    # Reconstruir la imagen
    yiq_reconstruido = np.stack([y_raiz, i, q], axis=-1)
    raiz = yiqArgb(yiq_reconstruido)

    # Nota, la raiz no necesita ser clipeada por que
    # la raiz de cualquier valor entre 0 y1
    # cae dentro de ese rango
    # √0.1= 0.32 : √0.9 = 0.95
    return raiz, y, y_raiz


def op_cuadrado(img):
    yiq = rgbAyiq(img)
    y, i, q = yiq[:, :, 0], yiq[:, :, 1], yiq[:, :, 2]

    y_cuad = y*y

    # Reconstruir la imagen interpolada
    yiq = np.stack([y_cuad, i, q], axis=-1)

    cuadrado = yiqArgb(yiq)

    # No necesita ser clipeada
    # 0.1^2= 0.01: 0.9^2=0.81
    return cuadrado, y, y_cuad


def funcion_lineal_a_trazos(y, y_min, y_max):
    # Si y es menor que el mínimo, la salida es 0
    if y < y_min:
        return 0.0
    # Si y es mayor que el máximo, la salida es 1
    elif y > y_max:
        return 1.0
    # Caso contrario: realizar interpolación (regla de tres lineal)
    else:
        return (y - y_min) / (y_max - y_min)


def op_lineal_a_trazos(img, y_min, y_max):
    yiq = rgbAyiq(img)
    y, i, q = yiq[:, :, 0], yiq[:, :, 1], yiq[:, :, 2]

    # Vectorizamos la función para que pueda operar elemento a elemento en un array NumPy
    # np.vectorize toma esa matriz y abre un bucle interno invisible. Va tomando cada píxel de uno en uno.
    # ej _scalar_funcion_lineal_a_trozos(pixel_1, y_min=0.1, y_max=0.7), devolviendo una nueva matriz con el mismo tamaño
    funcion_lineal_a_trozos_vectorizada = np.vectorize(funcion_lineal_a_trazos)

    y_lineal_a_trazos = funcion_lineal_a_trozos_vectorizada(y, y_min, y_max)

    # Reconstruir la imagen interpolada
    yiq_lineal_a_trazos = np.stack([y_lineal_a_trazos, i, q], axis=-1)

    linealTrazos = yiqArgb(yiq_lineal_a_trazos)

    # asegurarse de que los valores estén en el rango [0, 1]
    linealTrazos_clipped = np.clip(linealTrazos, 0, 1)

    return linealTrazos_clipped, y, y_lineal_a_trazos


def op_normalizada(img):
    yiq = rgbAyiq(img)
    y, i, q = yiq[:, :, 0], yiq[:, :, 1], yiq[:, :, 2]

    # Encontrar el valor mínimo y máximo de toda la imagen
    min_val = np.min(y)
    max_val = np.max(y)

    # Definir el nuevo rango deseado (en este caso, 0 y 1)
    new_min = 0.0
    new_max = 1.0

    # Aplicar la fórmula de normalización lineal
    y_imagen_normalizada = (y - min_val) * \
        ((new_max - new_min) / (max_val - min_val)) + new_min

    # Reconstruir la imagen interpolada
    yiq_nirmalizadas = np.stack([y_imagen_normalizada, i, q], axis=-1)

    img_normalizada = yiqArgb(yiq_nirmalizadas)

    # asegurarse de que los valores estén en el rango [0, 1]
    normal = np.clip(img_normalizada, 0, 1)

    return normal, y, y_imagen_normalizada


def op_ecualizacion(img):
    yiq = rgbAyiq(img)
    y, i, q = yiq[:, :, 0], yiq[:, :, 1], yiq[:, :, 2]

    # Asumimos que canal_y está en el rango [0, 1]
    # Convertir a 256 niveles discretos (enteros de 0 a 255)
    imagen_uint8 = (y * 255).astype(np.uint8)

    # Calcular el histograma (conteo de cada nivel de 0 a 255)
    hist, bins = np.histogram(imagen_uint8.flatten(), bins=256, range=[0, 256])

    # Calcular la Función de Distribución Acumulada (CDF)
    cdf = hist.cumsum()

    # Encontrar el valor mínimo de la CDF que sea mayor que cero
    cdf_min = cdf[cdf > 0].min()

    # Aplicar la fórmula de normalización de la CDF (L = 256)
    total_pixels = imagen_uint8.size  # M x N

    # Aplicamos round y aseguramos enteros uint8
    cdf_escala = np.round(
        ((cdf - cdf_min) / (total_pixels - cdf_min)) * (256 - 2) + 1).astype(np.uint8)

    # Mapear los nuevos valores usando cdf_escala
    # Esto reemplaza cada valor de pixel viejo por su nuevo valor ecualizado
    imagen_ecualizada_uint8 = cdf_escala[imagen_uint8]

    # Regresar la y ecualizado al formato normalizado [0, 1]
    y_ecualizada = imagen_ecualizada_uint8/255.0

    # Unir el nuevo canal Y ecualizado con los canales originales I y Q
    yiq_reconstruido = np.stack([y_ecualizada, i, q], axis=-1)

    # Convertir de YIQ nuevamente a RGB
    rgb_reconstruido = yiqArgb(yiq_reconstruido)

    # Asegurar que los valores estén en el rango [0, 1]
    img_ecualizada_clipped = np.clip(rgb_reconstruido, 0, 1)

    return img_ecualizada_clipped, y, y_ecualizada


def dibujar_histograma(canal_y, frame_destino, etiqueta_eje_x="Valor de intensidad (Y)", is_placeholder=False):
    """Dibuja un histograma en barras """
    for widget in frame_destino.winfo_children():
        widget.destroy()

    if is_placeholder:
        lbl = tk.Label(frame_destino, text="histograma procesada",
                       bg="#e8edf2", fg="#555555")
        lbl.pack(expand=True, fill="both")
        return

    # Ajustado al tamaño de caja
    fig, ax = plt.subplots(figsize=(4.7, 4.4), dpi=100)

    # Estilo de histograma en barras
    ax.hist(canal_y.ravel(), bins=30, color='dimgray',
            edgecolor='black', alpha=0.8)
    ax.set_xlabel(etiqueta_eje_x, fontsize=7)
    ax.set_ylabel('Frecuencia', fontsize=7)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=7)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame_destino)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ---------------------------------------------------


class AppTP2:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title(
            "Procesamiento Digital de Imágenes - TP3 Grupo N°3 ")
        self.ventana.geometry("910x810")  # ancho y alto
        self.ventana.config(bg="#f0f4f8")

        # Variables para almacenar las imágenes
        self.imagen1_data = None
        self.imagen_procesada_data = None

        # Datos para histogramas
        self.y_original = None
        self.y_procesada = None

        # Referencias a los labels inferiores de los histogramas para actualizar textos
        self.lbl_titulo_hist2 = None

        # Construir la interfaz gráfica
        self.crear_interfaz()

    def crear_interfaz(self):
        # TÍTULO SUPERIOR
        frame_titulo = tk.Frame(self.ventana, bg="#1e3d59", pady=10)
        frame_titulo.pack(side="top", fill="x")

        lbl_titulo = tk.Label(
            frame_titulo,
            text="Procesamiento Digital de Imágenes - TP3 Grupo N°3",
            font=("Arial", 16, "bold"),
            fg="white",
            bg="#1e3d59"
        )
        lbl_titulo.pack()

        # CONTENEDOR PRINCIPAL DE VISUALIZACIÓN
        frame_principal = tk.Frame(self.ventana, bg="#f0f4f8")
        frame_principal.pack(fill="both", expand=True, padx=10, pady=10)

        # Fila 1: Imágenes (Imagen y Procesada)
        frame_imgs = tk.Frame(frame_principal, bg="#f0f4f8")
        frame_imgs.pack(fill="x", pady=5)

        self.ANCHO_CAJA = 420
        self.ALTO_CAJA = 320
        estilo_marco = {"bg": "white", "bd": 2, "relief": "groove",
                        "width": self.ANCHO_CAJA, "height": self.ALTO_CAJA}

        # --- Imagen 1 ---
        marco_img1 = tk.Frame(frame_imgs, **estilo_marco)
        marco_img1.pack(side="left", padx=10)
        marco_img1.pack_propagate(False)
        tk.Label(marco_img1, text="Imagen", bg="white", font=(
            "Arial", 10, "bold"), fg="#333333").pack(side="bottom", fill="x", pady=2)
        self.label_img1 = tk.Label(
            marco_img1, text="Sin imagen", bg="#e8edf2", fg="#555555", font=("Arial", 10))
        self.label_img1.pack(fill="both", expand=True, padx=2, pady=2)

        # --- Imagen Procesada ---
        marco_proc = tk.Frame(frame_imgs, **estilo_marco)
        marco_proc.pack(side="left", padx=10)
        marco_proc.pack_propagate(False)
        tk.Label(marco_proc, text="Procesada", bg="white", font=(
            "Arial", 10, "bold"), fg="#333333").pack(side="bottom", fill="x", pady=2)
        self.label_procesada = tk.Label(
            marco_proc, text="Sin procesar", bg="#e8edf2", fg="#555555", font=("Arial", 10))
        self.label_procesada.pack(fill="both", expand=True, padx=2, pady=2)

        # Fila 2: Secciones de Histogramas
        frame_hists = tk.Frame(frame_principal, bg="#f0f4f8")
        frame_hists.pack(fill="x", pady=5)

        # --- Histograma 1 ---
        marco_hist1 = tk.Frame(frame_hists, **estilo_marco)
        marco_hist1.pack(side="left", padx=10)
        marco_hist1.pack_propagate(False)
        tk.Label(marco_hist1, text="Histograma Y (Original)", bg="white", font=(
            "Arial", 9, "bold"), fg="#333333").pack(side="bottom", fill="x", pady=2)
        self.frame_canvas_hist1 = tk.Frame(marco_hist1, bg="white")
        self.frame_canvas_hist1.pack(fill="both", expand=True, padx=2, pady=2)
        self.lbl_placeholder_hist1 = tk.Label(
            self.frame_canvas_hist1, text="Histograma Imagen", bg="#e8edf2", fg="#555555")
        self.lbl_placeholder_hist1.pack(expand=True, fill="both")

        # --- Histograma 2 ---
        marco_hist2 = tk.Frame(frame_hists, **estilo_marco)
        marco_hist2.pack(side="left", padx=10)
        marco_hist2.pack_propagate(False)

        # Etiqueta inferior dinámica para el histograma procesado
        self.lbl_titulo_hist2 = tk.Label(
            marco_hist2, text="Histograma Y(Procesada)", bg="white", font=("Arial", 9, "bold"), fg="#333333")
        self.lbl_titulo_hist2.pack(side="bottom", fill="x", pady=2)

        self.frame_canvas_hist2 = tk.Frame(marco_hist2, bg="white")
        self.frame_canvas_hist2.pack(fill="both", expand=True, padx=2, pady=2)
        self.lbl_placeholder_hist2 = tk.Label(
            self.frame_canvas_hist2, text="Histograma Procesada", bg="#e8edf2", fg="#555555")
        self.lbl_placeholder_hist2.pack(expand=True, fill="both")

        # PANEL INFERIOR (CONTROLES Y ACCIONES)
        frame_controles = tk.Frame(
            self.ventana, bg="#d8e2dc", pady=10, padx=15)
        frame_controles.pack(side="bottom", fill="x")

        # Botones de Carga
        frame_botones_carga = tk.Frame(frame_controles, bg="#d8e2dc")
        frame_botones_carga.pack(side="left", padx=5)

        btn_cargar1 = tk.Button(
            frame_botones_carga, text="Cargar Imagen", command=lambda: self.cargar_imagen(1),
            bg="#438a5e", fg="white", font=("Arial", 10, "bold"), padx=8, pady=4
        )
        btn_cargar1.pack(fill="x", pady=2)

        # Menús y Procesar
        frame_menus = tk.Frame(frame_controles, bg="#d8e2dc")
        frame_menus.pack(side="left", padx=20)

        lbl_operaciones = tk.Label(frame_menus, text="Operaciones", bg="#d8e2dc", font=(
            "Arial", 10, "bold"), fg="#333333")
        lbl_operaciones.grid(row=0, column=0, padx=5, pady=4, sticky="w")

        self.opcion_var = tk.StringVar(value="Raíz")
        operaciones = [
            "Raíz",
            "Cuadrado",
            "Lineal a Trazos",
            "Normalización",
            "Ecualizacion"
        ]
        menu_opciones = tk.OptionMenu(
            frame_menus, self.opcion_var, *operaciones)
        menu_opciones.config(bg="white", width=14, font=("Arial", 10))
        menu_opciones.grid(row=0, column=1, padx=5, pady=4, sticky="w")

        btn_procesar = tk.Button(
            frame_menus, text="Procesar", command=self.procesar_imagen,
            bg="#1e3d59", fg="white", font=("Arial", 10, "bold"), padx=12, pady=6
        )
        btn_procesar.grid(row=0, column=2, padx=15)

        # Guardar y Salir
        frame_acciones = tk.Frame(frame_controles, bg="#d8e2dc")
        frame_acciones.pack(side="right", padx=5)

        btn_guardar = tk.Button(
            frame_acciones, text="Guardar", command=self.guardar_imagen,
            bg="#17b978", fg="white", font=("Arial", 10, "bold"), padx=10, pady=4
        )
        btn_guardar.pack(side="left", padx=5)

        btn_salir = tk.Button(
            frame_acciones, text="Salir", command=self.ventana.quit,
            bg="#ff6e40", fg="white", font=("Arial", 10, "bold"), padx=10, pady=4
        )
        btn_salir.pack(side="left", padx=5)

        # BARRA DE ESTADO
        self.estado = tk.Label(
            self.ventana, text="Listo. Cargue la imagen para comenzar.", bd=1, relief="sunken", anchor="w", bg="#e2e8f0", font=("Arial", 10)
        )
        self.estado.pack(side="bottom", fill="x")

    # ========================================================
    # MÉTODOS DE LA APLICACIÓN
    # ========================================================

    def cargar_imagen(self, numero):
        ruta = filedialog.askopenfilename(
            title=f"Seleccionar imagen {numero}",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not ruta:
            return
        imagen_pil = Image.open(ruta).convert("RGB")
        imagen_array = np.array(imagen_pil) / 255.0

        if numero == 1:
            self.imagen1_data = imagen_array
            self.mostrar_imagen_en_label(self.imagen1_data, self.label_img1)

        # Limpiar procesada, su histograma y el histograma de procesada al cargar una nueva imagen
        self.imagen_procesada_data = None

        self.label_procesada.config(image="", text="Sin procesar")
        self.lbl_titulo_hist2.config(text="Histograma Y (Procesada)")

        # Limpiar histograma procesado
        for widget in self.frame_canvas_hist2.winfo_children():
            widget.destroy()
        self.lbl_placeholder_hist2 = tk.Label(
            self.frame_canvas_hist2, text="histograma procesada", bg="#e8edf2", fg="#555555")
        self.lbl_placeholder_hist2.pack(expand=True, fill="both")

        # Limpiar también el histograma original
        for widget in self.frame_canvas_hist1.winfo_children():
            widget.destroy()
        self.lbl_placeholder_hist1 = tk.Label(
            self.frame_canvas_hist1, text="histograma img1", bg="#e8edf2", fg="#555555")
        self.lbl_placeholder_hist1.pack(expand=True, fill="both")

        self.estado.config(text="Imagen 1 cargada correctamente.")

    def mostrar_imagen_en_label(self, array_imagen, label_destino):
        if array_imagen is None:
            return
        imagen_uint8 = (np.clip(array_imagen, 0, 1) * 255).astype(np.uint8)
        imagen_pil = Image.fromarray(imagen_uint8)

        imagen_redimensionada = imagen_pil.resize(
            (self.ANCHO_CAJA - 10, self.ALTO_CAJA - 40), Image.Resampling.LANCZOS)
        foto = ImageTk.PhotoImage(imagen_redimensionada)
        label_destino.foto = foto
        label_destino.config(image=foto, text="")

    def procesar_imagen(self):
        if self.imagen1_data is None:
            messagebox.showwarning(
                "Atención", "Debe cargar la 'Imagen' para procesar.")
            return

        operacion = self.opcion_var.get().strip()

        if operacion == "Raíz":
            resultado, y, y_procesada = op_raiz(self.imagen1_data)
        elif operacion == "Cuadrado":
            resultado, y, y_procesada = op_cuadrado(self.imagen1_data)
        elif operacion == "Lineal a Trazos":
            y_min = 0.3
            y_max = 0.6
            resultado, y, y_procesada = op_lineal_a_trazos(
                self.imagen1_data, y_min, y_max)
        elif operacion == "Normalización":
            resultado, y, y_procesada = op_normalizada(self.imagen1_data)
        elif operacion == "Ecualizacion":
            resultado, y, y_procesada = op_ecualizacion(self.imagen1_data)

        self.imagen_procesada_data = resultado
        # Mostrar imagen procesada
        self.mostrar_imagen_en_label(resultado, self.label_procesada)

        # Actualizar el texto inferior con el formato "Histograma Y ( Operación )"
        self.lbl_titulo_hist2.config(text=f"Histograma Y ({operacion})")

        # Llamada a la función externa de histograma modificado
        dibujar_histograma(y_procesada, self.frame_canvas_hist2,
                           "Valor de intensidad (Y')", is_placeholder=False)

        # Llamada a la función externa de histograma
        dibujar_histograma(y, self.frame_canvas_hist1,
                           "Valor de intensidad (Y)", is_placeholder=False)

        self.imagen_procesada_data = resultado

        if operacion == "Lineal a Trazos":
            self.estado.config(
                text=f"El Procesamiento de {operacion} se ha completado con éxito. Con y_min={y_min}, y_max={y_max}")
        else:
            self.estado.config(
                text=f"El Procesamiento de {operacion} se ha completado con éxito.")

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
