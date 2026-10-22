"""
Interfaz Gráfica - TP4 
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


def RGB_2_YIQ(rgb):
    yiq = np.zeros(rgb.shape)
    yiq[:, :, 0] = 0.229*rgb[:, :, 0] + 0.587*rgb[:, :, 1] + 0.114*rgb[:, :, 2]
    yiq[:, :, 1] = 0.595716*rgb[:, :, 0] - \
        0.274453*rgb[:, :, 1] - 0.321263*rgb[:, :, 2]
    yiq[:, :, 2] = 0.211456*rgb[:, :, 0] - \
        0.522591*rgb[:, :, 1] + 0.311135*rgb[:, :, 2]
    return yiq


def YIQ_2_RGB(yiq):
    rgb = np.zeros(yiq.shape)
    rgb[:, :, 0] = yiq[:, :, 0] + 0.9663 * yiq[:, :, 1] + 0.6210 * yiq[:, :, 2]
    rgb[:, :, 1] = yiq[:, :, 0] - 0.2721 * yiq[:, :, 1] - 0.6474 * yiq[:, :, 2]
    rgb[:, :, 2] = yiq[:, :, 0] - 1.1070 * yiq[:, :, 1] + 1.7046 * yiq[:, :, 2]
    return rgb

# --------- Función Externa  -------------------


def _convolution(image, kernel=np.ones((1, 1))):
    convolved = np.zeros((np.array(image.shape)-np.array(kernel.shape)+1))
    for x in range(convolved.shape[0]):
        for y in range(convolved.shape[1]):
            convolved[x, y] = (
                image[x:x+kernel.shape[0], y:y+kernel.shape[1]]*kernel).sum()
    return convolved


def kernel_pasa_bajo_llano(tamano):
    if tamano not in [3, 5, 7]:
        print("Error: El tamaño debe ser 3, 5 o 7.")
        return None
    matriz_unos = np.ones((tamano, tamano), dtype=np.float32)
    kernel = matriz_unos / (tamano * tamano)
    return kernel


def kernel_bartlett(s=3):
    a = (s+1)//2-np.abs(np.arange(s)-s//2)
    k = np.outer(a, a.T)
    return k / k.sum()


def kernel_gaussiano(nro):
    if nro == 3:
        v = np.array([1, 2, 1])
    elif nro == 5:
        v = np.array([1, 4, 6, 4, 1])
    else:
        v = np.array([1, 6, 15, 20, 15, 6, 1])

    # Producto externo para generar la matriz de numeradores
    numeradores = v[:, None] * v

    # Calculamos la suma total de la matriz automáticamente
    suma_total = numeradores.sum()

    # Dividir entre la suma total para normalizar la matriz
    kernel = numeradores / suma_total

    return kernel


def kernel_laplaciano(vecinos):
    if vecinos == 4:
        return np.array([
            [0, -1,  0],
            [-1, 4, -1],
            [0, -1,  0]
        ])
    return np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])


def kernel_direccional(direccion):
    # Kernel base (Oeste)
    oeste = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Kernel base diagonal (Suroeste) para derivar las demás diagonales
    suroeste = np.array([
        [0,  1,  2],
        [-1,  0,  1],
        [-2, -1,  0]
    ])

    match direccion.strip().lower():
        case "oeste":
            return oeste
        case "este":
            return np.fliplr(oeste)          # Inversión horizontal
        case "norte":
            return oeste.T                   # Transpuesta
        case "sur":
            # Inversión vertical de la transpuesta
            return np.flipud(oeste.T)
        case "suroeste":
            return suroeste
        case "sureste":
            return np.fliplr(suroeste)       # Inversión horizontal de Suroeste
        case "noroeste":
            return np.flipud(suroeste)       # Inversión vertical de Suroeste
        case "noreste":
            return np.fliplr(np.flipud(suroeste))  # Combinación de inversiones


def kernel_mejora(valor_mezcla):
    # Definir el kernel identidad de 3x3
    kernel_identidad = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.float32)

    # Obtener el kernel laplaciano de 8 vecinos
    kernel_lapac_8vecinos = kernel_laplaciano(8)

    # Calcular el kernel combinando la identidad y el laplaciano escalado por el valor del slider
    kernel_mejora_ = kernel_identidad + valor_mezcla * kernel_lapac_8vecinos
    return kernel_mejora_


def agregar_ceros_gaus3x3(gaus3x3):
    # Agregar filas arriba y abajo (de ceros)
    fila_ceros = np.array([[0.0, 0.0, 0.0]])
    # Convierte la matriz de 3x3 a 5x3
    paso_1 = np.vstack((fila_ceros, gaus3x3, fila_ceros))

    # Agregar columnas a los lados (de ceros)
    columna_ceros = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]).T
    matriz_final = np.hstack(
        (columna_ceros, paso_1, columna_ceros))  # Convierte a 5x5

    return matriz_final


def kernel_pasabanda():
    return kernel_gaussiano(5) - agregar_ceros_gaus3x3(kernel_gaussiano(3))


def duplicar_bordes(Y, borde):
    # Copia las primeras y últimas 'borde' filas completas
    primeras_filas = Y[:borde, :]
    ultimas_filas = Y[-borde:, :]

    # Concatenar verticalmente
    matriz_filas = np.vstack((primeras_filas, Y, ultimas_filas))

    # Copia las primeras y últimas 'borde' columnas de la matriz resultante
    primeras_cols = matriz_filas[:, :borde]
    ultimas_cols = matriz_filas[:, -borde:]

    # Concatenar horizontalmente
    resultado = np.hstack((primeras_cols, matriz_filas, ultimas_cols))

    return resultado


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
    ax.hist(canal_y.ravel(), bins=40, color='dimgray',
            edgecolor='black', alpha=0.8)
    ax.set_xlabel(etiqueta_eje_x, fontsize=7)
    ax.set_ylabel('Frecuencia', fontsize=7)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=7)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame_destino)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    plt.close(fig)
# ---------------------------------------------------


class AppTP4:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title(
            "Procesamiento Digital de Imágenes - TP4 Grupo N°3 ")
        self.ventana.geometry("1100x910")
        self.ventana.config(bg="#f0f4f8")

        self.operaciones = [
            "3x3",
            "5x5",
            "7x7"
        ]

        self.imagen1_data = None
        self.imagen_procesada_data = None

        # Datos para histogramas
        self.y_original = None
        self.y_procesada = None

        # Referencias a los labels inferiores de los histogramas para actualizar textos
        self.lbl_titulo_hist2 = None

        self.crear_interfaz()

    def crear_interfaz(self):
        # TÍTULO SUPERIOR
        frame_titulo = tk.Frame(self.ventana, bg="#1e3d59", pady=10)
        frame_titulo.pack(side="top", fill="x")

        lbl_titulo = tk.Label(
            frame_titulo,
            text="Procesamiento Digital de Imágenes - TP4 Grupo N°3",
            font=("Arial", 16, "bold"),
            fg="white",
            bg="#1e3d59"
        )
        lbl_titulo.pack()

        # CONTENEDOR PRINCIPAL DE VISUALIZACIÓN
        frame_principal = tk.Frame(self.ventana, bg="#f0f4f8")
        frame_principal.pack(fill="both", expand=True, padx=10, pady=10)

        frame_imgs = tk.Frame(frame_principal, bg="#f0f4f8")
        frame_imgs.pack(fill="x", pady=5)

        self.ANCHO_CAJA = 520
        self.ALTO_CAJA = 350
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

        # Menús filtros y kernels
        frame_menus = tk.Frame(frame_controles, bg="#d8e2dc")
        frame_menus.pack(side="left", padx=20)

        lbl_operaciones = tk.Label(frame_menus, text="Filtro", bg="#d8e2dc", font=(
            "Arial", 10, "bold"), fg="#333333")
        lbl_operaciones.grid(row=0, column=0, padx=5, pady=4, sticky="w")

        self.opcion_filtro = tk.StringVar(value="Pasabajo Llano")
        operaciones = [
            "Pasabajo Llano",
            "Pasabajos Bartlett",
            "Gaussiano",
            "Laplaciano",
            "Sobel",
            "Mejora",
            "Pasabanda"
        ]
        menu_opciones = tk.OptionMenu(
            frame_menus, self.opcion_filtro, *operaciones, command=self.verificar_kernel)
        menu_opciones.config(bg="white", width=14, font=("Arial", 10))
        menu_opciones.grid(row=0, column=1, padx=5, pady=4, sticky="w")

        # Menús kernel / Control Dinámico
        frame_menus_kernel = tk.Frame(frame_controles, bg="#d8e2dc")
        frame_menus_kernel.pack(side="left", padx=20)

        self.lbl_kernel_o_param = tk.Label(frame_menus_kernel, text="Kernel", bg="#d8e2dc", font=(
            "Arial", 10, "bold"), fg="#333333")
        self.lbl_kernel_o_param.grid(
            row=0, column=0, padx=5, pady=4, sticky="w")

        # 1. Menú desplegable tradicional del Kernel
        self.opcion_kelner = tk.StringVar(value="3x3")
        self.menu_kernel = tk.OptionMenu(
            frame_menus_kernel, self.opcion_kelner, *self.operaciones)
        self.menu_kernel.config(bg="white", width=14, font=("Arial", 10))
        self.menu_kernel.grid(row=0, column=1, padx=5, pady=4, sticky="w")

        # 2. Barra deslizante (Scale) entre 0 y 1 para el filtro "Nuevo"
        self.valor_slider = tk.DoubleVar(value=0.0)
        self.slider_nuevo = tk.Scale(
            frame_menus_kernel, from_=0.0, to=1.0, resolution=0.01,
            orient="horizontal", variable=self.valor_slider, bg="white", length=130
        )

        # Boton procesar
        btn_procesar = tk.Button(
            frame_menus_kernel, text="Procesar", command=self.procesar_imagen,
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
            self.ventana, text="Listo. Cargue la imagen para comenzar.", bd=1, relief="sunken", anchor="center", bg="#e2e8f0", font=("Arial", 10, "bold")
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

        self.imagen1_data = imagen_array
        self.mostrar_imagen_en_label(self.imagen1_data, self.label_img1)

        # Limpiar procesada, su histograma y el histograma de procesada al cargar una nueva imagen
        self.imagen_procesada_data = None
        self.label_procesada.config(image="", text="Sin procesar")
        self.estado.config(text="Imagen cargada correctamente.")

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

    def verificar_kernel(self, seleccion):
        menu = self.menu_kernel["menu"]
        menu.delete(0, "end")

        if seleccion == "Pasabanda":
            self.slider_nuevo.grid_remove()
            self.menu_kernel.grid()
            self.menu_kernel.config(state="disabled")
            self.lbl_kernel_o_param.config(text="Kernel")

        elif seleccion == "Mejora":
            self.menu_kernel.grid_remove()
            self.slider_nuevo.grid(row=0, column=1, padx=5, pady=4, sticky="w")
            self.lbl_kernel_o_param.config(text="Valor")

        else:
            self.slider_nuevo.grid_remove()
            self.menu_kernel.grid()
            self.menu_kernel.config(state="normal")
            self.lbl_kernel_o_param.config(text="Kernel")

            if seleccion == "Laplaciano":
                operaciones_kernel = ["4vec", "8vec"]
            elif seleccion == "Sobel":
                operaciones_kernel = [
                    "Norte", "Sur", "Este", "Oeste", "Suroeste", "Sureste", "Noroeste", "Noreste"]
            elif seleccion == "Gaussiano":
                operaciones_kernel = ["5x5", "7x7"]
            else:
                operaciones_kernel = self.operaciones

            self.opcion_kelner.set(operaciones_kernel[0])

            for op in operaciones_kernel:
                menu.add_command(
                    label=op, command=lambda v=op: self.opcion_kelner.set(v))

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

        filtro = self.opcion_filtro.get().strip()

        yiq = RGB_2_YIQ(self.imagen1_data)
        y = yiq[:, :, 0]

        kernel_str = self.opcion_kelner.get().strip()

        # de la Y original le ampliamos dependiendo el kernel ej 3x3 se solo duplica la exterior
        # en el caso 5x5 las dos ultimas y en 7x7 las tres ultimas.
        if filtro in ["Pasabajo Llano", "Pasabajos Bartlett", "Gaussiano"]:
            tam_kernel = int(kernel_str[0])
            borde = (tam_kernel - 1) // 2
        elif filtro == "Pasabanda":
            # Para Pasabanda 5x5 (borde 2)
            borde = 2
        else:
            # Para Laplaciano, Sobel y Mejora usan base 3x3 (borde 1)
            borde = 1

        y_ampliada = duplicar_bordes(y, borde)

        if filtro == "Mejora":
            valor_mezcla = self.valor_slider.get()  # Obtenemos el valor del slider aquí
            resultado = _convolution(y_ampliada, kernel_mejora(valor_mezcla))

        elif filtro == "Sobel":
            resultado = _convolution(
                y_ampliada, kernel_direccional(kernel_str))
        elif filtro == "Pasabanda":
            resultado = _convolution(y_ampliada, kernel_pasabanda())
        else:
            # Para el resto de los filtros (que usan un tamaño numérico)
            tam_kernel = int(kernel_str[0])

            if filtro == "Pasabajo Llano":
                resultado = _convolution(
                    y_ampliada, kernel_pasa_bajo_llano(tam_kernel))
            elif filtro == "Pasabajos Bartlett":
                resultado = _convolution(
                    y_ampliada, kernel_bartlett(tam_kernel))
            elif filtro == "Gaussiano":
                resultado = _convolution(
                    y_ampliada, kernel_gaussiano(tam_kernel))
            elif filtro == "Laplaciano":
                resultado = _convolution(
                    y_ampliada, kernel_laplaciano(tam_kernel))

        self.mostrar_imagen_en_label(y, self.label_img1)
        self.mostrar_imagen_en_label(resultado, self.label_procesada)

        self.imagen_procesada_data = resultado

        # mensajes
        if filtro == "Gaussiano":
            self.estado.config(
                text=f"Se aplicó el filtro {filtro}.\n"f"Tamaño Y {y.shape}; Filtrada {resultado.shape}.")
        elif filtro == "Mejora":
            self.estado.config(
                text=f"Se aplicó el filtro Mejora con valor de mezcla: {self.valor_slider.get():.2f}. \n"f"Tamaño Y {y.shape}; Filtrada {resultado.shape}.")
        elif filtro == "Pasabanda":
            self.estado.config(
                text=f"Se aplicó el filtro {filtro} con un kernel de tamaño 5x5.\n"f"Tamaño Y {y.shape}; Filtrada {resultado.shape}.")
        else:
            self.estado.config(
                text=f"Se aplicó el filtro {filtro} con un kernel de tamaño {kernel_str}.\n"f"Tamaño Y {y.shape}; Filtrada {resultado.shape}.")

        # Actualizar el texto inferior con el formato "Histograma Y ( Operación )"
        self.lbl_titulo_hist2.config(text=f"Histograma Y ({filtro})")

        # Llamada a la función externa de histograma modificado
        dibujar_histograma(resultado, self.frame_canvas_hist2,
                           "Valor de intensidad (Y')", is_placeholder=False)

        # Llamada a la función externa de histograma
        dibujar_histograma(y, self.frame_canvas_hist1,
                           "Valor de intensidad (Y)", is_placeholder=False)

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
    app = AppTP4(ventana)
    ventana.mainloop()
