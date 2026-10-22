import tkinter as tk

from tkinter import filedialog, messagebox, ttk

import os

import numpy as np

from PIL import Image, ImageTk

import my_functions as f

# ============================================================

# COLORES DE LA INTERFAZ

# ============================================================

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

COLOR_EXITO = "#15803D"

COLOR_ERROR = "#B91C1C"

COLOR_PROCESANDO = "#A16207"

# ============================================================

# OPERACIONES DISPONIBLES

# ============================================================

OPERACIONES = {

    "grises": {

        "nombre": "Escala de grises",

        "categoria": "Básicas",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "canal": {

        "nombre": "Canal RGB",

        "categoria": "Básicas",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "rgb_yiq": {

        "nombre": "RGB → YIQ",

        "categoria": "Color",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "yiq_rgb": {

        "nombre": "YIQ → RGB",

        "categoria": "Color",

        "requiere_rgb": False,

        "dos_imagenes": False,

    },

    "raiz": {

        "nombre": "Raíz",

        "categoria": "Luminancia",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "cuadrado": {

        "nombre": "Cuadrado",

        "categoria": "Luminancia",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "lineal": {

        "nombre": "Lineal a trazos",

        "categoria": "Luminancia",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "normalizada": {

        "nombre": "Normalizada",

        "categoria": "Luminancia",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "ecualizacion": {

        "nombre": "Ecualización",

        "categoria": "Luminancia",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "suma_clipeada": {

        "nombre": "Suma clipeada",

        "categoria": "Aritmética",

        "requiere_rgb": True,

        "dos_imagenes": True,

    },

    "suma_promedio": {

        "nombre": "Suma promedio",

        "categoria": "Aritmética",

        "requiere_rgb": True,

        "dos_imagenes": True,

    },

    "resta_clipeada": {

        "nombre": "Resta clipeada",

        "categoria": "Aritmética",

        "requiere_rgb": True,

        "dos_imagenes": True,

    },

    "resta_promedio": {

        "nombre": "Resta promedio",

        "categoria": "Aritmética",

        "requiere_rgb": True,

        "dos_imagenes": True,

    },

    "resta_absoluto": {

        "nombre": "Resta valor absoluto",

        "categoria": "Aritmética",

        "requiere_rgb": True,

        "dos_imagenes": True,

    },

    "producto": {

        "nombre": "Producto",

        "categoria": "Aritmética",

        "requiere_rgb": True,

        "dos_imagenes": True,

    },

    "cociente": {

        "nombre": "Cociente",

        "categoria": "Aritmética",

        "requiere_rgb": True,

        "dos_imagenes": True,

    },

    "interpolar": {

        "nombre": "Interpolar imágenes",

        "categoria": "Comparación",

        "requiere_rgb": True,

        "dos_imagenes": True,

    },

    "darker": {

        "nombre": "If darker",

        "categoria": "Comparación",

        "requiere_rgb": True,

        "dos_imagenes": True,

    },

    "lighter": {

        "nombre": "If lighter",

        "categoria": "Comparación",

        "requiere_rgb": True,

        "dos_imagenes": True,

    },

    "plano": {

        "nombre": "Filtro plano",

        "categoria": "Filtros",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "bartlett": {

        "nombre": "Filtro Bartlett",

        "categoria": "Filtros",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "gaussiano": {

        "nombre": "Filtro gaussiano",

        "categoria": "Filtros",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "pasa_bajos": {

        "nombre": "Filtro pasa bajos",

        "categoria": "Filtros",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "combinar_kernels": {

        "nombre": "Combinar kernels",

        "categoria": "Filtros",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "lap_v4": {

        "nombre": "Laplaciano V4",

        "categoria": "Filtros",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "lap_v8": {

        "nombre": "Laplaciano V8",

        "categoria": "Filtros",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "sobel_este": {

        "nombre": "Sobel Este",

        "categoria": "Filtros",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "sobel_norte": {

        "nombre": "Sobel Norte",

        "categoria": "Filtros",

        "requiere_rgb": True,

        "dos_imagenes": False,

    },

    "sobel_sur": {
        "nombre": "Sobel Sur",
        "categoria": "Filtros",
        "requiere_rgb": True,
        "dos_imagenes": False,
    },

    "sobel_oeste": {
        "nombre": "Sobel Oeste",
        "categoria": "Filtros",
        "requiere_rgb": True,
        "dos_imagenes": False,
    },

    "sobel_noreste": {
        "nombre": "Sobel Noreste",
        "categoria": "Filtros",
        "requiere_rgb": True,
        "dos_imagenes": False,
    },

    "sobel_sureste": {
        "nombre": "Sobel Sureste",
        "categoria": "Filtros",
        "requiere_rgb": True,
        "dos_imagenes": False,
    },

    "sobel_suroeste": {
        "nombre": "Sobel Suroeste",
        "categoria": "Filtros",
        "requiere_rgb": True,
        "dos_imagenes": False,
    },

    "sobel_noroeste": {
        "nombre": "Sobel Noroeste",
        "categoria": "Filtros",
        "requiere_rgb": True,
        "dos_imagenes": False,
    },

    "pasabanda": {
        "nombre": "Pasa banda (DoG)",
        "categoria": "Filtros",
        "requiere_rgb": True,
        "dos_imagenes": False,
    },

    "pasaaltos_02": {
        "nombre": "Pasaaltos 0,2",
        "categoria": "Filtros",
        "requiere_rgb": True,
        "dos_imagenes": False,
    },

    "pasaaltos_04": {
        "nombre": "Pasaaltos 0,4",
        "categoria": "Filtros",
        "requiere_rgb": True,
        "dos_imagenes": False,
    },
}

OPERACIONES_POR_CATEGORIA = {

    "Básicas": ["grises", "canal"],

    "Color": ["rgb_yiq", "yiq_rgb"],

    "Luminancia": ["raiz", "cuadrado", "lineal", "normalizada", "ecualizacion"],

    "Aritmética": [

        "suma_clipeada",

        "suma_promedio",

        "resta_clipeada",

        "resta_promedio",

        "resta_absoluto",

        "producto",

        "cociente",

    ],

    "Comparación": ["interpolar", "darker", "lighter"],

    "Filtros": [

        "pasa_bajos",

        "combinar_kernels",

        "lap_v4",

        "lap_v8",

        "sobel_este",

        "sobel_norte",

        "sobel_sur",

        "sobel_oeste",

        "sobel_noreste",

        "sobel_sureste",

        "sobel_suroeste",

        "sobel_noroeste",

        "pasabanda",

        "pasaaltos_02",

        "pasaaltos_04",

    ],

}

OPERACIONES_DOS_IMAGENES = {

    clave for clave, datos in OPERACIONES.items() if datos["dos_imagenes"]

}

# ============================================================

# APLICACIÓN

# ============================================================

class AppPDI:

    def __init__(self, ventana):

        self.ventana = ventana

        self.ventana.title("PDI - Procesamiento Digital de Imágenes")

        self.ventana.geometry("1250x850")

        self.ventana.minsize(1100, 720)

        self.ventana.configure(bg=COLOR_FONDO)

# Imagen original y entrada actual.

        self.imagen_original = None

        self.imagen_entrada = None

        self.espacio_color = "RGB"

# Último resultado.

        self.imagen_procesada = None

        self.espacio_procesada = None

# Segunda imagen para operaciones binarias.

        self.imagen_segunda = None

        self.ruta_segunda = None

# Operación actualmente seleccionada.

        self.operacion_actual = "grises"

        self.categoria_actual = "Básicas"

        self.botones_categoria = {}

        self.botones_operacion = {}

# Variables de parámetros.

        self.canal_var = tk.StringVar(value="Rojo")

        self.luminancia_var = tk.DoubleVar(value=1.0)

        self.cromaticidad_var = tk.DoubleVar(value=1.0)

        self.y_min_var = tk.StringVar(value="0.0")

        self.y_max_var = tk.StringVar(value="1.0")

        self.tamanio_kernel_var = tk.StringVar(value="3")

        self.tipo_pasabajo_var = tk.StringVar(value="Plano")

        self.kernel1_var = tk.StringVar(value="Identidad")

        self.kernel2_var = tk.StringVar(value="Laplaciano V4")

        self.porcentaje_kernel_var = tk.StringVar(value="0.2")

        self.crear_interfaz()

        self.seleccionar_categoria("Básicas")

        self.seleccionar_operacion("grises")

# ========================================================

# INTERFAZ

# ========================================================

    def crear_interfaz(self):

        self.crear_barra_superior()

        centro = tk.Frame(self.ventana, bg=COLOR_FONDO)

        centro.pack(

            fill="both",

            expand=True,

            padx=18,

            pady=(4, 8)

        )

        self.crear_panel_entrada(centro)

        self.crear_panel_controles(centro)

        self.crear_panel_procesada(centro)

        self.crear_panel_histogramas()

        self.estado = tk.Label(

            self.ventana,

            text="Listo. Seleccioná una operación para comenzar.",

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

    def crear_panel_histogramas(self):

        marco = tk.LabelFrame(
            self.ventana,
            text="HISTOGRAMAS DE LAS IMÁGENES",
            font=("Arial", 10, "bold"),
            bg=COLOR_PANEL,
            fg=COLOR_TEXTO,
            bd=1,
            relief="solid"
        )
        self.marco_histogramas = marco
        marco.pack(fill="x", padx=18, pady=(0, 7))

        for columna in range(3):
            marco.columnconfigure(columna, weight=1)

        self.histograma_entrada = self.crear_histograma(
            marco, "Imagen 1 · Entrada", 0
        )
        self.histograma_segunda = self.crear_histograma(
            marco, "Imagen 2 · Segunda imagen", 1
        )
        self.histograma_resultado = self.crear_histograma(
            marco, "Resultado", 2
        )

    def crear_histograma(self, padre, titulo, columna):

        panel = tk.Frame(padre, bg=COLOR_PANEL)
        panel.grid(
            row=0,
            column=columna,
            sticky="nsew",
            padx=6,
            pady=4
        )

        tk.Label(
            panel,
            text=titulo,
            font=("Arial", 9, "bold"),
            bg=COLOR_PANEL,
            fg=COLOR_TEXTO
        ).pack(pady=(0, 2))

        canvas = tk.Canvas(
            panel,
            height=180,
            bg=COLOR_BLANCO,
            highlightthickness=1,
            highlightbackground=COLOR_BORDE
        )
        canvas.pack(fill="both", expand=True)
        canvas.panel_histograma = panel
        canvas.imagen_histograma = None
        canvas.espacio_color_histograma = "RGB"
        canvas.bind(
            "<Configure>",
            lambda evento, c=canvas: self.dibujar_histograma(c)
        )
        self.dibujar_histograma(canvas)
        return canvas

    def actualizar_histograma(self, canvas, imagen, espacio_color="RGB"):

        canvas.imagen_histograma = imagen
        canvas.espacio_color_histograma = espacio_color
        self.dibujar_histograma(canvas)

    def dibujar_histograma(self, canvas):

        canvas.delete("all")

        ancho = canvas.winfo_width()
        alto = canvas.winfo_height()

        if ancho < 150:
            ancho = 360
        if alto < 70:
            alto = 240

        imagen = canvas.imagen_histograma

        if imagen is None:
            canvas.create_text(
                ancho / 2,
                alto / 2,
                text="Sin imagen",
                fill=COLOR_TEXTO_SECUNDARIO,
                font=("Arial", 10)
            )
            return

        imagen = np.asarray(imagen, dtype=float)

        if imagen.ndim == 2:
            luminancia = imagen
        elif canvas.espacio_color_histograma == "YIQ":
            luminancia = imagen[:, :, 0]
        else:
            luminancia = (
                0.299 * imagen[:, :, 0]
                + 0.587 * imagen[:, :, 1]
                + 0.114 * imagen[:, :, 2]
            )

        luminancia = np.clip(luminancia, 0.0, 1.0)
        # 16 grupos para que las barras sean más anchas y fáciles de distinguir.
        histograma, limites = np.histogram(
            luminancia,
            bins=25,
            range=(0.0, 1.0)
        )

        if histograma.size == 0 or luminancia.size == 0:
            canvas.create_text(
                ancho / 2,
                alto / 2,
                text="Sin datos para mostrar",
                fill=COLOR_TEXTO_SECUNDARIO,
                font=("Arial", 9)
            )
            return

        # El eje vertical representa el porcentaje de píxeles por nivel.
        porcentajes = histograma * 100.0 / luminancia.size
        maximo = porcentajes.max()
        if maximo == 0:
            maximo = 1.0

        margen_izq = 43
        margen_der = 8
        margen_arriba = 8
        margen_abajo = 23

        grafico_ancho = ancho - margen_izq - margen_der
        grafico_alto = alto - margen_arriba - margen_abajo
        base = margen_arriba + grafico_alto

        # Líneas de guía y escala dinámica para que el histograma sea visible.
        for fraccion in (0.0, 0.5, 1.0):
            y = base - fraccion * grafico_alto
            canvas.create_line(
                margen_izq,
                y,
                ancho - margen_der,
                y,
                fill="#E2E8F0",
                width=1
            )
            canvas.create_text(
                margen_izq - 5,
                y,
                text=f"{maximo * fraccion:.1f}%",
                anchor="e",
                fill=COLOR_TEXTO_SECUNDARIO,
                font=("Arial", 7)
            )

        paso = grafico_ancho / len(porcentajes)
        for indice, porcentaje in enumerate(porcentajes):
            x0 = margen_izq + indice * paso + 1.5
            # Separación visible para que no parezca una línea continua.
            x1 = margen_izq + (indice + 1) * paso - 2.0
            altura_barra = (porcentaje / maximo) * grafico_alto
            y0 = base - altura_barra
            canvas.create_rectangle(
                x0,
                y0,
                x1,
                base,
                fill=COLOR_PRIMARIO,
                outline=""
            )

        canvas.create_line(
            margen_izq,
            margen_arriba,
            margen_izq,
            base,
            fill=COLOR_TEXTO_SECUNDARIO
        )
        canvas.create_line(
            margen_izq,
            base,
            ancho - margen_der,
            base,
            fill=COLOR_TEXTO_SECUNDARIO
        )

        # Niveles de intensidad de 0 a 255.
        for nivel in (0, 64, 128, 192, 255):
            x = margen_izq + (nivel / 255.0) * grafico_ancho
            canvas.create_text(
                x,
                alto - 11,
                text=str(nivel),
                fill=COLOR_TEXTO_SECUNDARIO,
                font=("Arial", 7)
            )

    def crear_barra_superior(self):

        barra = tk.Frame(self.ventana, bg=COLOR_FONDO)

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

        boton_abrir.pack(side="left", padx=5)

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

        boton_restaurar.pack(side="left", padx=5)

        boton_usar = tk.Button(

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

        boton_usar.pack(side="left", padx=5)

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

        boton_guardar.pack(side="left", padx=5)

    def crear_panel_entrada(self, padre):

        panel = tk.Frame(
            padre,
            bg=COLOR_PANEL,
            bd=1,
            relief="solid"
        )
        panel.pack(
            side="left",
            fill="both",
            expand=True,
            padx=6
        )

        self.titulo_entrada = tk.Label(
            panel,
            text="IMAGEN DE ENTRADA • Espacio de color: RGB",
            font=("Arial", 13, "bold"),
            bg=COLOR_PANEL,
            fg=COLOR_TEXTO
        )
        self.titulo_entrada.pack(pady=(15, 10))

        # Área con desplazamiento para poder ver las dos imágenes completas.
        marco_scroll = tk.Frame(panel, bg=COLOR_PANEL)
        marco_scroll.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.canvas_entrada = tk.Canvas(
            marco_scroll,
            bg=COLOR_PANEL,
            highlightthickness=0
        )
        barra_scroll = ttk.Scrollbar(
            marco_scroll,
            orient="vertical",
            command=self.canvas_entrada.yview
        )
        self.canvas_entrada.configure(yscrollcommand=barra_scroll.set)
        barra_scroll.pack(side="right", fill="y")
        self.canvas_entrada.pack(side="left", fill="both", expand=True)

        self.contenido_entrada = tk.Frame(
            self.canvas_entrada,
            bg=COLOR_PANEL
        )
        self.ventana_contenido_entrada = self.canvas_entrada.create_window(
            (0, 0),
            window=self.contenido_entrada,
            anchor="nw"
        )

        self.contenido_entrada.bind(
            "<Configure>",
            lambda evento: self.canvas_entrada.configure(
                scrollregion=self.canvas_entrada.bbox("all")
            )
        )
        self.canvas_entrada.bind(
            "<Configure>",
            lambda evento: self.canvas_entrada.itemconfigure(
                self.ventana_contenido_entrada,
                width=evento.width
            )
        )

        # Imagen principal.
        marco_imagen_entrada = tk.LabelFrame(
            self.contenido_entrada,
            text="Imagen 1 · Entrada",
            bg=COLOR_PANEL,
            fg=COLOR_TEXTO,
            font=("Arial", 10, "bold"),
            height=330
        )
        marco_imagen_entrada.pack(
            fill="x",
            expand=True,
            padx=4,
            pady=(0, 8)
        )
        marco_imagen_entrada.pack_propagate(False)

        self.label_entrada = tk.Label(
            marco_imagen_entrada,
            text="Abrí una imagen para comenzar",
            bg=COLOR_AREA_IMAGEN,
            fg=COLOR_TEXTO_SECUNDARIO,
            font=("Arial", 12)
        )
        self.label_entrada.pack(fill="both", expand=True, padx=5, pady=5)

        # La segunda imagen se muestra debajo de la primera cuando la operación la requiere.
        self.seccion_segunda = tk.LabelFrame(
            self.contenido_entrada,
            text="Imagen 2 · Segunda imagen",
            font=("Arial", 10, "bold"),
            bg=COLOR_CONTROLES,
            fg=COLOR_TEXTO,
            bd=1,
            relief="solid"
        )
        self.seccion_segunda.pack(
            fill="x",
            padx=4,
            pady=(0, 12)
        )
        self.crear_seccion_segunda()

    def crear_panel_procesada(self, padre):

        panel = tk.Frame(

            padre,

            bg=COLOR_PANEL,

            bd=1,

            relief="solid"

        )

        panel.pack(

            side="left",

            fill="both",

            expand=True,

            padx=6

        )

        self.titulo_procesada = tk.Label(

            panel,

            text="IMAGEN PROCESADA • Sin resultado",

            font=("Arial", 13, "bold"),

            bg=COLOR_PANEL,

            fg=COLOR_TEXTO

        )

        self.titulo_procesada.pack(pady=(15, 10))

        self.label_procesada = tk.Label(

            panel,

            text="El resultado aparecerá aquí",

            bg=COLOR_AREA_IMAGEN,

            fg=COLOR_TEXTO_SECUNDARIO,

            font=("Arial", 12)

        )

        self.label_procesada.pack(

            fill="both",

            expand=True,

            padx=12,

            pady=(0, 12)

        )

    def crear_panel_controles(self, padre):

        self.controles = tk.Frame(
            padre,
            width=355,
            bg=COLOR_CONTROLES,
            bd=1,
            relief="solid"
        )
        self.controles.pack(
            side="left",
            fill="y",
            padx=10,
            pady=0
        )
        self.controles.pack_propagate(False)

        titulo = tk.Label(
            self.controles,
            text="OPERACIONES",
            font=("Arial", 13, "bold"),
            bg=COLOR_CONTROLES,
            fg=COLOR_TEXTO
        )
        titulo.pack(pady=(12, 2))

        # Temas: quedan fijos en la parte superior.
        self.marco_temas = tk.LabelFrame(
            self.controles,
            text="Temas",
            font=("Arial", 10, "bold"),
            bg=COLOR_CONTROLES,
            fg=COLOR_TEXTO,
            bd=1,
            relief="solid"
        )
        self.marco_temas.pack(
            fill="x",
            padx=12,
            pady=5
        )

        for columna in range(3):
            self.marco_temas.columnconfigure(columna, weight=1)

        for indice, categoria in enumerate(OPERACIONES_POR_CATEGORIA):
            boton = tk.Button(
                self.marco_temas,
                text=categoria,
                command=lambda cat=categoria: self.seleccionar_categoria(cat),
                bg=COLOR_BLANCO,
                fg=COLOR_TEXTO,
                activebackground=COLOR_PRIMARIO_HOVER,
                activeforeground=COLOR_BLANCO,
                font=("Arial", 8, "bold"),
                relief="solid",
                bd=1,
                padx=3,
                pady=6,
                cursor="hand2"
            )
            boton.grid(
                row=indice // 3,
                column=indice % 3,
                sticky="ew",
                padx=3,
                pady=3
            )
            self.botones_categoria[categoria] = boton

        # El mensaje y PROCESAR quedan fijos abajo, fuera del scroll.
        self.panel_fijo = tk.Frame(
            self.controles,
            bg=COLOR_CONTROLES
        )
        self.panel_fijo.pack(
            side="bottom",
            fill="x"
        )

        self.mensaje_procesando = tk.Label(
            self.panel_fijo,
            text="",
            font=("Arial", 10, "bold"),
            bg=COLOR_CONTROLES,
            fg=COLOR_PROCESANDO
        )
        self.mensaje_procesando.pack(
            fill="x",
            padx=12,
            pady=(8, 2)
        )

        self.boton_procesar = tk.Button(
            self.panel_fijo,
            text="PROCESAR",
            command=self.procesar,
            bg=COLOR_PRIMARIO,
            fg=COLOR_BLANCO,
            activebackground=COLOR_PRIMARIO_HOVER,
            activeforeground=COLOR_BLANCO,
            font=("Arial", 11, "bold"),
            padx=10,
            pady=10,
            relief="flat",
            bd=0,
            cursor="hand2"
        )
        self.boton_procesar.pack(
            fill="x",
            padx=12,
            pady=(8, 6)
        )

        # Zona desplazable central: contiene operaciones y parámetros.
        # De esta forma, los controles pueden desplazarse sin ocultar PROCESAR.
        self.marco_scroll_operaciones = tk.Frame(
            self.controles,
            bg=COLOR_CONTROLES
        )
        self.marco_scroll_operaciones.pack(
            side="top",
            fill="both",
            expand=True,
            padx=12,
            pady=5
        )

        self.canvas_operaciones = tk.Canvas(
            self.marco_scroll_operaciones,
            bg=COLOR_CONTROLES,
            highlightthickness=0
        )
        self.barra_operaciones = ttk.Scrollbar(
            self.marco_scroll_operaciones,
            orient="vertical",
            command=self.canvas_operaciones.yview
        )
        self.canvas_operaciones.configure(
            yscrollcommand=self.barra_operaciones.set
        )
        self.barra_operaciones.pack(side="right", fill="y")
        self.canvas_operaciones.pack(side="left", fill="both", expand=True)

        self.contenedor_operaciones = tk.Frame(
            self.canvas_operaciones,
            bg=COLOR_CONTROLES
        )
        self.ventana_operaciones = self.canvas_operaciones.create_window(
            (0, 0),
            window=self.contenedor_operaciones,
            anchor="nw"
        )
        self.contenedor_operaciones.bind(
            "<Configure>",
            lambda evento: self.canvas_operaciones.configure(
                scrollregion=self.canvas_operaciones.bbox("all")
            )
        )
        self.canvas_operaciones.bind(
            "<Configure>",
            lambda evento: self.canvas_operaciones.itemconfigure(
                self.ventana_operaciones,
                width=evento.width
            )
        )

        self.marco_operaciones = tk.LabelFrame(
            self.contenedor_operaciones,
            text="Operaciones",
            font=("Arial", 10, "bold"),
            bg=COLOR_CONTROLES,
            fg=COLOR_TEXTO,
            bd=1,
            relief="solid"
        )
        self.marco_operaciones.pack(
            fill="x",
            padx=2,
            pady=2
        )

        self.parametros = tk.LabelFrame(
            self.contenedor_operaciones,
            text="Parámetros",
            font=("Arial", 10, "bold"),
            bg=COLOR_CONTROLES,
            fg=COLOR_TEXTO,
            bd=1,
            relief="solid"
        )
        self.parametros.pack(
            fill="x",
            padx=2,
            pady=5
        )

    def seleccionar_categoria(self, categoria):

        self.categoria_actual = categoria

        for nombre, boton in self.botones_categoria.items():

            if nombre == categoria:

                boton.config(

                    bg=COLOR_PRIMARIO,

                    fg=COLOR_BLANCO,

                    activebackground=COLOR_PRIMARIO_HOVER,

                    activeforeground=COLOR_BLANCO

                )

            else:

                boton.config(

                    bg=COLOR_BLANCO,

                    fg=COLOR_TEXTO,

                    activebackground=COLOR_AREA_IMAGEN,

                    activeforeground=COLOR_TEXTO

                )

        for widget in self.marco_operaciones.winfo_children():

            widget.destroy()

        self.botones_operacion = {}

        self.marco_operaciones.columnconfigure(0, weight=1)

        self.marco_operaciones.columnconfigure(1, weight=1)

        claves = OPERACIONES_POR_CATEGORIA[categoria]

        for indice, clave in enumerate(claves):

            boton = tk.Button(

                self.marco_operaciones,

                text=OPERACIONES[clave]["nombre"],

                command=lambda op=clave: self.seleccionar_operacion(op),

                bg=COLOR_BLANCO,

                fg=COLOR_TEXTO,

                activebackground=COLOR_PRIMARIO_HOVER,

                activeforeground=COLOR_BLANCO,

                font=("Arial", 9, "bold"),

                relief="solid",

                bd=1,

                padx=5,

                pady=7,

                cursor="hand2"

            )

            boton.grid(

                row=indice // 2,

                column=indice % 2,

                sticky="ew",

                padx=4,

                pady=4

            )

            self.botones_operacion[clave] = boton

        self.canvas_operaciones.configure(

            scrollregion=self.canvas_operaciones.bbox("all")

        )

        self.canvas_operaciones.yview_moveto(0)

# Cada tema comienza mostrando su primera operación.

        self.seleccionar_operacion(claves[0])

    def crear_seccion_segunda(self):

        contenido = tk.Frame(
            self.seccion_segunda,
            bg=COLOR_CONTROLES
        )
        contenido.pack(fill="x", padx=8, pady=6)

        marco_imagen_segunda = tk.Frame(
            contenido,
            bg=COLOR_AREA_IMAGEN,
            height=320
        )
        marco_imagen_segunda.pack(fill="x", pady=(0, 6))
        marco_imagen_segunda.pack_propagate(False)

        self.label_segunda = tk.Label(
            marco_imagen_segunda,
            text="Agregá una segunda imagen",
            bg=COLOR_AREA_IMAGEN,
            fg=COLOR_TEXTO_SECUNDARIO,
            font=("Arial", 10)
        )
        self.label_segunda.pack(fill="both", expand=True)

        self.boton_segunda = tk.Button(
            contenido,
            text="Agregar segunda imagen",
            command=self.agregar_segunda_imagen,
            bg=COLOR_BLANCO,
            fg=COLOR_TEXTO,
            activebackground=COLOR_AREA_IMAGEN,
            font=("Arial", 9, "bold"),
            relief="solid",
            bd=1,
            cursor="hand2"
        )
        self.boton_segunda.pack(fill="x", pady=2)

        self.boton_quitar_segunda = tk.Button(
            contenido,
            text="Quitar segunda imagen",
            command=self.quitar_segunda_imagen,
            bg=COLOR_BLANCO,
            fg=COLOR_ERROR,
            activebackground=COLOR_AREA_IMAGEN,
            font=("Arial", 9, "bold"),
            relief="solid",
            bd=1,
            cursor="hand2"
        )
        self.boton_quitar_segunda.pack(fill="x", pady=2)

    def seleccionar_operacion(self, clave):

        self.operacion_actual = clave

        datos = OPERACIONES[clave]

        for boton_clave, boton in self.botones_operacion.items():

            if boton_clave == clave:

                boton.config(

                    bg=COLOR_PRIMARIO,

                    fg=COLOR_BLANCO,

                    activebackground=COLOR_PRIMARIO_HOVER,

                    activeforeground=COLOR_BLANCO

                )

            else:

                boton.config(

                    bg=COLOR_BLANCO,

                    fg=COLOR_TEXTO,

                    activebackground=COLOR_AREA_IMAGEN,

                    activeforeground=COLOR_TEXTO

                )

        self.mostrar_parametros()

        self.actualizar_seccion_segunda()

    def mostrar_parametros(self):

        for widget in self.parametros.winfo_children():
            widget.destroy()

        op = self.operacion_actual
        operaciones_con_parametros = {
            "canal", "rgb_yiq", "lineal",
            "plano", "bartlett", "gaussiano",
            "pasa_bajos", "combinar_kernels"
        }

        if op not in operaciones_con_parametros:
            self.parametros.pack_forget()
            return

        # Si la sección estaba oculta, volver a colocarla antes del estado y del botón.
        if not self.parametros.winfo_manager():
            self.parametros.pack(
                fill="x",
                padx=2,
                pady=5
            )

        if op == "canal":
            tk.Label(
                self.parametros,
                text="Canal a conservar:",
                bg=COLOR_CONTROLES,
                fg=COLOR_TEXTO,
                font=("Arial", 9, "bold")
            ).pack(anchor="w", padx=8, pady=(7, 4))

            ttk.Combobox(
                self.parametros,
                textvariable=self.canal_var,
                values=["Rojo", "Verde", "Azul"],
                state="readonly"
            ).pack(fill="x", padx=8, pady=(0, 8))
            return

        if op == "rgb_yiq":
            tk.Label(
                self.parametros,
                text="Luminancia",
                bg=COLOR_CONTROLES,
                fg=COLOR_TEXTO,
                font=("Arial", 9, "bold")
            ).pack(anchor="w", padx=8, pady=(7, 2))

            tk.Scale(
                self.parametros,
                from_=0.0,
                to=2.0,
                resolution=0.01,
                orient="horizontal",
                variable=self.luminancia_var,
                bg=COLOR_CONTROLES,
                fg=COLOR_TEXTO,
                troughcolor=COLOR_AREA_IMAGEN,
                highlightthickness=0,
                bd=0,
                length=265
            ).pack(fill="x", padx=8)

            tk.Label(
                self.parametros,
                text="Cromaticidad",
                bg=COLOR_CONTROLES,
                fg=COLOR_TEXTO,
                font=("Arial", 9, "bold")
            ).pack(anchor="w", padx=8, pady=(8, 2))

            tk.Scale(
                self.parametros,
                from_=-1.0,
                to=1.0,
                resolution=0.01,
                orient="horizontal",
                variable=self.cromaticidad_var,
                bg=COLOR_CONTROLES,
                fg=COLOR_TEXTO,
                troughcolor=COLOR_AREA_IMAGEN,
                highlightthickness=0,
                bd=0,
                length=265
            ).pack(fill="x", padx=8, pady=(0, 7))
            return

        if op == "lineal":
            tk.Label(
                self.parametros,
                text="Y mínimo",
                bg=COLOR_CONTROLES,
                fg=COLOR_TEXTO,
                font=("Arial", 9, "bold")
            ).pack(anchor="w", padx=8, pady=(7, 2))

            tk.Entry(
                self.parametros,
                textvariable=self.y_min_var,
                font=("Arial", 9)
            ).pack(fill="x", padx=8)

            tk.Label(
                self.parametros,
                text="Y máximo",
                bg=COLOR_CONTROLES,
                fg=COLOR_TEXTO,
                font=("Arial", 9, "bold")
            ).pack(anchor="w", padx=8, pady=(7, 2))

            tk.Entry(
                self.parametros,
                textvariable=self.y_max_var,
                font=("Arial", 9)
            ).pack(fill="x", padx=8, pady=(0, 8))
            return

        if op == "pasa_bajos":
            tk.Label(
                self.parametros,
                text="Tipo de filtro pasa bajos:",
                bg=COLOR_CONTROLES,
                fg=COLOR_TEXTO,
                font=("Arial", 9, "bold")
            ).pack(anchor="w", padx=8, pady=(7, 4))

            combo_tipo = ttk.Combobox(
                self.parametros,
                textvariable=self.tipo_pasabajo_var,
                values=["Plano", "Bartlett", "Gaussiano"],
                state="readonly"
            )
            combo_tipo.pack(fill="x", padx=8, pady=(0, 7))
            combo_tipo.bind(
                "<<ComboboxSelected>>",
                self.actualizar_tamanios_pasabajo
            )

            tk.Label(
                self.parametros,
                text="Tamaño del kernel:",
                bg=COLOR_CONTROLES,
                fg=COLOR_TEXTO,
                font=("Arial", 9, "bold")
            ).pack(anchor="w", padx=8, pady=(0, 4))

            self.combo_tamanio_pasabajo = ttk.Combobox(
                self.parametros,
                textvariable=self.tamanio_kernel_var,
                state="readonly"
            )
            self.combo_tamanio_pasabajo.pack(
                fill="x", padx=8, pady=(0, 8)
            )
            self.actualizar_tamanios_pasabajo()
            return

        if op == "combinar_kernels":
            opciones_kernels = [
                "Identidad", "Plano", "Bartlett", "Gaussiano",
                "Laplaciano V4", "Laplaciano V8",
                "Sobel Este", "Sobel Norte", "Sobel Sur", "Sobel Oeste",
                "Sobel Noreste", "Sobel Sureste", "Sobel Suroeste", "Sobel Noroeste"
            ]

            tk.Label(
                self.parametros,
                text="Primer kernel:",
                bg=COLOR_CONTROLES,
                fg=COLOR_TEXTO,
                font=("Arial", 9, "bold")
            ).pack(anchor="w", padx=8, pady=(7, 4))
            ttk.Combobox(
                self.parametros,
                textvariable=self.kernel1_var,
                values=opciones_kernels,
                state="readonly"
            ).pack(fill="x", padx=8, pady=(0, 6))

            tk.Label(
                self.parametros,
                text="Segundo kernel:",
                bg=COLOR_CONTROLES,
                fg=COLOR_TEXTO,
                font=("Arial", 9, "bold")
            ).pack(anchor="w", padx=8, pady=(2, 4))
            ttk.Combobox(
                self.parametros,
                textvariable=self.kernel2_var,
                values=opciones_kernels,
                state="readonly"
            ).pack(fill="x", padx=8, pady=(0, 6))

            tk.Label(
                self.parametros,
                text="Tamaño común (3, 5 o 7):",
                bg=COLOR_CONTROLES,
                fg=COLOR_TEXTO,
                font=("Arial", 9, "bold")
            ).pack(anchor="w", padx=8, pady=(2, 4))
            ttk.Combobox(
                self.parametros,
                textvariable=self.tamanio_kernel_var,
                values=["3", "5", "7"],
                state="readonly"
            ).pack(fill="x", padx=8, pady=(0, 6))

            tk.Label(
                self.parametros,
                text="Peso del segundo kernel (ej. 0.2):",
                bg=COLOR_CONTROLES,
                fg=COLOR_TEXTO,
                font=("Arial", 9, "bold")
            ).pack(anchor="w", padx=8, pady=(2, 4))
            tk.Entry(
                self.parametros,
                textvariable=self.porcentaje_kernel_var,
                font=("Arial", 9)
            ).pack(fill="x", padx=8, pady=(0, 8))
            return

        if op in {"plano", "bartlett", "gaussiano"}:
            tk.Label(
                self.parametros,
                text="Tamaño del kernel:",
                bg=COLOR_CONTROLES,
                fg=COLOR_TEXTO,
                font=("Arial", 9, "bold")
            ).pack(anchor="w", padx=8, pady=(7, 4))

            valores = ["5", "7"] if op == "gaussiano" else ["3", "5", "7"]
            ttk.Combobox(
                self.parametros,
                textvariable=self.tamanio_kernel_var,
                values=valores,
                state="readonly"
            ).pack(fill="x", padx=8, pady=(0, 8))

    def actualizar_tamanios_pasabajo(self, evento=None):

        if self.tipo_pasabajo_var.get() == "Gaussiano":
            valores = ["5", "7"]
        else:
            valores = ["3", "5", "7"]

        self.combo_tamanio_pasabajo.configure(values=valores)

        if self.tamanio_kernel_var.get() not in valores:
            self.tamanio_kernel_var.set(valores[0])

    def obtener_kernel_combinacion(self, nombre, tamanio):

        if nombre == "Identidad":
            kernel = np.zeros((tamanio, tamanio))
            kernel[tamanio // 2, tamanio // 2] = 1
            return kernel

        if nombre == "Plano":
            return f.kernel_plano(tamanio)

        if nombre == "Bartlett":
            return f.kernel_bartlett(tamanio)

        if nombre == "Gaussiano":
            return f.kernel_gaussiano(tamanio)

        kernels = {
            "Laplaciano V4": f.kernel_laplaciano_v4,
            "Laplaciano V8": f.kernel_laplaciano_v8,
            "Sobel Este": f.kernel_sobel_este,
            "Sobel Norte": f.kernel_sobel_norte,
            "Sobel Sur": f.kernel_sobel_sur,
            "Sobel Oeste": f.kernel_sobel_oeste,
            "Sobel Noreste": f.kernel_sobel_noreste,
            "Sobel Sureste": f.kernel_sobel_sureste,
            "Sobel Suroeste": f.kernel_sobel_suroeste,
            "Sobel Noroeste": f.kernel_sobel_noroeste
        }

        kernel = kernels[nombre]()

        if kernel.shape != (tamanio, tamanio):
            diferencia = tamanio - kernel.shape[0]
            kernel = np.pad(
                kernel,
                (
                    (diferencia // 2, diferencia - diferencia // 2),
                    (diferencia // 2, diferencia - diferencia // 2)
                ),
                mode="constant"
            )

        return kernel

    def actualizar_seccion_segunda(self):

        necesita = self.operacion_actual in OPERACIONES_DOS_IMAGENES

        # Mostrar el histograma de la segunda imagen solo en operaciones binarias.
        panel_segunda = self.histograma_segunda.panel_histograma
        panel_resultado = self.histograma_resultado.panel_histograma

        if necesita:
            panel_segunda.grid(
                row=0,
                column=1,
                sticky="nsew",
                padx=6,
                pady=4
            )
            panel_resultado.grid(
                row=0,
                column=2,
                sticky="nsew",
                padx=6,
                pady=4
            )
            for columna in range(3):
                self.marco_histogramas.columnconfigure(columna, weight=1)
        else:
            panel_segunda.grid_remove()
            panel_segunda.grid_forget()
            panel_resultado.grid(
                row=0,
                column=1,
                sticky="nsew",
                padx=6,
                pady=4
            )
            self.marco_histogramas.columnconfigure(0, weight=1)
            self.marco_histogramas.columnconfigure(1, weight=1)
            self.marco_histogramas.columnconfigure(2, weight=0)

        if necesita:
            if not self.seccion_segunda.winfo_manager():
                self.seccion_segunda.pack(
                    fill="x",
                    padx=4,
                    pady=(0, 12)
                )

            self.boton_segunda.config(state="normal")
            self.boton_quitar_segunda.config(
                state="normal" if self.imagen_segunda is not None else "disabled"
            )
        else:
            # No mostrar ni pedir la segunda imagen para operaciones que no la usan.
            self.seccion_segunda.pack_forget()
            self.boton_segunda.config(state="disabled")
            self.boton_quitar_segunda.config(state="disabled")

        self.actualizar_vista_segunda()

    def agregar_segunda_imagen(self):

        if self.imagen_entrada is None:

            self.estado.config(

                text="Primero abrí la imagen principal.",

                fg=COLOR_ERROR

            )

            messagebox.showwarning(

                "Atención",

                "Primero abrí la imagen principal."

            )

            return

        ruta = filedialog.askopenfilename(

            title="Seleccionar segunda imagen",

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

# ------------------------------------------------

# Tamaño de la imagen principal

# ------------------------------------------------

            alto_principal = self.imagen_entrada.shape[0]

            ancho_principal = self.imagen_entrada.shape[1]

# ------------------------------------------------

# Tamaño original de la segunda imagen

# ------------------------------------------------

            ancho_original, alto_original = imagen_pil.size

# ------------------------------------------------

# Si tienen distinto tamaño, ajustamos la segunda

# imagen al tamaño de la primera.

# ------------------------------------------------

            ajustada = False

            if (

                ancho_original != ancho_principal

                or alto_original != alto_principal

            ):

                imagen_pil = imagen_pil.resize(

                    (

                        ancho_principal,

                        alto_principal

                    )

                )

                ajustada = True

# ------------------------------------------------

# Pillow -> NumPy

# ------------------------------------------------

            imagen = (

                np.array(

                    imagen_pil,

                    dtype=float

                ) / 255.0

            )

            self.imagen_segunda = imagen

            self.ruta_segunda = ruta

            self.actualizar_vista_segunda()

# ------------------------------------------------

# Mensaje

# ------------------------------------------------

            if ajustada:

                self.estado.config(

                    text=(

                        "Segunda imagen cargada y "

                        "ajustada al tamaño de la imagen principal."

                    ),

                    fg=COLOR_EXITO

                )

            else:

                self.estado.config(

                    text=(

                        "Segunda imagen cargada correctamente."

                    ),

                    fg=COLOR_EXITO

                )

        except Exception as error:

            self.estado.config(

                text="Error al cargar la segunda imagen.",

                fg=COLOR_ERROR

            )

            messagebox.showerror(

                "Error",

                (

                    "No se pudo cargar la segunda imagen."

                    f"\n\n{error}"

                )

            )

    def actualizar_vista_segunda(self):

        if self.imagen_segunda is None:

            self.label_segunda.config(

                image="",

                text="No hay segunda imagen cargada"

            )

            self.label_segunda.foto = None

            if hasattr(self, "histograma_segunda"):
                self.actualizar_histograma(
                    self.histograma_segunda,
                    None
                )

            self.boton_segunda.config(text="Agregar segunda imagen")

            self.boton_quitar_segunda.config(state="disabled")

            return

        self.mostrar_imagen(

            self.label_segunda,

            self.imagen_segunda,

            espacio_color="RGB",

            tamanio=(320, 300)

        )

        nombre = os.path.basename(self.ruta_segunda) if self.ruta_segunda else "Imagen cargada"

        self.boton_segunda.config(text=f"Cambiar segunda imagen\n{nombre}")

        self.boton_quitar_segunda.config(state="normal")

    def quitar_segunda_imagen(self):

        self.imagen_segunda = None

        self.ruta_segunda = None

        self.actualizar_vista_segunda()

        self.estado.config(

            text="Segunda imagen eliminada.",

            fg=COLOR_TEXTO_SECUNDARIO

        )

# ========================================================

# IMAGEN PRINCIPAL

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

            imagen_pil = Image.open(ruta).convert("RGB")

            imagen = np.array(imagen_pil, dtype=float) / 255.0

            self.imagen_original = imagen.copy()

            self.imagen_entrada = imagen.copy()

            self.espacio_color = "RGB"

            self.imagen_procesada = None

            self.espacio_procesada = None

# Una nueva imagen principal también reinicia la segunda imagen.

            self.imagen_segunda = None

            self.ruta_segunda = None

            self.actualizar_vista_segunda()

            self.mostrar_imagen(

                self.label_entrada,

                self.imagen_entrada,

                espacio_color="RGB",

                tamanio=(320, 300)

            )

            self.limpiar_resultado()

            self.actualizar_titulos()

            self.estado.config(

                text=f"Imagen cargada correctamente: {os.path.basename(ruta)}",

                fg=COLOR_EXITO

            )

        except Exception as error:

            self.estado.config(

                text="Error al abrir la imagen.",

                fg=COLOR_ERROR

            )

            messagebox.showerror(

                "Error",

                f"No se pudo abrir la imagen.\n\n{error}"

            )

    def restaurar(self):

        if self.imagen_original is None:

            self.estado.config(text="Primero abrí una imagen.", fg=COLOR_ERROR)

            messagebox.showwarning(

                "Atención",

                "Primero abrí una imagen."

            )

            return

        self.imagen_entrada = self.imagen_original.copy()

        self.espacio_color = "RGB"

        self.imagen_procesada = None

        self.espacio_procesada = None

        self.mostrar_imagen(

            self.label_entrada,

            self.imagen_entrada,

            espacio_color="RGB",

            tamanio=(320, 300)

        )

        self.limpiar_resultado()

        self.actualizar_titulos()

        self.estado.config(

            text="Imagen original restaurada.",

            fg=COLOR_EXITO

        )

    def usar_procesada(self):

        if self.imagen_procesada is None:

            self.estado.config(

                text="Todavía no hay una imagen procesada.",

                fg=COLOR_ERROR

            )

            messagebox.showwarning(

                "Atención",

                "Todavía no hay una imagen procesada."

            )

            return

        self.imagen_entrada = self.imagen_procesada.copy()

        self.espacio_color = self.espacio_procesada

# Se reinicia el resultado anterior, pero se conserva la imagen secundaria.

        self.imagen_procesada = None

        self.espacio_procesada = None

        self.mostrar_imagen(

            self.label_entrada,

            self.imagen_entrada,

            espacio_color=self.espacio_color,

            tamanio=(320, 300)

        )

        self.limpiar_resultado()

        self.actualizar_titulos()

        self.estado.config(

            text="La imagen procesada ahora es la imagen de entrada.",

            fg=COLOR_EXITO

        )

# ========================================================

# PROCESAMIENTO

# ========================================================

    def procesar(self):

        if self.imagen_entrada is None:

            self.estado.config(

                text="Primero abrí una imagen.",

                fg=COLOR_ERROR

            )

            messagebox.showwarning(

                "Atención",

                "Primero abrí una imagen."

            )

            return

        operacion = self.operacion_actual

        datos = OPERACIONES[operacion]

        if datos["requiere_rgb"] and self.espacio_color != "RGB":

            self.mostrar_error(

                "La operación seleccionada requiere que la imagen de entrada esté en RGB.\n\n"

                "Convertí primero de YIQ a RGB."

            )

            return

        if datos["dos_imagenes"]:

            if self.imagen_segunda is None:

                self.mostrar_error(

                    "Esta operación necesita una segunda imagen.\n\n"

                    "Usá el botón 'Agregar segunda imagen'."

                )

                return

            if self.imagen_entrada.shape != self.imagen_segunda.shape:

                self.mostrar_error(

                    "La imagen de entrada y la segunda imagen deben tener el mismo tamaño."

                )

                return

        self.boton_procesar.config(

      state="disabled"

      )

        self.mensaje_procesando.config(

      text=f"⏳ Procesando: {datos['nombre']}...",

      fg=COLOR_PROCESANDO

      )

        self.estado.config(

      text="La operación se está procesando...",

      fg=COLOR_PROCESANDO

      )

        self.ventana.update_idletasks()

        try:

            resultado, espacio_resultado = self.ejecutar_operacion(operacion)

            self.imagen_procesada = resultado

            self.espacio_procesada = espacio_resultado

            self.mostrar_imagen(

                self.label_procesada,

                resultado,

                espacio_color=espacio_resultado

            )

            # Actualizar siempre el histograma del resultado procesado.
            self.actualizar_histograma(
                self.histograma_resultado,
                resultado,
                espacio_resultado
            )

            self.actualizar_titulos()

            self.mensaje_procesando.config(

                  text="✓ Procesamiento terminado correctamente.",

                  fg=COLOR_EXITO

                  )

            self.estado.config(

                  text=f"Procesamiento terminado correctamente: {datos['nombre']}",

                  fg=COLOR_EXITO

                  )

            self.ventana.update_idletasks()

        except Exception as error:

            self.imagen_procesada = None

            self.espacio_procesada = None

            self.limpiar_resultado()

            self.actualizar_titulos()

            self.mensaje_procesando.config(

                  text="✗ Error durante el procesamiento.",

                  fg=COLOR_ERROR

                  )

            self.estado.config(

                text=f"Error durante el procesamiento: {error}",

                fg=COLOR_ERROR

            )

            self.ventana.update_idletasks()

            messagebox.showerror(

                "Error de procesamiento",

                f"La operación '{datos['nombre']}' produjo un error.\n\n{error}"

            )

        finally:

            self.boton_procesar.config(state="normal")

    def ejecutar_operacion(self, operacion):

        imagen = self.imagen_entrada

        if operacion == "grises":

            return f.escala_grises(imagen), "RGB"

        if operacion == "canal":

            canales = {

                "Rojo": 0,

                "Verde": 1,

                "Azul": 2

            }

            canal = canales[self.canal_var.get()]

            return f.solo_canal(imagen, canal), "RGB"

        if operacion == "rgb_yiq":

            resultado = f.rgbAyiq(

                imagen,

                self.luminancia_var.get(),

                self.cromaticidad_var.get()

            )

            return resultado, "YIQ"

        if operacion == "yiq_rgb":

            return f.yiqArgb(imagen), "RGB"

        if operacion == "raiz":

            return f.op_raiz(imagen)[0], "RGB"

        if operacion == "cuadrado":

            return f.op_cuadrado(imagen)[0], "RGB"

        if operacion == "lineal":

            y_min = float(self.y_min_var.get())

            y_max = float(self.y_max_var.get())

            if y_min >= y_max:

                raise ValueError("Y mínimo debe ser menor que Y máximo.")

            return f.op_lineal_a_trazos(imagen, y_min, y_max)[0], "RGB"

        if operacion == "normalizada":

            return f.op_normalizada(imagen)[0], "RGB"

        if operacion == "ecualizacion":

            return f.op_ecualizacion(imagen)[0], "RGB"

        if operacion == "suma_clipeada":

            return f.sumaClipeada(imagen, self.imagen_segunda), "RGB"

        if operacion == "suma_promedio":

            return f.sumaPromedio(imagen, self.imagen_segunda), "RGB"

        if operacion == "resta_clipeada":

            return f.restaClipeada(imagen, self.imagen_segunda), "RGB"

        if operacion == "resta_promedio":

            return f.restaPromedio(imagen, self.imagen_segunda), "RGB"

        if operacion == "resta_absoluto":

            return f.restaValorAbsoluto(imagen, self.imagen_segunda), "RGB"

        if operacion == "producto":

            return f.producto(imagen, self.imagen_segunda), "RGB"

        if operacion == "cociente":

            return f.cociente(imagen, self.imagen_segunda), "RGB"

        if operacion == "interpolar":

            return f.interpolar_imagenes(imagen, self.imagen_segunda), "RGB"

        if operacion == "darker":

            return f.if_darker(imagen, self.imagen_segunda), "RGB"

        if operacion == "lighter":

            return f.if_lighter(imagen, self.imagen_segunda), "RGB"

# ----------------------------------------------------

# Filtros: se trabaja sobre la luminancia Y.

# ----------------------------------------------------

        if operacion in {
            "plano", "bartlett", "gaussiano",
            "pasa_bajos", "combinar_kernels",
            "lap_v4", "lap_v8",
            "sobel_este", "sobel_norte", "sobel_sur", "sobel_oeste",
            "sobel_noreste", "sobel_sureste", "sobel_suroeste", "sobel_noroeste",
            "pasabanda", "pasaaltos_02", "pasaaltos_04"
        }:
            yiq = f.rgbAyiq(imagen)
            y = yiq[:, :, 0]

            if operacion == "pasa_bajos":
                tamanio = int(self.tamanio_kernel_var.get())
                tipo = self.tipo_pasabajo_var.get()

                if tipo == "Plano":
                    kernel = f.kernel_plano(tamanio)
                elif tipo == "Bartlett":
                    kernel = f.kernel_bartlett(tamanio)
                else:
                    kernel = f.kernel_gaussiano(tamanio)

                resultado = f.convolucion(y, kernel)

            elif operacion == "combinar_kernels":
                tamanio = int(self.tamanio_kernel_var.get())
                kernel1 = self.obtener_kernel_combinacion(
                    self.kernel1_var.get(), tamanio
                )
                kernel2 = self.obtener_kernel_combinacion(
                    self.kernel2_var.get(), tamanio
                )
                porcentaje = float(self.porcentaje_kernel_var.get())
                kernel = f.combinar_kernels(kernel1, kernel2, porcentaje)

                if kernel is None:
                    raise ValueError("No se pudieron combinar los kernels seleccionados.")

                resultado = f.convolucion(y, kernel)

            elif operacion == "plano":
                tamanio = int(self.tamanio_kernel_var.get())
                kernel = f.kernel_plano(tamanio)
                resultado = f.convolucion(y, kernel)

            elif operacion == "bartlett":
                tamanio = int(self.tamanio_kernel_var.get())
                kernel = f.kernel_bartlett(tamanio)
                resultado = f.convolucion(y, kernel)

            elif operacion == "gaussiano":
                tamanio = int(self.tamanio_kernel_var.get())
                kernel = f.kernel_gaussiano(tamanio)
                resultado = f.convolucion(y, kernel)

            elif operacion == "lap_v4":
                resultado = f.convolucion(y, f.kernel_laplaciano_v4())

            elif operacion == "lap_v8":
                resultado = f.convolucion(y, f.kernel_laplaciano_v8())

            elif operacion == "sobel_este":
                resultado = f.convolucion(y, f.kernel_sobel_este())

            elif operacion == "sobel_norte":
                resultado = f.convolucion(y, f.kernel_sobel_norte())

            elif operacion == "sobel_sur":
                resultado = f.convolucion(y, f.kernel_sobel_sur())

            elif operacion == "sobel_oeste":
                resultado = f.convolucion(y, f.kernel_sobel_oeste())

            elif operacion == "sobel_noreste":
                resultado = f.convolucion(y, f.kernel_sobel_noreste())

            elif operacion == "sobel_sureste":
                resultado = f.convolucion(y, f.kernel_sobel_sureste())

            elif operacion == "sobel_suroeste":
                resultado = f.convolucion(y, f.kernel_sobel_suroeste())

            elif operacion == "sobel_noroeste":
                resultado = f.convolucion(y, f.kernel_sobel_noroeste())

            elif operacion == "pasabanda":
                resultado = f.pasabanda_diff_gauss(
                    y,
                    f.kernel_gaussiano(3),
                    f.kernel_gaussiano(5)
                )

            elif operacion == "pasaaltos_02":
                kernel = f.kernel_pasaaltos(0.2)
                resultado = f.convolucion(y, kernel)

            else:
                kernel = f.kernel_pasaaltos(0.4)
                resultado = f.convolucion(y, kernel)

            # Los filtros de bordes pueden dar valores negativos.
            # Se muestran con el cero en gris, sin recortar los negativos.
            if operacion in {
                "lap_v4", "lap_v8",
                "sobel_este", "sobel_norte", "sobel_sur", "sobel_oeste",
                "sobel_noreste", "sobel_sureste", "sobel_suroeste", "sobel_noroeste",
                "pasabanda", "pasaaltos_02", "pasaaltos_04"
            }:
                maximo = np.max(np.abs(resultado))

                if maximo > 0:
                    resultado = resultado / (2 * maximo) + 0.5
                else:
                    resultado = np.zeros(resultado.shape) + 0.5
            else:
                resultado = np.clip(resultado, 0.0, 1.0)

            resultado = np.clip(resultado, 0.0, 1.0)
            resultado_rgb = np.stack(
                [resultado, resultado, resultado],
                axis=2
            )
            return resultado_rgb, "RGB"

        raise ValueError("La operación seleccionada no está implementada.")

    def mostrar_error(self, mensaje):

        self.estado.config(

            text="Error: revisá la operación y sus parámetros.",

            fg=COLOR_ERROR

        )

        self.ventana.update_idletasks()

        messagebox.showerror("Error", mensaje)

# ========================================================

# VISUALIZACIÓN Y GUARDADO

# ========================================================

    def preparar_rgb_para_mostrar(self, array_imagen, espacio_color):

        if espacio_color == "YIQ":

            imagen_rgb = f.yiqArgb(array_imagen)

        elif array_imagen.ndim == 2:

            imagen_rgb = np.stack(

                [array_imagen, array_imagen, array_imagen],

                axis=2

            )

        else:

            imagen_rgb = array_imagen

        return np.clip(imagen_rgb, 0.0, 1.0)

    def mostrar_imagen(

        self,

        label,

        array_imagen,

        espacio_color="RGB",

        tamanio=(500, 420)

    ):

        imagen_rgb = self.preparar_rgb_para_mostrar(

            array_imagen,

            espacio_color

        )

        imagen_uint8 = (

            np.clip(imagen_rgb, 0, 1) * 255

        ).astype(np.uint8)

        imagen_pil = Image.fromarray(imagen_uint8)

        imagen_pil.thumbnail(tamanio)

        foto = ImageTk.PhotoImage(imagen_pil)

        label.foto = foto

        label.config(

            image=foto,

            text=""

        )

        if hasattr(self, "histograma_entrada"):
            if label is self.label_entrada:
                self.actualizar_histograma(
                    self.histograma_entrada,
                    array_imagen,
                    espacio_color
                )
            elif label is self.label_segunda:
                self.actualizar_histograma(
                    self.histograma_segunda,
                    array_imagen,
                    espacio_color
                )
            elif label is self.label_procesada:
                self.actualizar_histograma(
                    self.histograma_resultado,
                    array_imagen,
                    espacio_color
                )

    def limpiar_resultado(self):

        self.label_procesada.config(

            image="",

            text="El resultado aparecerá aquí"

        )

        self.label_procesada.foto = None

        if hasattr(self, "histograma_resultado"):
            self.actualizar_histograma(
                self.histograma_resultado,
                None
            )

    def actualizar_titulos(self):

        self.titulo_entrada.config(

            text=(

                f"IMAGEN DE ENTRADA • Espacio de color: "

                f"{self.espacio_color}"

            )

        )

        if self.imagen_procesada is None:

            texto_resultado = "IMAGEN PROCESADA • Sin resultado"

        else:

            texto_resultado = (

                f"IMAGEN PROCESADA • Espacio de color: "

                f"{self.espacio_procesada}"

            )

            if self.espacio_procesada == "YIQ":

                texto_resultado += "  •  vista RGB"

        self.titulo_procesada.config(text=texto_resultado)

    def guardar_procesada(self):

        if self.imagen_procesada is None:

            self.estado.config(

                text="No hay una imagen procesada para guardar.",

                fg=COLOR_ERROR

            )

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

            imagen_rgb = self.preparar_rgb_para_mostrar(

                self.imagen_procesada,

                self.espacio_procesada

            )

            imagen_uint8 = (

                np.clip(imagen_rgb, 0, 1) * 255

            ).astype(np.uint8)

            Image.fromarray(imagen_uint8).save(ruta)

            self.estado.config(

                text=f"Imagen guardada correctamente: {ruta}",

                fg=COLOR_EXITO

            )

        except Exception as error:

            self.estado.config(

                text="Error al guardar la imagen.",

                fg=COLOR_ERROR

            )

            messagebox.showerror(

                "Error",

                f"No se pudo guardar la imagen.\n\n{error}"

            )

# ============================================================

# INICIO

# ============================================================

if __name__ == "__main__":

    ventana = tk.Tk()

    app = AppPDI(ventana)

    ventana.mainloop()