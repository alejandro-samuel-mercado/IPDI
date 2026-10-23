import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk


COLOR_FONDO = "#E8EEF5"
COLOR_PANEL = "#F8FAFC"
COLOR_CONTROLES = "#EEF3F8"
COLOR_IMAGEN = "#DCE4EE"
COLOR_PRIMARIO = "#2563EB"
COLOR_HOVER = "#1D4ED8"
COLOR_TEXTO = "#1E293B"
COLOR_SECUNDARIO = "#64748B"
COLOR_BORDE = "#CBD5E1"
COLOR_BLANCO = "#FFFFFF"


# ============================================================
# FUNCIONES DE CONVOLUCIÓN Y CONSTRUCCIÓN DE KERNELS
# ============================================================

def rgb_a_luminancia(imagen_rgb):
    """Obtiene la luminancia Y de una imagen RGB normalizada entre 0 y 1."""
    imagen_rgb = np.asarray(imagen_rgb, dtype=float)
    if imagen_rgb.ndim == 2:
        return imagen_rgb.copy()

    rojo = imagen_rgb[:, :, 0]
    verde = imagen_rgb[:, :, 1]
    azul = imagen_rgb[:, :, 2]
    return 0.299 * rojo + 0.587 * verde + 0.114 * azul


def convolucion(imagen, kernel):
  
    imagen = np.asarray(imagen, dtype=float)
    kernel = np.asarray(kernel, dtype=float)

    if imagen.ndim != 2:
        raise ValueError("La convolución debe recibir una imagen de un solo canal.")
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("El kernel debe ser una matriz de dimensiones impares.")

    alto, ancho = imagen.shape
    borde_y = kernel.shape[0] // 2
    borde_x = kernel.shape[1] // 2
    ampliada = np.pad(imagen, ((borde_y, borde_y), (borde_x, borde_x)), mode="edge")
    resultado = np.zeros((alto, ancho), dtype=float)

    # Cada valor del kernel multiplica la zona correspondiente de la imagen.
    for fila in range(kernel.shape[0]):
        for columna in range(kernel.shape[1]):
            zona = ampliada[fila:fila + alto, columna:columna + ancho]
            resultado += zona * kernel[fila, columna]

    return resultado


def kernel_identidad(tamanio=3):
    kernel = np.zeros((tamanio, tamanio), dtype=float)
    kernel[tamanio // 2, tamanio // 2] = 1.0
    return kernel


def kernel_plano(tamanio=3):
    """Filtro pasa bajos: todos los coeficientes tienen el mismo peso."""
    if tamanio not in (3, 5, 7):
        raise ValueError("El tamaño del kernel plano debe ser 3, 5 o 7.")
    return np.ones((tamanio, tamanio), dtype=float) / (tamanio * tamanio)


def kernel_bartlett(tamanio=3):
    """Filtro pasa bajos triangular de Bartlett, normalizado para sumar 1."""
    if tamanio not in (3, 5, 7):
        raise ValueError("El tamaño del kernel Bartlett debe ser 3, 5 o 7.")
    centro = tamanio // 2
    vector = np.array([centro + 1 - abs(i - centro) for i in range(tamanio)], dtype=float)
    kernel = np.outer(vector, vector)
    return kernel / kernel.sum()


def kernel_gaussiano(tamanio=5):
    """Aproximación gaussiana usando coeficientes binomiales."""
    vectores = {
        3: [1, 2, 1],
        5: [1, 4, 6, 4, 1],
        7: [1, 6, 15, 20, 15, 6, 1]
    }
    if tamanio not in vectores:
        raise ValueError("El tamaño del kernel gaussiano debe ser 3, 5 o 7.")

    vector = np.array(vectores[tamanio], dtype=float)
    kernel = np.outer(vector, vector)
    return kernel / kernel.sum()


def kernel_laplaciano(vecinos=4):
    """Kernel para resaltar variaciones bruscas de intensidad."""
    if vecinos == 4:
        return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float)
    if vecinos == 8:
        return np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float)
    raise ValueError("El Laplaciano admite 4 u 8 vecinos.")


def kernel_sobel_direccional(direccion):
    """Devuelve uno de los ocho kernels direccionales usados en el segundo código."""
    oeste = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    suroeste = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=float)
    direccion = direccion.strip().lower()

    kernels = {
        "oeste": oeste,
        "este": np.fliplr(oeste),
        "norte": oeste.T,
        "sur": np.flipud(oeste.T),
        "suroeste": suroeste,
        "sureste": np.fliplr(suroeste),
        "noroeste": np.flipud(suroeste),
        "noreste": np.fliplr(np.flipud(suroeste))
    }
    if direccion not in kernels:
        raise ValueError("Dirección Sobel no reconocida.")
    return kernels[direccion].copy()


def ajustar_kernel(kernel, tamanio):
    """Centra un kernel pequeño en una matriz mayor rellenando con ceros."""
    kernel = np.asarray(kernel, dtype=float)
    if kernel.shape == (tamanio, tamanio):
        return kernel.copy()
    if kernel.shape[0] > tamanio or kernel.shape[1] > tamanio:
        raise ValueError("No se puede ajustar este kernel al tamaño elegido.")

    diferencia_filas = tamanio - kernel.shape[0]
    diferencia_columnas = tamanio - kernel.shape[1]
    arriba = diferencia_filas // 2
    izquierda = diferencia_columnas // 2
    return np.pad(
        kernel,
        ((arriba, diferencia_filas - arriba),
         (izquierda, diferencia_columnas - izquierda)),
        mode="constant"
    )


def kernel_pasabanda():
    gaussiano_5 = kernel_gaussiano(5)
    gaussiano_3 = ajustar_kernel(kernel_gaussiano(3), 5)
    return gaussiano_5 - gaussiano_3


def kernel_pasaaltos(peso=0.2):
    """Realce de detalles: identidad más un Laplaciano V8 ponderado."""
    if not 0 <= peso <= 1:
        raise ValueError("El peso del pasaaltos debe estar entre 0 y 1.")
    return kernel_identidad(3) + peso * kernel_laplaciano(8)


def obtener_kernel(nombre, tamanio=3):
    """Construye el kernel indicado para aplicar o combinar filtros."""
    if nombre == "Identidad":
        return kernel_identidad(tamanio)
    if nombre == "Plano":
        return kernel_plano(tamanio)
    if nombre == "Bartlett":
        return kernel_bartlett(tamanio)
    if nombre == "Gaussiano":
        return kernel_gaussiano(tamanio)
    if nombre == "Laplaciano V4":
        return ajustar_kernel(kernel_laplaciano(4), tamanio)
    if nombre == "Laplaciano V8":
        return ajustar_kernel(kernel_laplaciano(8), tamanio)
    if nombre.startswith("Sobel "):
        return ajustar_kernel(kernel_sobel_direccional(nombre[6:]), tamanio)
    raise ValueError(f"No se reconoce el kernel: {nombre}.")


def combinar_kernels(kernel1, kernel2, peso_segundo=0.2):
    """Combina dos kernels: (1 - peso) * kernel1 + peso * kernel2."""
    if not 0 <= peso_segundo <= 1:
        raise ValueError("El peso del segundo kernel debe estar entre 0 y 1.")
    if kernel1.shape != kernel2.shape:
        raise ValueError("Los kernels deben tener el mismo tamaño para combinarlos.")
    return (1 - peso_segundo) * kernel1 + peso_segundo * kernel2


def normalizar_resultado(resultado, kernel):
    """Ajusta una salida a [0,1], centrando en gris los resultados con signo."""
    if abs(np.sum(kernel)) < 1e-8:
        maximo = np.max(np.abs(resultado))
        if maximo == 0:
            return np.full(resultado.shape, 0.5, dtype=float)
        return np.clip(resultado / (2 * maximo) + 0.5, 0.0, 1.0)
    return np.clip(resultado, 0.0, 1.0)


def kernel_a_texto(kernel):
    """Convierte la matriz del kernel en texto para mostrarla en la interfaz."""
    return "\n".join("  ".join(f"{valor:7.3f}" for valor in fila) for fila in kernel)


# ============================================================
# APLICACIÓN GRÁFICA
# ============================================================

class AppConvolucion:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("PDI - Convolución y filtros por kernels")
        self.ventana.geometry("1320x900")
        self.ventana.minsize(1080, 720)
        self.ventana.configure(bg=COLOR_FONDO)

        self.imagen_original = None
        self.imagen_entrada = None
        self.imagen_procesada = None
        self.foto_entrada = None
        self.foto_procesada = None

        self.operacion_var = tk.StringVar(value="Pasa bajos")
        self.tipo_pasabajo_var = tk.StringVar(value="Plano")
        self.tamanio_var = tk.StringVar(value="3")
        self.kernel1_var = tk.StringVar(value="Identidad")
        self.kernel2_var = tk.StringVar(value="Laplaciano V8")
        self.peso_var = tk.StringVar(value="0.2")

        self.crear_interfaz()
        self.actualizar_controles()
        self.actualizar_kernel_preview()

    def crear_interfaz(self):
        barra = tk.Frame(self.ventana, bg=COLOR_FONDO)
        barra.pack(fill="x", padx=16, pady=(12, 7))

        self.crear_boton(barra, "Abrir imagen", self.abrir_imagen, principal=True).pack(side="left", padx=4)
        self.crear_boton(barra, "Restaurar original", self.restaurar).pack(side="left", padx=4)
        self.crear_boton(barra, "Usar procesada como entrada", self.usar_procesada).pack(side="left", padx=4)
        self.crear_boton(barra, "Guardar resultado", self.guardar_resultado).pack(side="left", padx=4)

        cuerpo = tk.Frame(self.ventana, bg=COLOR_FONDO)
        cuerpo.pack(fill="both", expand=True, padx=12, pady=5)

        self.panel_entrada = self.crear_panel_imagen(cuerpo, "IMAGEN DE ENTRADA")
        self.panel_entrada.pack(side="left", fill="both", expand=True, padx=5)

        self.crear_panel_controles(cuerpo)

        self.panel_salida = self.crear_panel_imagen(cuerpo, "IMAGEN RESULTANTE")
        self.panel_salida.pack(side="left", fill="both", expand=True, padx=5)

        self.histogramas = tk.Frame(self.ventana, bg=COLOR_FONDO)
        self.histogramas.pack(fill="x", padx=17, pady=(2, 5))
        self.histograma_entrada = self.crear_histograma(self.histogramas, "Histograma · Entrada")
        self.histograma_entrada.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.histograma_salida = self.crear_histograma(self.histogramas, "Histograma · Resultado")
        self.histograma_salida.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self.estado = tk.Label(
            self.ventana, text="Abrí una imagen para comenzar.", anchor="w",
            bg=COLOR_FONDO, fg=COLOR_SECUNDARIO, font=("Arial", 10)
        )
        self.estado.pack(fill="x", padx=19, pady=(2, 10))

    def crear_boton(self, padre, texto, comando, principal=False):
        return tk.Button(
            padre, text=texto, command=comando,
            bg=COLOR_PRIMARIO if principal else COLOR_BLANCO,
            fg=COLOR_BLANCO if principal else COLOR_TEXTO,
            activebackground=COLOR_HOVER if principal else COLOR_IMAGEN,
            activeforeground=COLOR_BLANCO if principal else COLOR_TEXTO,
            font=("Arial", 9, "bold"), padx=9, pady=6,
            relief="flat" if principal else "solid", bd=0 if principal else 1,
            cursor="hand2"
        )

    def crear_panel_imagen(self, padre, titulo):
        panel = tk.Frame(padre, bg=COLOR_PANEL, bd=1, relief="solid")
        tk.Label(
            panel, text=titulo, font=("Arial", 11, "bold"),
            bg=COLOR_PANEL, fg=COLOR_TEXTO
        ).pack(pady=(10, 6))
        etiqueta = tk.Label(
            panel, text="Abrí una imagen para comenzar" if "ENTRADA" in titulo else "El resultado aparecerá aquí",
            bg=COLOR_IMAGEN, fg=COLOR_SECUNDARIO, font=("Arial", 10)
        )
        etiqueta.pack(fill="both", expand=True, padx=9, pady=(0, 9))
        panel.etiqueta_imagen = etiqueta
        return panel

    def crear_panel_controles(self, padre):
        panel = tk.Frame(padre, width=310, bg=COLOR_CONTROLES, bd=1, relief="solid")
        panel.pack(side="left", fill="y", padx=5)
        panel.pack_propagate(False)
        tk.Label(panel, text="CONVOLUCIÓN", font=("Arial", 13, "bold"), bg=COLOR_CONTROLES, fg=COLOR_TEXTO).pack(pady=(12, 8))

        tk.Label(panel, text="Operación:", font=("Arial", 9, "bold"), bg=COLOR_CONTROLES, fg=COLOR_TEXTO).pack(anchor="w", padx=10)
        operaciones = [
            "Pasa bajos", "Laplaciano V4", "Laplaciano V8",
            "Sobel Este", "Sobel Oeste", "Sobel Norte", "Sobel Sur",
            "Sobel Noreste", "Sobel Noroeste", "Sobel Sureste", "Sobel Suroeste",
            "Pasa banda (DoG)", "Pasaaltos 0.2", "Pasaaltos 0.4", "Combinar kernels"
        ]
        self.combo_operacion = ttk.Combobox(panel, textvariable=self.operacion_var, values=operaciones, state="readonly")
        self.combo_operacion.pack(fill="x", padx=10, pady=(3, 10))
        self.combo_operacion.bind("<<ComboboxSelected>>", self.al_cambiar_operacion)

        self.marco_pasabajos = tk.LabelFrame(panel, text="Parámetros del pasa bajos", bg=COLOR_CONTROLES, fg=COLOR_TEXTO, font=("Arial", 9, "bold"))
        tk.Label(self.marco_pasabajos, text="Tipo de kernel:", bg=COLOR_CONTROLES, fg=COLOR_TEXTO).pack(anchor="w", padx=7, pady=(5, 2))
        self.combo_tipo = ttk.Combobox(self.marco_pasabajos, textvariable=self.tipo_pasabajo_var, values=["Plano", "Bartlett", "Gaussiano"], state="readonly")
        self.combo_tipo.pack(fill="x", padx=7)
        self.combo_tipo.bind("<<ComboboxSelected>>", self.al_cambiar_tipo_pasabajo)
        tk.Label(self.marco_pasabajos, text="Tamaño:", bg=COLOR_CONTROLES, fg=COLOR_TEXTO).pack(anchor="w", padx=7, pady=(7, 2))
        self.combo_tamanio = ttk.Combobox(self.marco_pasabajos, textvariable=self.tamanio_var, values=["3", "5", "7"], state="readonly")
        self.combo_tamanio.pack(fill="x", padx=7, pady=(0, 7))
        self.combo_tamanio.bind("<<ComboboxSelected>>", self.actualizar_kernel_preview)

        self.marco_combinacion = tk.LabelFrame(panel, text="Combinar kernels", bg=COLOR_CONTROLES, fg=COLOR_TEXTO, font=("Arial", 9, "bold"))
        opciones_kernels = [
            "Identidad", "Plano", "Bartlett", "Gaussiano", "Laplaciano V4", "Laplaciano V8",
            "Sobel Este", "Sobel Oeste", "Sobel Norte", "Sobel Sur",
            "Sobel Noreste", "Sobel Noroeste", "Sobel Sureste", "Sobel Suroeste"
        ]
        tk.Label(self.marco_combinacion, text="Primer kernel:", bg=COLOR_CONTROLES, fg=COLOR_TEXTO).pack(anchor="w", padx=7, pady=(5, 2))
        self.combo_kernel1 = ttk.Combobox(self.marco_combinacion, textvariable=self.kernel1_var, values=opciones_kernels, state="readonly")
        self.combo_kernel1.pack(fill="x", padx=7)
        self.combo_kernel1.bind("<<ComboboxSelected>>", self.actualizar_kernel_preview)
        tk.Label(self.marco_combinacion, text="Segundo kernel:", bg=COLOR_CONTROLES, fg=COLOR_TEXTO).pack(anchor="w", padx=7, pady=(7, 2))
        self.combo_kernel2 = ttk.Combobox(self.marco_combinacion, textvariable=self.kernel2_var, values=opciones_kernels, state="readonly")
        self.combo_kernel2.pack(fill="x", padx=7)
        self.combo_kernel2.bind("<<ComboboxSelected>>", self.actualizar_kernel_preview)
        tk.Label(self.marco_combinacion, text="Tamaño común (3, 5 o 7):", bg=COLOR_CONTROLES, fg=COLOR_TEXTO).pack(anchor="w", padx=7, pady=(7, 2))
        self.combo_tamanio_combinacion = ttk.Combobox(self.marco_combinacion, textvariable=self.tamanio_var, values=["3", "5", "7"], state="readonly")
        self.combo_tamanio_combinacion.pack(fill="x", padx=7)
        self.combo_tamanio_combinacion.bind("<<ComboboxSelected>>", self.actualizar_kernel_preview)
        tk.Label(self.marco_combinacion, text="Peso del segundo kernel (0 a 1):", bg=COLOR_CONTROLES, fg=COLOR_TEXTO).pack(anchor="w", padx=7, pady=(7, 2))
        self.entrada_peso = tk.Entry(self.marco_combinacion, textvariable=self.peso_var)
        self.entrada_peso.pack(fill="x", padx=7, pady=(0, 7))
        self.entrada_peso.bind("<KeyRelease>", self.actualizar_kernel_preview)

        tk.Label(panel, text="Kernel aplicado:", font=("Arial", 9, "bold"), bg=COLOR_CONTROLES, fg=COLOR_TEXTO).pack(anchor="w", padx=10, pady=(8, 2))
        self.label_kernel = tk.Label(
            panel, text="", justify="center", anchor="center",
            font=("Courier New", 9), bg=COLOR_BLANCO, fg=COLOR_TEXTO,
            bd=1, relief="solid", padx=4, pady=6
        )
        self.label_kernel.pack(fill="x", padx=10, pady=(0, 8))

        self.crear_boton(panel, "Procesar imagen", self.procesar, principal=True).pack(fill="x", padx=10, pady=(4, 10))
       
    def crear_histograma(self, padre, titulo):
        panel = tk.LabelFrame(padre, text=titulo, bg=COLOR_PANEL, fg=COLOR_TEXTO, font=("Arial", 9, "bold"), bd=1, relief="solid")
        canvas = tk.Canvas(panel, height=125, bg=COLOR_BLANCO, highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=5, pady=5)
        canvas.datos_histograma = None
        canvas.bind("<Configure>", lambda evento, c=canvas: self.dibujar_histograma(c))
        return panel

    def obtener_canvas_histograma(self, panel):
        return panel.winfo_children()[0]

    def actualizar_histograma(self, panel, imagen):
        canvas = self.obtener_canvas_histograma(panel)
        canvas.datos_histograma = None if imagen is None else rgb_a_luminancia(imagen).ravel()
        self.dibujar_histograma(canvas)

    def dibujar_histograma(self, canvas):
        canvas.delete("all")
        ancho = max(canvas.winfo_width(), 300)
        alto = max(canvas.winfo_height(), 100)
        datos = canvas.datos_histograma
        if datos is None or len(datos) == 0:
            canvas.create_text(ancho / 2, alto / 2, text="Sin datos", fill=COLOR_SECUNDARIO, font=("Arial", 9))
            return

        histograma, _ = np.histogram(np.clip(datos, 0, 1), bins=32, range=(0, 1))
        porcentajes = histograma * 100.0 / len(datos)
        maximo = max(float(porcentajes.max()), 1.0)
        margen_izq, margen_der, margen_arriba, margen_abajo = 42, 9, 8, 22
        grafico_ancho = ancho - margen_izq - margen_der
        grafico_alto = alto - margen_arriba - margen_abajo
        base = alto - margen_abajo

        for fraccion in (0, 0.5, 1):
            y = base - fraccion * grafico_alto
            canvas.create_line(margen_izq, y, ancho - margen_der, y, fill="#E2E8F0")
            canvas.create_text(margen_izq - 4, y, text=f"{maximo * fraccion:.1f}%", anchor="e", fill=COLOR_SECUNDARIO, font=("Arial", 7))

        paso = grafico_ancho / len(porcentajes)
        for indice, porcentaje in enumerate(porcentajes):
            x0 = margen_izq + indice * paso + 1
            x1 = margen_izq + (indice + 1) * paso - 1
            altura = porcentaje / maximo * grafico_alto
            canvas.create_rectangle(x0, base - altura, x1, base, fill=COLOR_PRIMARIO, outline="")

        for nivel in (0, 64, 128, 192, 255):
            x = margen_izq + nivel / 255 * grafico_ancho
            canvas.create_text(x, alto - 10, text=str(nivel), fill=COLOR_SECUNDARIO, font=("Arial", 7))

    def al_cambiar_operacion(self, evento=None):
        self.actualizar_controles()
        self.actualizar_kernel_preview()

    def al_cambiar_tipo_pasabajo(self, evento=None):
        if self.tipo_pasabajo_var.get() == "Gaussiano":
            valores = ["5", "7"]
            if self.tamanio_var.get() not in valores:
                self.tamanio_var.set("5")
        else:
            valores = ["3", "5", "7"]
        self.combo_tamanio.configure(values=valores)
        self.actualizar_kernel_preview()

    def actualizar_controles(self):
        self.marco_pasabajos.pack_forget()
        self.marco_combinacion.pack_forget()
        operacion = self.operacion_var.get()

        if operacion == "Pasa bajos":
            self.marco_pasabajos.pack(fill="x", padx=8, pady=(0, 5))
            if self.tipo_pasabajo_var.get() == "Gaussiano":
                self.combo_tamanio.configure(values=["5", "7"])
                if self.tamanio_var.get() not in ("5", "7"):
                    self.tamanio_var.set("5")
            else:
                self.combo_tamanio.configure(values=["3", "5", "7"])
        elif operacion == "Combinar kernels":
            self.marco_combinacion.pack(fill="x", padx=8, pady=(0, 5))

    def construir_kernel_seleccionado(self):
        operacion = self.operacion_var.get()
        tamanio = int(self.tamanio_var.get())

        if operacion == "Pasa bajos":
            tipo = self.tipo_pasabajo_var.get()
            if tipo == "Plano":
                return kernel_plano(tamanio)
            if tipo == "Bartlett":
                return kernel_bartlett(tamanio)
            return kernel_gaussiano(tamanio)
        if operacion == "Laplaciano V4":
            return kernel_laplaciano(4)
        if operacion == "Laplaciano V8":
            return kernel_laplaciano(8)
        if operacion.startswith("Sobel "):
            return kernel_sobel_direccional(operacion[6:])
        if operacion == "Pasa banda (DoG)":
            return kernel_pasabanda()
        if operacion == "Pasaaltos 0.2":
            return kernel_pasaaltos(0.2)
        if operacion == "Pasaaltos 0.4":
            return kernel_pasaaltos(0.4)
        if operacion == "Combinar kernels":
            kernel1 = obtener_kernel(self.kernel1_var.get(), tamanio)
            kernel2 = obtener_kernel(self.kernel2_var.get(), tamanio)
            peso = float(self.peso_var.get().replace(",", "."))
            return combinar_kernels(kernel1, kernel2, peso)
        raise ValueError("La operación seleccionada no está implementada.")

    def actualizar_kernel_preview(self, evento=None):
        try:
            kernel = self.construir_kernel_seleccionado()
            self.label_kernel.config(text=kernel_a_texto(kernel))
        except (ValueError, TypeError):
            self.label_kernel.config(text="Revisá los parámetros")

    def mostrar_imagen(self, etiqueta, imagen):
        if imagen is None:
            etiqueta.config(image="", text="Sin imagen")
            etiqueta.image_ref = None
            return

        imagen = np.asarray(imagen, dtype=float)
        if imagen.ndim == 2:
            imagen = np.stack([imagen, imagen, imagen], axis=2)
        datos = (np.clip(imagen, 0, 1) * 255).round().astype(np.uint8)
        imagen_pil = Image.fromarray(datos, mode="RGB")
        imagen_pil.thumbnail((510, 410), Image.Resampling.LANCZOS)
        foto = ImageTk.PhotoImage(imagen_pil)
        etiqueta.config(image=foto, text="")
        etiqueta.image_ref = foto

    def abrir_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not ruta:
            return
        try:
            imagen_pil = Image.open(ruta).convert("RGB")
            imagen = np.asarray(imagen_pil, dtype=float) / 255.0
            self.imagen_original = imagen.copy()
            self.imagen_entrada = imagen.copy()
            self.imagen_procesada = None
            self.mostrar_imagen(self.panel_entrada.etiqueta_imagen, self.imagen_entrada)
            self.mostrar_imagen(self.panel_salida.etiqueta_imagen, None)
            self.actualizar_histograma(self.histograma_entrada, self.imagen_entrada)
            self.actualizar_histograma(self.histograma_salida, None)
            self.estado.config(text=f"Imagen cargada: {imagen.shape[1]} × {imagen.shape[0]} píxeles.", fg=COLOR_SECUNDARIO)
        except Exception as error:
            messagebox.showerror("Error al abrir", f"No se pudo abrir la imagen.\n\n{error}")

    def procesar(self):
        if self.imagen_entrada is None:
            messagebox.showwarning("Atención", "Primero abrí una imagen.")
            return
        try:
            kernel = self.construir_kernel_seleccionado()
            luminancia = rgb_a_luminancia(self.imagen_entrada)
            valores = convolucion(luminancia, kernel)
            resultado = normalizar_resultado(valores, kernel)

            self.imagen_procesada = np.stack([resultado, resultado, resultado], axis=2)
            self.mostrar_imagen(self.panel_salida.etiqueta_imagen, self.imagen_procesada)
            self.actualizar_histograma(self.histograma_salida, self.imagen_procesada)
            self.estado.config(
                text=f"Filtro aplicado: {self.operacion_var.get()} · kernel {kernel.shape[0]}×{kernel.shape[1]} · salida {resultado.shape[1]}×{resultado.shape[0]}.",
                fg=COLOR_TEXTO
            )
        except (ValueError, TypeError, ZeroDivisionError) as error:
            messagebox.showerror("Parámetros incorrectos", str(error))
            self.estado.config(text="No se pudo aplicar el filtro. Revisá los parámetros.", fg="#B91C1C")

    def usar_procesada(self):
        if self.imagen_procesada is None:
            messagebox.showwarning("Atención", "Primero procesá una imagen.")
            return
        self.imagen_entrada = self.imagen_procesada.copy()
        self.imagen_procesada = None
        self.mostrar_imagen(self.panel_entrada.etiqueta_imagen, self.imagen_entrada)
        self.mostrar_imagen(self.panel_salida.etiqueta_imagen, None)
        self.actualizar_histograma(self.histograma_entrada, self.imagen_entrada)
        self.actualizar_histograma(self.histograma_salida, None)
        self.estado.config(text="La imagen procesada ahora es la entrada para el siguiente filtro.", fg=COLOR_TEXTO)

    def restaurar(self):
        if self.imagen_original is None:
            messagebox.showwarning("Atención", "Primero abrí una imagen.")
            return
        self.imagen_entrada = self.imagen_original.copy()
        self.imagen_procesada = None
        self.mostrar_imagen(self.panel_entrada.etiqueta_imagen, self.imagen_entrada)
        self.mostrar_imagen(self.panel_salida.etiqueta_imagen, None)
        self.actualizar_histograma(self.histograma_entrada, self.imagen_entrada)
        self.actualizar_histograma(self.histograma_salida, None)
        self.estado.config(text="Se restauró la imagen original.", fg=COLOR_TEXTO)

    def guardar_resultado(self):
        if self.imagen_procesada is None:
            messagebox.showwarning("Atención", "No hay un resultado para guardar.")
            return
        ruta = filedialog.asksaveasfilename(
            title="Guardar imagen resultante",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("BMP", "*.bmp")]
        )
        if not ruta:
            return
        try:
            datos = (np.clip(self.imagen_procesada, 0, 1) * 255).round().astype(np.uint8)
            Image.fromarray(datos, mode="RGB").save(ruta)
            self.estado.config(text=f"Imagen guardada: {ruta}", fg=COLOR_TEXTO)
        except Exception as error:
            messagebox.showerror("Error al guardar", f"No se pudo guardar la imagen.\n\n{error}")


def probar_funciones():
    """Pruebas básicas rápidas para comprobar formas, límites y casos neutros."""
    imagen = np.full((9, 11), 0.4, dtype=float)
    identidad = convolucion(imagen, kernel_identidad(3))
    assert identidad.shape == imagen.shape
    assert np.allclose(identidad, imagen)

    for tamanio in (3, 5, 7):
        for constructor in (kernel_plano, kernel_bartlett, kernel_gaussiano):
            if constructor is kernel_gaussiano and tamanio == 3:
                kernel = constructor(3)
            else:
                kernel = constructor(tamanio)
            assert kernel.shape == (tamanio, tamanio)
            assert np.isclose(kernel.sum(), 1.0)
            assert convolucion(imagen, kernel).shape == imagen.shape

    for vecinos in (4, 8):
        assert np.allclose(convolucion(imagen, kernel_laplaciano(vecinos)), 0.0)
    assert np.allclose(convolucion(imagen, kernel_pasabanda()), 0.0, atol=1e-12)


if __name__ == "__main__":
    probar_funciones()
    ventana = tk.Tk()
    app = AppConvolucion(ventana)
    ventana.mainloop()
