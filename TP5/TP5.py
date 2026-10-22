import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk


# Elemento estructurante en cruz de 3x3.
ELEMENTO_ESTRUCTURANTE = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)


def convertir_escala_grises(imagen_pil):
    rgb = np.asarray(imagen_pil.convert("RGB"), dtype=np.float32)
    rojo = rgb[:, :, 0]
    verde = rgb[:, :, 1]
    azul = rgb[:, :, 2]

    gris = 0.299 * rojo + 0.587 * verde + 0.114 * azul
    return np.clip(np.rint(gris), 0, 255).astype(np.uint8)


def binarizar_umbral(imagen, umbral=127):
    """Devuelve una imagen binaria: blanco si el gris > umbral; negro en otro caso."""
    return np.where(imagen > umbral, 255, 0).astype(np.uint8)


def filtro_minimo_3x3(imagen):
    alto, ancho = imagen.shape
    extendida = np.pad(imagen, 1, mode="edge")
    resultado = np.empty_like(imagen)

    for fila in range(alto):
        for columna in range(ancho):
            vecindad = extendida[fila:fila + 3, columna:columna + 3]
            minimo = 255

            for i in range(3):
                for j in range(3):
                    if ELEMENTO_ESTRUCTURANTE[i, j] == 1:
                        minimo = min(minimo, int(vecindad[i, j]))

            resultado[fila, columna] = minimo

    return resultado


def filtro_maximo_3x3(imagen):
    alto, ancho = imagen.shape
    extendida = np.pad(imagen, 1, mode="edge")
    resultado = np.empty_like(imagen)

    for fila in range(alto):
        for columna in range(ancho):
            vecindad = extendida[fila:fila + 3, columna:columna + 3]
            maximo = 0

            for i in range(3):
                for j in range(3):
                    if ELEMENTO_ESTRUCTURANTE[i, j] == 1:
                        maximo = max(maximo, int(vecindad[i, j]))

            resultado[fila, columna] = maximo

    return resultado


def erosion(imagen):
    return filtro_minimo_3x3(imagen)


def dilatacion(imagen):
    return filtro_maximo_3x3(imagen)


def apertura(imagen):
    """Apertura: primero erosión y luego dilatación."""
    return dilatacion(erosion(imagen))


def cierre(imagen):
    """Cierre: primero dilatación y luego erosión."""
    return erosion(dilatacion(imagen))


def borde_exterior(imagen):
    """Borde exterior en gris: dilatación menos imagen original."""
    expandida = dilatacion(imagen).astype(np.int16)
    original = imagen.astype(np.int16)
    return np.clip(expandida - original, 0, 255).astype(np.uint8)


def borde_interior(imagen):
    """Borde interior en gris: imagen original menos erosión."""
    original = imagen.astype(np.int16)
    reducida = erosion(imagen).astype(np.int16)
    return np.clip(original - reducida, 0, 255).astype(np.uint8)


def gradiente_morfologico(imagen):
    """Gradiente morfológico: dilatación menos erosión."""
    expandida = dilatacion(imagen).astype(np.int16)
    reducida = erosion(imagen).astype(np.int16)
    return np.clip(expandida - reducida, 0, 255).astype(np.uint8)


def top_hat(imagen):
    """Top-hat blanco: imagen original menos su apertura."""
    original = imagen.astype(np.int16)
    abierta = apertura(imagen).astype(np.int16)
    return np.clip(original - abierta, 0, 255).astype(np.uint8)


def mediana_3x3(imagen):
    """Filtro de mediana: reemplaza cada píxel por la mediana de su vecindad 3x3."""
    alto, ancho = imagen.shape
    extendida = np.pad(imagen, 1, mode="edge")
    resultado = np.empty_like(imagen)

    for fila in range(alto):
        for columna in range(ancho):
            vecindad = extendida[fila:fila + 3, columna:columna + 3]
            resultado[fila, columna] = np.median(vecindad)

    return resultado


def erosion_binaria(imagen_binaria):
    """Erosión binaria con fondo negro fuera de la imagen; usada en esqueletización."""
    alto, ancho = imagen_binaria.shape
    extendida = np.pad(imagen_binaria, 1, mode="constant", constant_values=0)
    resultado = np.empty_like(imagen_binaria)

    for fila in range(alto):
        for columna in range(ancho):
            vecindad = extendida[fila:fila + 3, columna:columna + 3]
            minimo = 255
            for i in range(3):
                for j in range(3):
                    if ELEMENTO_ESTRUCTURANTE[i, j] == 1:
                        minimo = min(minimo, int(vecindad[i, j]))
            resultado[fila, columna] = minimo

    return resultado


def dilatacion_binaria(imagen_binaria):
    """Dilatación binaria con fondo negro fuera de la imagen; usada en esqueletización."""
    alto, ancho = imagen_binaria.shape
    extendida = np.pad(imagen_binaria, 1, mode="constant", constant_values=0)
    resultado = np.empty_like(imagen_binaria)

    for fila in range(alto):
        for columna in range(ancho):
            vecindad = extendida[fila:fila + 3, columna:columna + 3]
            maximo = 0
            for i in range(3):
                for j in range(3):
                    if ELEMENTO_ESTRUCTURANTE[i, j] == 1:
                        maximo = max(maximo, int(vecindad[i, j]))
            resultado[fila, columna] = maximo

    return resultado


def esqueletizacion(imagen, umbral=127):
    """Esqueletización binaria por erosiones sucesivas; usa el umbral indicado."""
    actual = binarizar_umbral(imagen, umbral)
    esqueleto = np.zeros_like(actual)

    while np.any(actual):
        abierta = dilatacion_binaria(erosion_binaria(actual))
        residuo = np.where((actual > 0) & (abierta == 0), 255, 0).astype(np.uint8)
        esqueleto = np.maximum(esqueleto, residuo)
        siguiente = erosion_binaria(actual)

        # Protección ante una eventual falta de cambio, para evitar un bucle infinito.
        if np.array_equal(siguiente, actual):
            break
        actual = siguiente

    return esqueleto


def mean_shift(imagen, radio_espacial=3, rango_intensidad=25, max_iteraciones=10):
    """Filtro Mean Shift en escala de grises mediante desplazamiento en (x, y, intensidad).

    radio_espacial define el alcance espacial en píxeles y rango_intensidad define
    qué diferencias de gris se aceptan en cada iteración. Es un algoritmo iterativo
    explícito; puede tardar en imágenes muy grandes.
    """
    alto, ancho = imagen.shape
    fuente = imagen.astype(np.float32)
    resultado = np.empty_like(imagen)
    radio_espacial = max(1, int(radio_espacial))
    rango_intensidad = max(1, int(rango_intensidad))
    coordenadas_y = np.arange(alto)[:, None]
    coordenadas_x = np.arange(ancho)[None, :]

    for fila in range(alto):
        for columna in range(ancho):
            centro_y = float(fila)
            centro_x = float(columna)
            centro_gris = float(fuente[fila, columna])

            for _ in range(max_iteraciones):
                y_centro = int(round(centro_y))
                x_centro = int(round(centro_x))
                y0 = max(0, y_centro - radio_espacial)
                y1 = min(alto, y_centro + radio_espacial + 1)
                x0 = max(0, x_centro - radio_espacial)
                x1 = min(ancho, x_centro + radio_espacial + 1)

                vecindad = fuente[y0:y1, x0:x1]
                ys = coordenadas_y[y0:y1]
                xs = coordenadas_x[:, x0:x1]

                distancia_espacial = (ys - centro_y) ** 2 + (xs - centro_x) ** 2
                mascara_espacial = distancia_espacial <= radio_espacial ** 2
                mascara_intensidad = np.abs(vecindad - centro_gris) <= rango_intensidad
                seleccion = mascara_espacial & mascara_intensidad

                if not np.any(seleccion):
                    break

                xs_completas = np.broadcast_to(xs, vecindad.shape)
                ys_completas = np.broadcast_to(ys, vecindad.shape)
                nuevo_x = float(np.mean(xs_completas[seleccion]))
                nuevo_y = float(np.mean(ys_completas[seleccion]))
                nuevo_gris = float(np.mean(vecindad[seleccion]))

                desplazamiento_espacial = np.sqrt(
                    (nuevo_x - centro_x) ** 2 + (nuevo_y - centro_y) ** 2
                )
                desplazamiento_gris = abs(nuevo_gris - centro_gris)

                centro_x, centro_y, centro_gris = nuevo_x, nuevo_y, nuevo_gris
                if desplazamiento_espacial < 0.1 and desplazamiento_gris < 0.5:
                    break

            resultado[fila, columna] = np.uint8(np.clip(np.rint(centro_gris), 0, 255))

    return resultado


OPERACIONES = {
    "Binarización por umbral": binarizar_umbral,
    "Erosión 3x3": erosion,
    "Dilatación 3x3": dilatacion,
    "Apertura (erosión y dilatación)": apertura,
    "Cierre (dilatación y erosión)": cierre,
    "Borde exterior": borde_exterior,
    "Borde interior": borde_interior,
    "Gradiente morfológico": gradiente_morfologico,
    "Top Hat blanco": top_hat,
    "Esqueletización binaria": esqueletizacion,
    "Mediana 3x3": mediana_3x3,
    "Mean Shift": mean_shift,
}


COLOR_FONDO = "#E8EEF5"
COLOR_PANEL = "#F8FAFC"
COLOR_AREA_IMAGEN = "#DCE4EE"
COLOR_CONTROLES = "#EEF3F8"
COLOR_PRIMARIO = "#2563EB"
COLOR_PRIMARIO_HOVER = "#1D4ED8"
COLOR_TEXTO = "#1E293B"
COLOR_SECUNDARIO = "#64748B"
COLOR_BORDE = "#CBD5E1"
COLOR_BLANCO = "#FFFFFF"


class AppMorfologia:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("PDI - Morfología y filtros en escala de gris")
        self.ventana.geometry("1400x850")
        self.ventana.minsize(1120, 700)
        self.ventana.configure(bg=COLOR_FONDO)

        self.imagen_original = None
        self.imagen_entrada = None
        self.imagen_procesada = None
        self.ruta_original = None
        self.operacion = tk.StringVar(value="Erosión 3x3")
        self.umbral = tk.IntVar(value=127)
        self.radio_espacial = tk.IntVar(value=3)
        self.rango_intensidad = tk.IntVar(value=25)

        self.crear_interfaz()

    def crear_interfaz(self):
        barra = tk.Frame(self.ventana, bg=COLOR_FONDO)
        barra.pack(side="top", fill="x", padx=18, pady=(16, 8))

        self.crear_boton(barra, "Abrir imagen", self.abrir_imagen, principal=True).pack(side="left", padx=4)
        self.crear_boton(barra, "Restaurar original", self.restaurar).pack(side="left", padx=4)
        self.crear_boton(barra, "Usar procesada como entrada", self.usar_procesada).pack(side="left", padx=4)
        self.crear_boton(barra, "Guardar procesada", self.guardar_procesada).pack(side="left", padx=4)

        centro = tk.Frame(self.ventana, bg=COLOR_FONDO)
        centro.pack(fill="both", expand=True, padx=14, pady=8)

        panel_entrada = tk.Frame(centro, bg=COLOR_PANEL, bd=1, relief="solid", highlightbackground=COLOR_BORDE)
        panel_entrada.pack(side="left", fill="both", expand=True, padx=5)
        tk.Label(
            panel_entrada, text="IMAGEN DE ENTRADA", font=("Arial", 12, "bold"),
            bg=COLOR_PANEL, fg=COLOR_TEXTO
        ).pack(pady=(13, 8))
        self.label_entrada = tk.Label(
            panel_entrada, text="Abrí una imagen para comenzar", bg=COLOR_AREA_IMAGEN,
            fg=COLOR_SECUNDARIO, font=("Arial", 11), bd=0
        )
        self.label_entrada.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        controles = tk.Frame(centro, bg=COLOR_CONTROLES, bd=1, relief="solid", padx=14, pady=14)
        controles.pack(side="left", fill="y", padx=8)
        tk.Label(
            controles, text="Procesamiento de imágenes", font=("Arial", 13, "bold"),
            bg=COLOR_CONTROLES, fg=COLOR_TEXTO, justify="center"
        ).pack(pady=(3, 8))
        tk.Label(
            controles, text="Elemento estructurante: cruz 3x3", font=("Arial", 9),
            bg=COLOR_CONTROLES, fg=COLOR_SECUNDARIO
        ).pack(pady=(0, 12))
        tk.Label(
            controles, text="Operación", font=("Arial", 10, "bold"),
            bg=COLOR_CONTROLES, fg=COLOR_TEXTO
        ).pack(anchor="w", pady=(0, 5))

        self.lista_operaciones = ttk.Combobox(
            controles, textvariable=self.operacion, values=list(OPERACIONES.keys()),
            state="readonly", width=31
        )
        self.lista_operaciones.pack(fill="x", pady=(0, 10))
        self.lista_operaciones.bind("<<ComboboxSelected>>", self.actualizar_parametros)

        self.marco_umbral = tk.Frame(controles, bg=COLOR_CONTROLES)
        tk.Label(
            self.marco_umbral, text="Umbral (0–255)",
            font=("Arial", 9, "bold"), bg=COLOR_CONTROLES, fg=COLOR_TEXTO
        ).pack(anchor="w", pady=(0, 3))
        ttk.Spinbox(
            self.marco_umbral, from_=0, to=255, increment=1,
            textvariable=self.umbral, width=8
        ).pack(anchor="w", pady=(0, 3))
        tk.Label(
            self.marco_umbral, text="Los píxeles mayores al umbral se vuelven blancos; los demás, negros.",
            font=("Arial", 8), bg=COLOR_CONTROLES, fg=COLOR_SECUNDARIO,
            wraplength=260, justify="left"
        ).pack(anchor="w", pady=(0, 8))

        self.marco_mean_shift = tk.Frame(controles, bg=COLOR_CONTROLES)
        tk.Label(
            self.marco_mean_shift, text="Parámetros de Mean Shift",
            font=("Arial", 9, "bold"), bg=COLOR_CONTROLES, fg=COLOR_TEXTO
        ).pack(anchor="w", pady=(0, 3))
        fila_radio = tk.Frame(self.marco_mean_shift, bg=COLOR_CONTROLES)
        fila_radio.pack(fill="x", pady=(0, 4))
        tk.Label(fila_radio, text="Radio espacial", font=("Arial", 8),
                 bg=COLOR_CONTROLES, fg=COLOR_TEXTO).pack(side="left")
        ttk.Spinbox(fila_radio, from_=1, to=10, increment=1,
                    textvariable=self.radio_espacial, width=5).pack(side="right")
        fila_rango = tk.Frame(self.marco_mean_shift, bg=COLOR_CONTROLES)
        fila_rango.pack(fill="x", pady=(0, 10))
        tk.Label(fila_rango, text="Rango de gris", font=("Arial", 8),
                 bg=COLOR_CONTROLES, fg=COLOR_TEXTO).pack(side="left")
        ttk.Spinbox(fila_rango, from_=1, to=255, increment=1,
                    textvariable=self.rango_intensidad, width=5).pack(side="right")

        boton_aplicar = tk.Button(
            controles, text="Aplicar operación", command=self.aplicar_operacion,
            bg=COLOR_PRIMARIO, fg=COLOR_BLANCO,
            activebackground=COLOR_PRIMARIO_HOVER, activeforeground=COLOR_BLANCO,
            font=("Arial", 10, "bold"), padx=12, pady=9, relief="flat", bd=0,
            cursor="hand2"
        )
        boton_aplicar.pack(fill="x", pady=(2, 12))

        tk.Label(
            controles,
            text="Para aplicar otra operación al resultado, pulsá «Usar procesada como entrada».",
            justify="left", font=("Arial", 8), bg=COLOR_CONTROLES,
            fg=COLOR_SECUNDARIO, wraplength=270
        ).pack(anchor="w", pady=(0, 6))

        panel_salida = tk.Frame(centro, bg=COLOR_PANEL, bd=1, relief="solid", highlightbackground=COLOR_BORDE)
        panel_salida.pack(side="left", fill="both", expand=True, padx=5)
        self.titulo_salida = tk.Label(
            panel_salida, text="IMAGEN PROCESADA", font=("Arial", 12, "bold"),
            bg=COLOR_PANEL, fg=COLOR_TEXTO
        )
        self.titulo_salida.pack(pady=(13, 8))
        self.label_salida = tk.Label(
            panel_salida, text="El resultado aparecerá aquí", bg=COLOR_AREA_IMAGEN,
            fg=COLOR_SECUNDARIO, font=("Arial", 11), bd=0
        )
        self.label_salida.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.estado = tk.Label(
            self.ventana, text="Listo. Abrí una imagen para comenzar.",
            font=("Arial", 10), bg=COLOR_FONDO, fg=COLOR_SECUNDARIO, anchor="w"
        )
        self.estado.pack(side="bottom", fill="x", padx=20, pady=(2, 10))
        self.actualizar_parametros()

    def actualizar_parametros(self, evento=None):
        self.marco_umbral.pack_forget()
        self.marco_mean_shift.pack_forget()

        nombre = self.operacion.get()
        if nombre in ("Binarización por umbral", "Esqueletización binaria"):
            self.marco_umbral.pack(fill="x", pady=(0, 6))
        elif nombre == "Mean Shift":
            self.marco_mean_shift.pack(fill="x", pady=(0, 6))

    def crear_boton(self, padre, texto, comando, principal=False):
        if principal:
            fondo, texto_color, activo = COLOR_PRIMARIO, COLOR_BLANCO, COLOR_PRIMARIO_HOVER
        else:
            fondo, texto_color, activo = COLOR_BLANCO, COLOR_TEXTO, COLOR_AREA_IMAGEN

        return tk.Button(
            padre, text=texto, command=comando, bg=fondo, fg=texto_color,
            activebackground=activo, activeforeground=texto_color,
            font=("Arial", 10, "bold"), padx=12, pady=8,
            relief="flat" if principal else "solid", bd=0 if principal else 1,
            cursor="hand2"
        )

    def abrir_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp")]
        )
        if not ruta:
            return

        try:
            with Image.open(ruta) as imagen_pil:
                imagen = convertir_escala_grises(imagen_pil)

            self.ruta_original = ruta
            self.imagen_original = imagen.copy()
            self.imagen_entrada = imagen.copy()
            self.imagen_procesada = None
            self.mostrar_imagen(self.label_entrada, self.imagen_entrada)
            self.limpiar_salida("Elegí una operación y presioná Aplicar")
            self.estado.config(text="Imagen cargada y convertida a escala de grises.")
        except Exception as error:
            messagebox.showerror("Error", f"No se pudo abrir la imagen.\n\n{error}")

    def aplicar_operacion(self):
        if self.imagen_entrada is None:
            messagebox.showwarning("Atención", "Primero abrí una imagen.")
            return

        nombre = self.operacion.get()
        funcion = OPERACIONES.get(nombre)
        if funcion is None:
            messagebox.showwarning("Atención", "Seleccioná una operación válida.")
            return

        try:
            umbral = int(self.umbral.get())
            if not 0 <= umbral <= 255:
                raise ValueError("El umbral debe estar entre 0 y 255.")
            radio_espacial = int(self.radio_espacial.get())
            rango_intensidad = int(self.rango_intensidad.get())

            if nombre == "Binarización por umbral":
                resultado = binarizar_umbral(self.imagen_entrada, umbral)
            elif nombre == "Esqueletización binaria":
                resultado = esqueletizacion(self.imagen_entrada, umbral)
            elif nombre == "Mean Shift":
                resultado = mean_shift(
                    self.imagen_entrada,
                    radio_espacial=radio_espacial,
                    rango_intensidad=rango_intensidad
                )
            else:
                resultado = funcion(self.imagen_entrada)

            self.imagen_procesada = resultado
            self.mostrar_imagen(self.label_salida, self.imagen_procesada)
            self.titulo_salida.config(text=f"IMAGEN PROCESADA - {nombre.upper()}")
            if nombre in ("Binarización por umbral", "Esqueletización binaria"):
                self.estado.config(text=f"{nombre} aplicada con umbral {umbral}.")
            elif nombre == "Mean Shift":
                self.estado.config(
                    text=f"Mean Shift aplicado (radio espacial {radio_espacial}, rango de gris {rango_intensidad})."
                )
            else:
                self.estado.config(text=f"Operación aplicada: {nombre}.")
        except (ValueError, tk.TclError) as error:
            messagebox.showwarning("Parámetros inválidos", str(error))
        except Exception as error:
            messagebox.showerror("Error", f"No se pudo aplicar la operación.\n\n{error}")

    def usar_procesada(self):
        if self.imagen_procesada is None:
            messagebox.showwarning("Atención", "Todavía no hay una imagen procesada.")
            return

        self.imagen_entrada = self.imagen_procesada.copy()
        self.mostrar_imagen(self.label_entrada, self.imagen_entrada)
        self.estado.config(text="La imagen procesada ahora es la entrada para la siguiente operación.")

    def restaurar(self):
        if self.imagen_original is None:
            messagebox.showwarning("Atención", "Primero abrí una imagen.")
            return

        self.imagen_entrada = self.imagen_original.copy()
        self.imagen_procesada = None
        self.mostrar_imagen(self.label_entrada, self.imagen_entrada)
        self.limpiar_salida("El resultado aparecerá aquí")
        self.estado.config(text="Imagen original restaurada.")

    def guardar_procesada(self):
        if self.imagen_procesada is None:
            messagebox.showwarning("Atención", "No hay una imagen procesada para guardar.")
            return

        ruta = filedialog.asksaveasfilename(
            title="Guardar imagen procesada", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"),
                       ("BMP", "*.bmp"), ("TIFF", "*.tif *.tiff")]
        )
        if not ruta:
            return

        try:
            Image.fromarray(self.imagen_procesada).save(ruta)
            self.estado.config(text=f"Imagen procesada guardada: {ruta}")
        except Exception as error:
            messagebox.showerror("Error", f"No se pudo guardar la imagen.\n\n{error}")

    def limpiar_salida(self, texto):
        self.label_salida.config(image="", text=texto)
        self.label_salida.foto = None
        self.titulo_salida.config(text="IMAGEN PROCESADA")

    def mostrar_imagen(self, label, array_imagen):
        imagen_pil = Image.fromarray(array_imagen).convert("RGB")
        imagen_pil.thumbnail((520, 540))
        foto = ImageTk.PhotoImage(imagen_pil)
        label.config(image=foto, text="")
        label.foto = foto


if __name__ == "__main__":
    ventana = tk.Tk()
    app = AppMorfologia(ventana)
    ventana.mainloop()
