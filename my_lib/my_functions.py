import numpy as np


# ============================================================
# ESCALA DE GRISES
# ============================================================

def escala_grises(imagen):
    """
    Convierte una imagen RGB a escala de grises.

    Cada píxel [R, G, B] se reemplaza por el promedio
    de sus tres componentes.
    """

    gris = imagen.mean(axis=2)

    resultado = np.stack(
        [gris, gris, gris],
        axis=2
    )

    return resultado


# ============================================================
# SOLO UN CANAL
# ============================================================

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
# OPERACIONES ARITMÉTICAS
# ============================================================

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


# ============================================================
# RGB -> YIQ
# ============================================================

def rgbAyiq(imagen, luminancia=1.0, cromaticidad=1.0):

    matriz_rgb_yiq = np.array(
        [
            [0.299, 0.587, 0.114],
            [0.595716, -0.274453, -0.321263],
            [0.211456, -0.522591, 0.311135]
        ]
    )

    pixeles_rgb = imagen.reshape(-1, 3)

    pixeles_yiq = pixeles_rgb @ matriz_rgb_yiq.T

    imagen_yiq = pixeles_yiq.reshape(
        imagen.shape
    )

    Y = imagen_yiq[:, :, 0]
    I = imagen_yiq[:, :, 1]
    Q = imagen_yiq[:, :, 2]

    Y_nuevo = luminancia * Y

    I_nuevo = cromaticidad * I
    Q_nuevo = cromaticidad * Q

    Y_nuevo = np.clip(
        Y_nuevo,
        0,
        1
    )

    I_nuevo = np.clip(
        I_nuevo,
        -0.5957,
        0.5957
    )

    Q_nuevo = np.clip(
        Q_nuevo,
        -0.5226,
        0.5226
    )

    yiq_final = np.stack(
        [Y_nuevo, I_nuevo, Q_nuevo],
        axis=2
    )

    return yiq_final


# ============================================================
# YIQ -> RGB
# ============================================================

def yiqArgb(imagen):

    matriz_yiq_rgb = np.array(
        [
            [1.0,  0.9663,  0.6210],
            [1.0, -0.2721, -0.6474],
            [1.0, -1.1070,  1.7046]
        ]
    )

    pixeles_yiq = imagen.reshape(-1, 3)

    pixeles_rgb = pixeles_yiq @ matriz_yiq_rgb.T

    imagen_rgb = pixeles_rgb.reshape(
        imagen.shape
    )

    imagen_rgb = np.clip(
        imagen_rgb,
        0,
        1
    )

    return imagen_rgb



# ============================================================
# OPERACIONES DE LUMINANCIA
# ============================================================

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


# ============================================================
# CONVOLUCIÓN
# ============================================================

def validar_kernel(kernel_):
    
    kernel = np.asarray(kernel_, dtype=float)
    
    if kernel.ndim != 2:
        print("El kernel deben ser una matriz 2D")
        return None

    return kernel

def convolution(image, kernel = np.ones((1,1))):
    convolved = np.zeros((np.array(image.shape)-np.array(kernel.shape)+1))
    for x in range(convolved.shape[0]):
        for y in range(convolved.shape[1]):
            convolved[x,y] = (image[x:x+kernel.shape[0],y:y+kernel.shape[1]]*kernel).sum()
    return convolved



#----- Convolución con copiado/duplicado ------
def convolucion(imagen, kernel):
    imagen = np.asarray(imagen, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    # Rotar el kernel 180 grados
    kernel = np.rot90(kernel, 2)

    # Calcular el centro del kernel
    centro_f = kernel.shape[0] // 2
    centro_c = kernel.shape[1] // 2

    # Replicar los bordes de la imagen
    imagen_pad = np.pad(
        imagen,
        (
            (centro_f, kernel.shape[0] - 1 - centro_f),
            (centro_c, kernel.shape[1] - 1 - centro_c)
        ),
        mode="edge"
    )

    resultado = np.zeros(imagen.shape)

    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):

            region = imagen_pad[
                i:i + kernel.shape[0],
                j:j + kernel.shape[1]
            ]

            resultado[i, j] = (region * kernel).sum()

    return resultado


def convolucion_manual(imagen, kernel):
    # El kernel tiene una celda central porque su tamaño
    # es impar: 3x3, 5x5, 7x7, etc.

    centro = kernel.shape[0] // 2

    # --------------------------------------------------------
    # El kernel se rota 180°.
    # --------------------------------------------------------

    kernel = np.rot90(
        kernel,
        2   
    )

    resultado = np.zeros(
        imagen.shape
    )

    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):

            suma = 0

            for ki in range(kernel.shape[0]):

                for kj in range(kernel.shape[1]):

                    # ----------------------------------------
                    # Calculamos qué píxel de la imagen
                    # corresponde a esta posición del kernel.
                    # ----------------------------------------

                    fila = i + ki - centro
                    columna = j + kj - centro


                    # ----------------------------------------
                    # Si el kernel sale de la imagen,
                    # repetimos el píxel del borde.
                    # ----------------------------------------

                    if fila < 0:
                        fila = 0

                    if fila >= imagen.shape[0]:
                        fila = imagen.shape[0] - 1

                    if columna < 0:
                        columna = 0

                    if columna >= imagen.shape[1]:
                        columna = imagen.shape[1] - 1

                    # ----------------------------------------
                    # Tomamos el píxel de la imagen.
                    # ----------------------------------------

                    pixel = imagen[
                        fila,
                        columna
                    ]

                    # ----------------------------------------
                    # Tomamos el valor correspondiente
                    # del kernel.
                    # ----------------------------------------

                    valor_kernel = kernel[
                        ki,
                        kj
                    ]

                    # ----------------------------------------
                    # Multiplicamos y acumulamos.
                    # ----------------------------------------

                    suma = suma + pixel * valor_kernel

            # ------------------------------------------------
            # La suma de productos se convierte en el nuevo
            # valor del píxel (i,j).
            # ------------------------------------------------

            resultado[i, j] = suma

    return resultado



#----------------
# KERNELS
#----------------
def kernel_plano(tamanio):

    valor = 1 / (tamanio * tamanio)

    kernel = np.ones(
        (tamanio, tamanio)
    )

    kernel = kernel * valor

    return kernel


# ----- FILTRO PASA BAJOS (difuminado, suavizado ) ------
def kernel_bartlett(tamanio):

    centro = tamanio // 2

    #print(f"Centro: {centro}")

    secuencia = []

    for i in range(tamanio):

        if i <= centro:

            valor = i + 1

        else:

            valor = tamanio - i

        secuencia.append(valor)

    #print(f"secuencia: {secuencia}")


    kernel = []

    for i in range(tamanio):

        fila = []

        for j in range(tamanio):

            valor = (
                secuencia[i]
                * secuencia[j]
            )

            fila.append(valor)

        kernel.append(fila)

    kernel = np.array(
        kernel,
        dtype=float
    )

    #print(f"pre-kernel: {kernel}")

    #print(f"kernel-sum: {kernel.sum()}")
    kernel = kernel / kernel.sum()

    print(f"kernel final: {kernel}")

    return kernel



def kernel_gaussiano(tamanio):

    if tamanio ==3:
        secuencia=[1,2,1]
    
    elif tamanio == 5:

        secuencia = [
            1, 4, 6, 4, 1
        ]

    elif tamanio == 7:

        secuencia = [
            1, 6, 15, 20, 15, 6, 1
        ]

    else:

        return None


    kernel = []

    for i in range(tamanio):

        fila = []

        for j in range(tamanio):

            valor = (
                secuencia[i]
                * secuencia[j]
            )

            fila.append(valor)

        kernel.append(fila)


    kernel = np.array(
        kernel,
        dtype=float
    )


    kernel = kernel / kernel.sum()

    return kernel



##### FILTRO PASA ALTOS (Detectores de bordes)

# Con sus 4 vecinos por fila y columna
def kernel_laplaciano_v4():

    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ],
        dtype=float
    )

    return kernel


# Con sus 8 vecinos por fila y columna
def kernel_laplaciano_v8():

    kernel = np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ],
        dtype=float
    )

    return kernel


#---------- kernel Sobel (Filtros direccionales) -----------

def kernel_sobel_este():

    kernel = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ],
        dtype=float
    )

    return kernel


def kernel_sobel_sur():
    
    kernel = np.array(
        [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ],
        dtype=float
    )

    return kernel



def kernel_sobel_oeste():

    kernel = np.array(
        [
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ],
        dtype=float
    )

    return kernel


def kernel_sobel_norte():

    kernel = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],
        dtype=float
    )

    return kernel


def kernel_sobel_noreste():

    kernel = np.array(
        [
            [2, 1, 0],
            [1, 0, -1],
            [0, -1, -2]
        ],
        dtype=float
    )

    return kernel



def kernel_sobel_sureste():

    kernel = np.array(
        [
            [0, -1, -2],
            [1, 0, -1],
            [2, 1, 0]
        ],
        dtype=float
    )

    return kernel


def kernel_sobel_suroeste():

    kernel = np.array(
        [
            [-2, -1, 0],
            [-1, 0, 1],
            [0, 1, 2]
        ],
        dtype=float
    )

    return kernel


def kernel_sobel_noroeste():

    kernel = np.array(
        [
            [0, 1, 2],
            [-1, 0, 1],
            [-2, -1, 0]
        ],
        dtype=float
    )

    return kernel




#---------- COMBINACIÓN (Filtro de mejora)  -----------
def combinar_kernels(kernel1, kernel2, porcentaje):

    kernel1 = np.asarray(kernel1, dtype=float)
    kernel2 = np.asarray(kernel2, dtype=float)

    if kernel1.ndim != 2 or kernel2.ndim != 2:
        print("Los kernels deben ser matrices 2D")
        return None

    if kernel1.shape != kernel2.shape:
        print("Los kernels deben tener el mismo tamaño")
        return None

    kernel2 = kernel2 * porcentaje

    kernel_resultado = kernel1 + kernel2
    
    return kernel_resultado





#---------- PASA ALTOS DE FRECUENCIA DE CORTE 0,2 Y 0,4  -----------

def kernel_pasaaltos(frecuencia_corte, tamanio=15):

    if frecuencia_corte <= 0 or frecuencia_corte >= 0.5:
        print("La frecuencia debe estar entre 0 y 0.5")
        return None

   #Genera una lista de las frecuencias que se van a representar
    frecuencias = np.fft.fftfreq(tamanio)

   #Transforma esa lista en dos matrices que representan las frecuencias de cada dirección
    fx, fy = np.meshgrid(frecuencias, frecuencias)
  
   #Calcular la distancia desde la frecuencia cero hasta cada posición del plano de frecuencias
    distancia = np.sqrt(fx**2 + fy**2)

    # Eliminar las frecuencias inferiores al corte
    mascara = (distancia >= frecuencia_corte).astype(float)

    # Obtener el kernel espacial
    kernel = np.fft.fftshift(
        np.fft.ifft2(mascara).real
    )

    return kernel



#---------- PASA BANDA -----------

def pasabanda_diff_gauss(imagen,kernel1,kernel2):

  
  if kernel1 is None:
        kernel1 = kernel_gaussiano(3)

  if kernel2 is None:
        kernel2 = kernel_gaussiano(5)


  kernel1 = np.asarray(kernel1, dtype=float)
  kernel2 = np.asarray(kernel2, dtype=float)

  suma1 = kernel1.sum()
  suma2 = kernel2.sum()

# Igualar las ganancias de ambas Gaussianas
  kernel1 = kernel1 / suma1
  kernel2 = kernel2 / suma2

    # Tamaño común
  alto = max(kernel1.shape[0], kernel2.shape[0])
  ancho = max(kernel1.shape[1], kernel2.shape[1])

# Centrar los kernels y completar con ceros
  kernel1 = np.pad(
        kernel1,
        (
            ((alto - kernel1.shape[0]) // 2,
             (alto - kernel1.shape[0]) // 2),
            ((ancho - kernel1.shape[1]) // 2,
             (ancho - kernel1.shape[1]) // 2)
        ),
        mode="constant"
    )

  kernel2 = np.pad(
        kernel2,
        (
            ((alto - kernel2.shape[0]) // 2,
             (alto - kernel2.shape[0]) // 2),
            ((ancho - kernel2.shape[1]) // 2,
             (ancho - kernel2.shape[1]) // 2)
        ),
        mode="constant"
    )

  kernel_resultado= kernel1-kernel2

  resultado= convolucion(imagen, kernel_resultado)

  return resultado






