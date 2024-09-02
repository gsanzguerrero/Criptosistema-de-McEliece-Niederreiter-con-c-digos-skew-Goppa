import galois
import numpy as np
import random
import hashlib
import sys
from progress.bar import Bar
import time


def reverse_array(arr):
    """
    Invierte el orden de los elementos de una lista.

    Parámetros:
    arr (list): La lista de elementos a invertir.

    Retorna:
    list: Una nueva lista con los elementos en orden inverso.
    """
    return arr[::-1]


def multiplicacion_nc(f, g, h, field):
    """
    Realiza la multiplicación no conmutativa de dos polinomios f y g en un cuerpo dado.

    Parámetros:
    f (galois.Poly): El primer polinomio.
    g (galois.Poly): El segundo polinomio.
    h (int): Parámetro del criptosistema.
    field (galois.Field): El cuerpo sobre el cual se realiza la operación.

    Retorna:
    galois.Poly: El polinomio resultado de la multiplicación no conmutativa.
    """
    p = field.characteristic  # Característica del cuerpo

    coef_f2 = reverse_array(f.coefficients())  # Coeficientes de f en orden inverso
    f2 = [coeficiente for coeficiente in coef_f2]

    coef_g2 = reverse_array(g.coefficients())  # Coeficientes de g en orden inverso
    g2 = [coeficiente for coeficiente in coef_g2]

    tam = len(f2) + len(g2) - 1  # Tamaño del polinomio resultado
    resultado = [field(0)] * tam  # Inicialización del resultado con ceros

    for i in range(len(f2)):
        exponente = p ** (h * i)  # Calcula el exponente para la operación no conmutativa
        for j in range(len(g2)):
            resultado[i + j] = resultado[i + j] + (f2[i] * (g2[j] ** exponente)) #Calcula los coeficientes del resultado (de menos a mayor)

    resultado = galois.Poly(reverse_array(resultado), field=field)  # Invierte el orden de los coeficientes (se guardan de mayor a menos en un polinomio)

    return resultado


def division_nc(f, g, h, field):
    """
    Realiza la división no conmutativa de dos polinomios f y g en un cuerpo dado.

    Parámetros:
    f (galois.Poly): El dividendo.
    g (galois.Poly): El divisor.
    h (int): Parámetro del criptosistema.
    field (galois.Field): El cuerpo sobre el cual se realiza la operación.

    Retorna:
    tuple: El cociente y el residuo de la división no conmutativa.
    """
    p = field.characteristic  # Característica del cuerpo
    
    if g == galois.Poly([0], field=field): #Si el divisor es 0 salimos de la operación
        return None

    # Parámetros iniciales
    c = galois.Poly([0], field=field)  # Cociente inicializado en cero
    r = f  # Residuo inicializado como el dividendo
    coefs_g = [coeficiente for coeficiente in g.coefficients()]  # Coeficientes de g (de mayor a menor)
    b = (coefs_g[0] ** (-1))  # Inverso del primer coeficiente de g

    while g.degree <= r.degree and r != galois.Poly([0], field=field):
        coefs_r = [coeficiente for coeficiente in r.coefficients()]  # Coeficientes de r (de mayor a menor)
        expB = p ** (h * (r.degree - g.degree))  # Exponente para la operación
        a = coefs_r[0] * (pow(b, expB))  # Coeficiente principal para el monomio

        # Construcción del monomio correspondiente
        vector_monomio = [a]
        for i in range(r.degree - g.degree):
            vector_monomio.append(0)

        monomio = galois.Poly(vector_monomio, field=field)  # Creación del monomio
        c = c + monomio  # Actualización del cociente
        r = r - multiplicacion_nc(monomio, g, h, field)  # Actualización del residuo

    return c, r


def PCP(f, g, h, field):
    """
    Calcula el máximo común divisor a la derecha de dos polinomios f y g utilizando el algoritmo de Euclides extendido 
    no conmutativo para quedarnos con el cpeficiente de Bezout correspondiente al primer polinomio.

    Parámetros:
    f (galois.Poly): El primer polinomio.
    g (galois.Poly): El segundo polinomio.
    h (int): Parámetro del criptosistema.
    field (galois.Field): El cuerpo sobre el cual se realiza la operación.

    Retorna:
    galois.Poly: El máximo común divisor de f y g, o cero si no existe.
    """
    if g == field(0): #Si el segundo polinomio es 0 salimos de la función
        return None

    # Inicialización de variables
    f0 = f
    f1 = g

    u0 = galois.Poly.One(field)  # Polinomio unidad
    u1 = galois.Poly.Zero(field)  # Polinomio cero

    # Primer paso del algoritmo
    cociente, resto = division_nc(f0, f1, h, field)
    f0 = f1
    f1 = resto
    u = galois.Poly.One(field)
    u0 = u1
    u1 = u

    # Iteración hasta que el residuo sea cero
    while f1 != galois.Poly.Zero(field):
        cociente, resto = division_nc(f0, f1, h, field)
        f0 = f1
        f1 = resto
        u = u0 - multiplicacion_nc(cociente, u1, h, field)
        u0 = u1
        u1 = u

    coefs_f0 = [coeficiente for coeficiente in f0.coefficients()]  # Coeficientes de f0

    # Si el grado es uno, retorna el inverso del primer coeficiente multiplicado por u0, si no, devuelve 0
    if len(coefs_f0) == 1:
        return (coefs_f0[0] ** (-1) * u0)
    else:
        return galois.Poly.Zero(field)


def MCM(f, g, h, field):
    """
    Calcula el mínimo común múltiplo de dos polinomios f y g utilizando un algoritmo no conmutativo.

    Parámetros:
    f (galois.Poly): El primer polinomio.
    g (galois.Poly): El segundo polinomio.
    h (int): Parámetro del criptosistema.
    field (galois.Field): El cuerpo sobre el cual se realiza la operación.

    Retorna:
    galois.Poly: El mínimo común múltiplo de f y g.
    """
    if g == galois.Poly([0], field=field): #Si el segundo polinomio es 0 devolvemos el valor 0
        return galois.Poly([0], field=field)

    # Inicialización de variables
    f0 = f
    f1 = g

    u0 = galois.Poly([1], field=field)  # Polinomio unidad
    u1 = galois.Poly([0], field=field)  # Polinomio cero

    # Primer paso del algoritmo
    cociente, resto = division_nc(f0, f1, h, field)
    f0 = f1
    f1 = resto
    u = galois.Poly([1], field=field)
    u0 = u1
    u1 = u

    # Iteración hasta que el residuo sea cero
    while f1 != galois.Poly([0], field=field):
        cociente, resto = division_nc(f0, f1, h, field)
        f0 = f1
        f1 = resto
        u = u0 - multiplicacion_nc(cociente, u1, h, field)
        u0 = u1
        u1 = u

    # Multiplicación del resultado por el polinomio original
    u = multiplicacion_nc(u1, f, h, field)
    coefs_u = [coeficiente for coeficiente in u.coefficients()]  # Coeficientes de u

    return (coefs_u[0] ** (-1) * u)


def emb(elto_F, matriz_B2, L):
    """
    Embebe un elemento de un cuerpo F en un cuerpo L utilizando una matriz de base.

    Parámetros:
    elto_F (galois.Poly): El elemento del cuerpo F.
    matriz_B2 (np.array): La matriz de base utilizada para la transformación.
    L (galois.Field): El cuerpo en el que se embebe el elemento.

    Retorna:
    galois.Poly: El elemento embebido en el cuerpo L.
    """
    # Pasamos elto_F a coordenadas ascendentes
    vector = reverse_array(list(F.vector(elto_F)))

    # Extender el vector con ceros para el tamaño adecuado
    vector += [galois.GF(int(2))(0)] * (d * (m - 1))

    # Convertir la lista a un array del cuerpo Galois
    vector_GF2 = galois.GF(int(2))(vector)

    # Multiplicar el vector extendido por la matriz B2
    vector_L = np.dot(vector_GF2, matriz_B2)

    # Convertir el vector al cuerpo L
    elto_L = L.Vector(reverse_array(vector_L))

    return elto_L


def coord(elto_L, matriz_B2_inversa, m, F):
    """
    Calcula las coordenadas de un elemento de un cuerpo L en un cuerpo F utilizando una matriz inversa.

    Parámetros:
    elto_L (galois.Poly): El elemento en el cuerpo L.
    matriz_B2_inversa (np.array): La matriz inversa utilizada para la transformación.
    m (int): Un parámetro relacionado con la extensión del cuerpo.
    F (galois.Field): El cuerpo objetivo para las coordenadas.

    Retorna:
    list: Una lista de elementos de F que corresponden a las coordenadas.
    """
    # Convertir el elemento L a un vector y luego invertir el orden
    vector_elto_L = reverse_array(L.vector(elto_L))

    # Multiplicar el vector por la matriz inversa
    vector_eltos_F = np.dot(vector_elto_L, matriz_B2_inversa)

    # Dividir el vector en subvectores según el tamaño m
    subvectores_F = np.split(vector_eltos_F, m)

    # Convertir los subvectores en elementos de F
    eltos_F = [F.Vector(reverse_array(list(subvector))) for subvector in subvectores_F]

    return eltos_F


def es_normal(elemento, p, delta, mu, cuerpo):
    """
    Verifica si un elemento es normal, es decir, si genera una base normal de la extensión K sobre L.

    Parámetros:
    elemento (galois.Poly): El elemento a verificar.
    p (int): La característica del cuerpo.
    delta (int): Un parámetro relacionado con la extensión.
    mu (int): El grado de la extensión.
    cuerpo (galois.Field): El cuerpo sobre el cual se realiza la verificación.

    Retorna:
    bool: True si el elemento genera una base normal, False en caso contrario.
    """
    # Crear el primer polinomio 1 - x^mu
    terminos_pol_1 = [1]
    for i in range(mu - 1):
        terminos_pol_1.append(0)
    terminos_pol_1.append(-1)

    polinomio_1 = galois.Poly(terminos_pol_1, field=cuerpo)

    # Crear el segundo polinomio con los términos elemento^(p^(i*delta))
    terms_2 = [elemento ** (p ** (i * delta)) for i in range(mu)]
    polinomio2 = galois.Poly(terms_2, field=cuerpo)

    # Calcular el máximo común divisor (MCD) de los dos polinomios
    gcd = galois.gcd(polinomio_1, polinomio2)

    # Verificar si el MCD es 1
    return gcd == galois.Poly.One(cuerpo)



# Verificar que se pasen dos argumentos (los nombres de los archivos)
if len(sys.argv) != 3:
    print()
    print("ERROR: Debe ejecutar el archivo con el siguiente comando: python.exe encapsular.py <nombre_archivo_clave> <nombre_archivo_criptograma>")
    sys.exit(1)

# Obtener el de los archivos desde los argumentos
archivo_clave = "../"+sys.argv[1]
#Obtener el criptograma del archivo
archivo_criptograma = "../"+sys.argv[2]

inicio = time.time() #Iniciamos el tiempo de ejecución del programa

# Cargar los datos desde el archivo .npz
data = np.load(archivo_clave, allow_pickle=True)

print()
print("Desencapsulamiento de un criptograma recibido usando la clave previamente generada")

print("Obtención de los parámetros del archivo...")
# Obtenemos los parámetros del archivo y los convertimos al tipo correspondiente
n = int(data['n'])
t = int(data['t'])
p = int(data['p'])
d = int(data['d'])
k = int(data['k'])
m = int(data['m'])
delta = int(data['delta'])
h = int(data['h'])
mu = int(data['mu'])

#Construcción del cuerpo L

#Obtenemos el polinomio irreducible de L del archivo
coefs_def_L = data['polinomio_def_L']
polinomio_def_L = galois.Poly(coefs_def_L,field=galois.GF(2))

#Construimos el cuerpo L y obtenemos algunos elementos del mismo
L = galois.GF(pow(p,d*m), irreducible_poly=polinomio_def_L, repr='poly')
primitivo_L=L.primitive_element
num_aleatorio=random.randint(0, p**(d*m))
L_elemento = primitivo_L**num_aleatorio
print()
print(f"El cuerpo finito de Galois L es GF(p^(d*m)): {L.properties}")
print(f"Elemento random de L: {L_elemento} ")

#Construcción del cuerpo F

#Obtenemos el polinomio irreducible de F del archivo
coefs_def_F = data['polinomio_def_F']
polinomio_def_F = galois.Poly(coefs_def_F,field=galois.GF(2))

#Construimos el cuerpo F y obtenemos algunos elementos del mismo
F = galois.GF(pow(p,d), irreducible_poly=polinomio_def_F, repr='poly')
F_elemento = random.choice(F.elements)
print()
print(f"El cuerpo finito de Galois F es GF(p^d): {F.properties}")
print(f"Elemento random de F: {F_elemento} ")

#Obtención del resto de elementos de la clave del archivo
r= int(data['r'])

matriz_B2 = data['matriz_B2']
matriz_B2=galois.GF(2)(matriz_B2)
matriz_B2_inv = np.linalg.inv(matriz_B2) #Cálculo de la matriz inversa de B2

polinomios_nc_paridad = data['polinomios_nc_paridad']
elementos_aleatorios_eta = data['elementos_aleatorios_eta']
puntos_evaluacion = data['puntos_evaluacion']
g_coefficients = data['g']
g = galois.Poly(g_coefficients, field=L)

#Obtención de la clave pública del archivo
H_pub = data['H_pub']
H_pub=F(H_pub)
print("Clave obtenida del archivo")


#Obtención del criptograma del archivo
criptograma = F(np.load(archivo_criptograma, allow_pickle=True))
print()
print("Criptograma recibido desde el archivo.")

print()
print("PARÁMETROS:")
print("n: ",n, "t: ",t," p: ",p," d: ",d," k: ",k," m: ",m," delta: ",delta," h: ",h, ", mu: ", mu, " r: ", r)
print()


print("Comienzo del desencapsulado del criptograma")

print("Cálculo del vector y: ")

#Cálculo del vector y
y=[F(0)] * n #Creamos un vector y con n posiciones nulas

# OBTENER POSICIONES DE LOS PIVOTES de H_pub
posiciones = []

# Iterar sobre cada fila de la matriz
for i in range(len(H_pub)):
    # Inicializamos la posición como -1 (por si no hay elementos no nulos)
    posicion = -1
    # Recorremos cada elemento de la fila
    for j in range(len(H_pub[i])):
        if H_pub[i][j] != 0: #Cuando se diferente de 0 (será el pivote)
            posicion = j
            break  # Salimos del bucle cuando encontramos el primer no nulo
    
    y[posicion]=criptograma[i] #Actualizamos la posición donde se encuentra el pivote en el vector y con el criptograma correspondiente de la fila de H_pub

print("Polinomio y calculado.")

#ALGORITMO DEC
print()
print("Comienzo del algoritmo DEC(): ")

print()
print("Calculando el polinomio síndrome...")

#Calculamos el síndrome (s)
barra_syndrome = Bar('Polinomio síndrome:', max=n) #Este elemento de la libreria progress genera una barra en la salida del programa que va mostrando el progreso del bucle
s=galois.Poly(L(0), field=L) #Definimos el polinomio sindrome como s

for i in range(n): #Recorremos todo el vector y
    elto_L = emb(y[i], matriz_B2, L) # Calcular el término
    resultado = multiplicacion_nc(polinomios_nc_paridad[i], galois.Poly(elementos_aleatorios_eta[i] * elto_L, field=L), h, L) # Realizar la multiplicación
    s += resultado # Acumulación en `s`
    barra_syndrome.next() #Actualizamos la barra de progeso

print()
print("Grado del polinomio síndrome: ", s.degree)

print()
print("Cálculo de polinomios localizador y evaluador de errores")

r0 = g  # r0 se inicializa con el polinomio g de la clave
r1 = s  # r1 se inicializa con el polinomio s
v0 = galois.Poly(L(0), field=L)  # Polinomio inicial v0, que es el polinomio cero en el cuerpo L
v1 = galois.Poly(L(1), field=L)  # Polinomio inicial v1, que es el polinomio uno en el cuerpo L

# Iteración usando el algoritmo de Euclides extendido hasta que el grado de r1 sea menor que t
while r1.degree >= t:
    
    c, r = division_nc(r0, r1, h, L) # División no conmutativa de r0 por r1 en el cuerpo L con un parámetro h
    r0 = r1  # r0 toma el valor de r1 (anterior)
    r1 = r   # r1 toma el valor del residuo r (nuevo)

    v2 = (v0 - (multiplicacion_nc(c, v1, h, L)))  # Se calcula el nuevo v2 como v0 - c * v1# Actualización del polinomio v usando la operación de multiplicación y sustracción
    
    # Desplazamiento de v0 y v1 para la próxima iteración
    v0 = v1  # v0 toma el valor de v1 (anterior)
    v1 = v2  # v1 toma el valor de v2 (nuevo)

v = v1 # Al final del bucle, v1 contiene el polinomio localizador y evaluador de errores

print()
print("Grado del polinomio localizador y evaluador de errores: ", v.degree)


if(v.degree != t): #Si el grado del polinomio localizador no es t, ha habido un error
    print("Fallo de desencapsulado!")
    #Terminar ejecución
    sys.exit(1)

r = r1 # Inicializa el polinomio r con el valor de r1
A = set()  # Conjunto que almacenará las posiciones de error encontradas, inicialmente vacio
fl = 0  # Indicador para determinar la primera ejecución del bucle principal

# Bucle que continúa hasta que se encuentren exactamente t posiciones de error
while len(A) != t:

    if fl != 0:  # Si no es la primera iteración
        
        B = set(range(n)) - A # B es el conjunto complementario de A en el rango de evaluación
        
        
        f = v # Se inicializa f como el polinomio v
        gr = f.degree  # Se guarda el grado del polinomio f
        i = min(B)  # Se selecciona el menor índice de B
        B.remove(i)  # Se elimina el índice i de B
        
        # Se calcula el MCM de f y (x - punto_de_evaluacion[i]) y se actualiza f
        f = MCM(f, galois.Poly([1, puntos_evaluacion[i]], field=L), h, L)

        # Se incrementa el grado de f hasta que sea igual al grado inicial gr, recorriendo todo el conjunto B
        while f.degree != gr:
            gr += 1
            i = min(B)
            B.remove(i)
            f = MCM(f, galois.Poly([1, puntos_evaluacion[i]], field=L), h, L)

        # Añade la posición i al conjunto de posiciones de error A
        A.add(i)
        
        # Se calcula el MCM de v y (x - punto_de_evaluacion[i]) para obtener v_negada
        v_negada = MCM(v, galois.Poly([1, puntos_evaluacion[i]], field=L), h, L)
        
        # Se divide v_negada por v para obtener el cociente c
        c, u = division_nc(v_negada, v, h, L)
        
        # Se actualiza v con v_negada
        v = v_negada
        
        r_negada = multiplicacion_nc(c, r, h, L) # Se multiplica c por r 
        r = r_negada #Se actualiza r con el resultado

    # Marca que se ha completado la primera iteración
    fl = 1
    
    # B es el conjunto de índices no presentes en A
    B = set(range(n)) - A
    
    print()
    print("Calculando las posiciones de error...")
    print()
    
    # Bucle para encontrar las posiciones de error restantes (en la primera iteración entra aquí directamente)
    while len(B) > 0 and len(A) != t:
        i = min(B)  # Selecciona el menor índice de B
        vector_v = reverse_array(v.coefficients())  # Se obtienen los coeficientes de v de menor a mayor
        polinomio = L(0)  # Inicializa el polinomiko 0 en el cuerpo L
        
        # Calcula el polinomio en el punto de evaluación
        for k in range(len(vector_v)):
            Pk = ((p ** (k * h)) - 1) // ((p ** h) - 1)
            polinomio += (vector_v[k] * (puntos_evaluacion[i] ** Pk))
        
        # Si el polinomio evaluado es cero, se ha encontrado una posición de error y se añade la posición a la lista A
        if polinomio == galois.Poly([0], field=L):
            A.add(i)
        
        # Elimina el índice i de B
        B.remove(i)

    # Si no se encuentran suficientes posiciones de error, se asume que el criptograma o clave es incorrecta
    if len(A) < t:
        print("Criptograma o clave incorrecta.")
        # Termina la ejecución del programa
        sys.exit(1)


#Calculamos elementos necesarios para constuir el sistema de ecuaciones
rho = [L(0)] * t #Calculamos una lista de t elementos nulos
u = [L(0)] * t #Calculamos una lista de t elementos nulos
A=list(A) #Convertimos el conjunto A a una lista 
A.sort() #Ordenamos la lista A

print("Construyendo el sistema de ecuaciones para obtener el secreto compartido...")
for j in range(t): #Cálculo del vector rho, realizando la división correspondiente
    rho[j],u[j] = division_nc(v, galois.Poly([1, puntos_evaluacion[A[j]]], field=L), h, L)


#RESOLVER SISTEMA LINEAL FINAL
vector_a=[0] * t  #Creamos el vector a con t posiciones nulas

for i in range (t):
    vector_a[i]=p ** (h * ((-i) % mu)) #Rellenamos el vector a con el cálculo correspondiente

coefs_r=reverse_array(r.coefficients()) #Obtenemos los coeficientes de r de menor a mayor
num_ceros = (t) - len(coefs_r) #Si el grado es menor que t, rellenamos con 0
if num_ceros > 0:
    coefs_r = coefs_r +( [0] * num_ceros)

matriz_r= [L(0)] * t #Creamos la matriz de una fila r con t elementos nulos
for i in range(t):
    matriz_r[i]=coefs_r[i]**vector_a[i] #Rellenamos r con el cálculo correspondiente
matriz_r=L(matriz_r).T #Trasponemos r para que sea una matriz de una columna

matriz_sistema= [[L(0) for _ in range(t)] for _ in range(t)] #Creamos la matriz sistema de t*t elementos nulos

for j in range (t): #Rellenamos la matriz sistema

    coefs_rho=reverse_array(rho[j].coefficients()) #Obtenemos los coeficientes de rho de menor a mayor
    for i in range (t):
        #Actualizamos cada elemento de matriz sistema con el cálculo correspondiente
        matriz_sistema[i][j]=(coefs_rho[i]**vector_a[i])*(elementos_aleatorios_eta[A[j]]) 

matriz_sistema=L(matriz_sistema) #Convertimos la matriz sistema a una matriz de L


print("Resolviendo el sistema de ecuaciones...")

#Tenemos el siguiente sistema de ecuaciones: matriz_r=matriz_sistema*matriz_e
solucion_e =np.linalg.solve(matriz_sistema, matriz_r) #Hallamos la solución del sistema de ecuaciones con la libreria numpy

print("¡Sistema resuelto!")

print()
print("Obteniendo el valor del secreto compartido...")

#Obtención de la salida del algoritmo de desencapsulado:
vector_salida=F([F(0) for _ in range(n)]) #Creamos el vector de salida con n posiciones nulas

for i in range(len(solucion_e)): #recorremos la solución del sistema de ecuaciones
    lista_eltos_F = coord(solucion_e[i], matriz_B2_inv, m, F) #Obtenemos las coordenadas en F del elemento en L de la solución
    primer_elto_F = lista_eltos_F[0] #Nos quedamos con la primera coordenada
    vector_salida[A[i]]=primer_elto_F #Actualizamos la posición del error correspondiente con el elemento correspondiente

#Hemos desencapsulado el criptograma y obtenido el secreto compartido
print("Secreto compartido obtenido.")
print()

#Creación de un hash del secreto compartido para testear posteriormente el funcionamiento del sistema

print("Generando un hash para el secreto compartido recibido...")
# Crear un objeto hash usando SHA-3 (en este caso, sha3_256)
hash_obj = hashlib.sha3_256()

salida_str = str(vector_salida)

# Alimentar el objeto hash con el texto en bytes
hash_obj.update(salida_str.encode('utf-8'))

# Obtener el hash resultante en formato hexadecimal
hash_result = hash_obj.hexdigest()

print("El secreto compartido ha generado el siguiente hash: ", hash_result)

#GUARDAR EL HASH DEL SECRETO COMPARTIDO EN UN ARCHIVO PARA COMPROBAR QUE FUNCIONA ENCAPSULADO Y DES()

# Convertir el hash a un objeto numpy (array de cadena de texto)
hash_array = np.array([hash_result])
# Guardar el hash en un archivo usando np.save
file_name = "hash_desencapsulado.npy"
np.save("../"+file_name, hash_array)

print()
print("El hash del secreto compartido recibido se ha guardado en el siguiente archivo: ", file_name)

final = time.time() #Finalizamos el tiempo de ejecución del programa
tiempo_transcurrido = final - inicio #Calculamos el tiempo transcurrido durante la ejecución
print()
print("Criptograma desencapsulado en: ",tiempo_transcurrido," segundos.")
print()

