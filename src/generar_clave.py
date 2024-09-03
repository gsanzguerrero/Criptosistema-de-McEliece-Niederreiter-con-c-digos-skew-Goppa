import galois
from math import ceil, floor, gcd
import random
import numpy as np
import time
from progress.bar import Bar

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


print()
print("Generando la clave de nuestro criptosistema.")
print()

inicio = time.time() #Iniciamos el tiempo de ejecución del programa

print("Calculando parámetros...")
print()

# CÁLCULO DE PARÁMETROS

#base_prime, base_exp = 2, 1
#base_prime, base_exp = 2, 2
#base_prime, base_exp = 2, 3
#base_prime, base_exp = 2, 4
base_prime, base_exp = 2, 8 #Byte
p, d = base_prime, base_exp

#length, correction_capability = 512, 6
length, correction_capability = 512, 5
#length, correction_capability = 1024, 10

n, t = length, correction_capability

# Calcula los valores de mmin y mmax
mmin, mmax = ceil(n / (10 * t)), floor(n / (4 * t))

k=n-(2*t*mmax)

# Encuentra las opciones válidas de m y delta
opciones = []
for m in range(mmin, mmax+1):
    for delta in range(1, d * m): 
        if (d * m) % delta == 0 and n * delta / (d * (p ** delta - 1)) <= m:
            opciones.append((m, delta))

#Elección de una pareja de m y delta válida
m, delta = random.choice(opciones)

#Cáluclo de mu 
mu=(d*m)//delta

while(2*t<mu):
    m, delta = random.choice(opciones)
    mu=int((d*m)/delta)

#2t > mu para dar más variabiliad al polinomio g, si 2t<mu g sería x^(2*t) y daría más informacion
# a un posible interceptor.

#Cálculo de h
h=random.randint(1,d*m)
while(gcd(h,d*m)!=delta):
    h=random.randint(1,d*m)

#Cálculo de r
r=(sum(p**(i*d) for i in range (m)))

print()
print("PARÁMETROS: ")
print("n: ",n, "t: ",t," p: ",p," d: ",d," k: ",k," m: ",m," delta: ",delta," h: ",h, ", mu: ", mu, " r: ", r)
print()

#Construcción del cuerpo L

#Obtención del polinomio irreducible de L
polinomio_def_L = galois.primitive_poly(p, d*m)

#Construimos L con el polinomio irredicible calculado.
L = galois.GF(pow(p,d*m), irreducible_poly=galois.primitive_poly(p, d*m), repr='poly')
primitivo_L=L.primitive_element
num_aleatorio=random.randint(0, p**(d*m))
L_elemento = primitivo_L**num_aleatorio
print()
print("El cuerpo finito de Galois L es GF(p^d): ",L.properties)
print("Elemento random de L: ", L_elemento)

#Construcción del cuerpo F

#Obtención del polinomio irreducible de F a partir del de L
primitivo_FenL = primitivo_L**r

lista_irreducibles = list(galois.irreducible_polys(p,d))
for polinomio in lista_irreducibles:
    aux = list(polinomio.coefficients(order='asc'))
    suma = L(0)
    for i in range(len(aux)):
        suma += int(aux[i])*primitivo_FenL**i
    if suma == galois.Poly.Zero(field=L):
        polinomio_def_F = polinomio

#Construimos F con el polinomio irredicible calculado
F = galois.GF(pow(p,d), irreducible_poly=polinomio_def_F, repr='poly')
F_elemento = random.choice(F.elements)
print()
print("El cuerpo finito de Galois F es GF(p^d): ",F.properties)
print("Elemento random de F: ", F_elemento)


# Obtención de las matrices de cambio de base entre los cuerpos F y L

B2 = [primitivo_L**(i+j*r) for i in range(m) for j in range(d) ] 

B2_aux = [reverse_array(list(L.vector(i))) for i in B2]

matriz_B2_np = np.array(B2_aux).reshape(d*m, d*m)

matriz_B2=galois.GF(int(2))(matriz_B2_np)
matriz_B2_inversa=np.linalg.inv(matriz_B2)

print()
print("Matriz para embebimiento de F en L: ", matriz_B2)


print()
#Seleccion de un elemento aleatorio que constituye una base normal
alpha=L_elemento

while(not es_normal(alpha, p, delta, mu, L)): #Calculamos elementos hasta que sea normal
    num_aleatorio=random.randint(0, p**(d*m))
    alpha = primitivo_L**num_aleatorio

print("El elememento ",alpha," es normal.")

#Cálculo del vector normalizador siguiendo las restricciones correspondiente
print()
print("Calculando el vector normalizador...")
print()


#Seleccion de n elementos no nulos de L (lista eta)

elementos_aleatorios_eta=[] #Definimos una lista donde guardaremos los elementos no nulos
barra_eta = Bar('Vector normalizador:', max=n) #Este elemento de la libreria progress genera una barra en la salida del programa que va mostrando el progreso del bucle

#Se deben seleccionar n elementos no nulos de L.
while len(elementos_aleatorios_eta)<n:
    num_aleatorio=random.randint(0, p**(d*m))
    elemento_eta = primitivo_L**num_aleatorio #Calculamos un elemento aleatorio de L

    if elemento_eta != L(0):
        elementos_aleatorios_eta.append(elemento_eta) #Si es no nulo, lo añadimos
        barra_eta.next() #Actualizamos la barra de progreso

print("Vector normalizador calculado")
print("Tamaño del vector normalizador: ", len(elementos_aleatorios_eta))

#Cálculo de los puntos de evaluación
print()
print("Cálculo de los puntos de evaluación...")
puntos_evaluacion=[] #Definimos una lista donde guardaremos los puntos de evaluación
gamma=L.primitive_element #Obtenemos un elemento primitivo de L
print("Gamma = ", gamma)
print()
barra_puntos_evaluacion = Bar('Puntos de evaluación:', max=n) #Este elemento de la libreria progress genera una barra en la salida del programa que va mostrando el progreso del bucle
#Selección de n elementos random diferentes del cuerpo L
while(len(puntos_evaluacion)<n):
    i=random.randint(0,mu-1)
    j=random.randint(0, (p**delta)-2)
    punto_evaluacion=(gamma**(j))*((alpha**(p**(h*(i+1))))/(alpha**((p**(h*i))))) #Calculamos el punto de evaluación
    if(punto_evaluacion not in puntos_evaluacion):
        puntos_evaluacion.append(punto_evaluacion) #Si el elemento no se encuentra ya en la lista, lo añadimos
        barra_puntos_evaluacion.next() #Actualizamos la barra de progreso

print("Puntos de evaluación calculados")
print("Tamaño del vector de puntos de evaluación: ", len(puntos_evaluacion))

#Construcción del polinomio g (normal o bilatero)
#G es un polinomio bilatero. estos se construyen como el producto de un polinomio central por una potencia de x. 
#Los polinomios centrales tienen coeficientes en K y valores no nulos solo en potencias múltiplos de mu. 
#Para ello:

print()
print("Construcción del polinomio de Goppa y los polinomios de paridad")

#Calculamos un primitivo de K (K es el subcuerpo de L de orden p^(delta))
exponente_K = sum(p**(i*delta) for i in range(mu))

primitivo_K = primitivo_L**exponente_K

#Fabricamos polinomio de K en L -> el grado es s=2t/mu (si no es entero, pillamos la parte entera)
#Queremos polinomio de ese grado -> pillamos una lista de 2t/mu+1 elementos (un 1 y 2t/mu elementos de K)
#¿Cómo cogemos eltos de K de forma aleatoria y homogenea? Repetimos S veces
#Cogemos nº aleatorio entre 0 y p**(delta) -1 randint(0,(p**delta)-1 ). Si este nº es p**(delta)-1 añadimos el 0 a la lista.
#Si no, elevamos primK a ese numero aleatorio y añadimos. Tenemos coeficientes de polonimio de grado s

#Este comentario de arriba no se si quitarlo

cociente_S, resto_S=divmod(2*t,mu) #Calculamos el cociente y el resto de dividir 2t entre mu

repetir=True

while repetir: #Calculamos (en bucle, si es necesario) los coeficientes de g
    repetir=False
    coeficientes_g=[1] #El primer coeficiente de g es 1 para que g tenga una potencia de x
    for i in range(0,cociente_S): #De 0 a cociente_S-1 veces
        num_aleatorio=random.randint(0,(p**delta)-1) #Calculamos un nº aleatorio entre 0 y p^(delta)-1
        if num_aleatorio == ((p**delta)-1):
            coeficientes_g.append(L(0)) #Si el nº es exactamente p^(delta)-1, añadimos un 0 a los coeficientes
        else:
            coeficientes_g.append(primitivo_K**num_aleatorio) #Si no, añadimos el elemento correspondiente visto en K.

    print("Tamaño Coeficientes polinomio Goppa: ", len(coeficientes_g))

    #Tenemos que extender el polinomio de coeficientes g para que los coeficientes sean solo en posiciones múltiplos de mu y que g tenga el grado deseado
    coeficientes_extendidos_g = []
    i=mu-1
    j=resto_S
    for elemento in coeficientes_g:
        if(len(coeficientes_extendidos_g)>0):
            coeficientes_extendidos_g.extend([0] * i) #Calculamos y rellenamos las posiciones múltiplos de mu
        coeficientes_extendidos_g.append(elemento)

    # Añadir j ceros al final de la lista
    coeficientes_extendidos_g.extend([0] * j) #Extendemos el vector de coeficientes al final para generar un polinomio del grado deseado

    g=galois.Poly(coeficientes_extendidos_g, field=L) #Construimos el polinomio g 
    print("Polinomio g construido. ")
    print()

    #Polinomios de paridad son los inversos de polinomios tipo x-punto_evaluacion mod g. 
    print("Cálculando los polinomio no conmutativos de paridad...")
    print()

    polinomios_nc_paridad=[] #Definimos una lista donde guardaremos los polinomios de paridad
    barra_polinomios_paridad = Bar('Polinomios de paridad:', max=len(puntos_evaluacion)) #Este elemento de la libreria progress genera una barra en la salida del programa que va mostrando el progreso del bucle
    for punto_evaluacion in puntos_evaluacion: #Para cada punto de evaluación
        polinomio=galois.Poly([1, punto_evaluacion], field=L) #calculamos el polinomio de la forma x-punto de evaluación
        polinomio_nc_paridad=PCP(polinomio, g, h, L) #Calculamos el polinomio nc de paridad correspondiente
        if(polinomio_nc_paridad==galois.Poly.Zero(L)):
            repetir=True
            print("Polinomio de Goppa no válido, reiniciamos el cálculo.") #Si algun modulo es 0, el polinomio g no es válido y volvemos a construirlo
            break
        polinomios_nc_paridad.append(polinomio_nc_paridad) #Añadimos el polinomio de paridad a la lista de polinomios de paridad
        barra_polinomios_paridad.next() #Actualizamos la barra de progreso

print()

print("Tamaño de la lista de polinomios no conmutativos de paridad H: ", len(polinomios_nc_paridad))

print()

#Generación de la clave pública:

#Cálculo de la matriz de paridad H
#Definición de las distintas matrices necesarias para constuir la clave pública
print("Cálculo de la matriz de paridad...")
H=[[L(0) for _ in range (n)] for _ in range (2*t)]
H_negada=[[L(0) for _ in range (n)] for _ in range (2*t)]
H_negada_en_F=[[[] for _ in range (n)] for _ in range (2*t)]
matriz_H=[[F(0) for _ in range (n)] for _ in range (2*t*m)] #matriz de paridad H

for j in range(n):
    coefs_polinomio=reverse_array(polinomios_nc_paridad[j].coefficients()) #Obtenemos los parámetros de menor a mayor de cada polinomio de paridad

    for i in range(2*t):
        H[i][j]=coefs_polinomio[i] #Guardamos los coeficientes de los polinomios de paridad
        H_negada[i][j]=(H[i][j]**(p**(((-i)%mu)*h)))*elementos_aleatorios_eta[j] #Calculamos los elementos de la matriz H_negada
        H_negada_en_F[i][j]=coord(H_negada[i][j],matriz_B2_inversa, m, F) #Calculamos las coordenadas en F de los elementos de H_negada

print("Paso 1 completado")
#Cálculo de los elementos de la matriz de paridad a partir de 
for i in range(2*t*m):
    for j in range (n):
        a,b=divmod(i,m) #Calculamos las posiciones a y b
        matriz_H[i][j]=H_negada_en_F[a][j][b] #seleccionamos los elementos necesarios y los guardamos en la matriz de paridad H

print()
print("La matriz de paridad H se ha calculado. Tiene un tamaño de ",len(H),"x",len(H[0]))

#CÁLCULO DE LA CLAVE PÚBLICA
print()
print("Calculando la clave pública...")

matriz_H=F(matriz_H)

if np.linalg.matrix_rank(matriz_H)==(n-k): #Si el rango de la matriz de paridad es n-k, 
    H_pub = matriz_H.row_reduce()          #la clave pública es la forma escalonada reducida por filas de la matriz de paridad,  
else:
    sigue=True
    while(sigue): #Si no, repetimos en bucle:
        matriz_r=([[F(0) for _ in range (n)] for _ in range (n-k-np.linalg.matrix_rank(matriz_H))]) #Construimos una matriz auxiliar r de tamaño n-k-rango(matriz_H) x n
        matriz_r=F(matriz_r) #Transformamos la matriz de una lista de listas a una matriz del cuerpo F
        for i in range (n-k-np.linalg.matrix_rank(matriz_H)):
            for j in range(n):
                matriz_r[i][j]=random.choice(F.elements) #Rellenamos la matriz r con elementos random del cuerpo F

        matriz_q=matriz_H
        matriz_q=matriz_q.tolist() #Transformamos la matriz de una matriz de F a una lista de listas
        for fila in matriz_r:
            matriz_q.append(fila) #Añadimos las filas de la matriz r a la matriz aleatoria Q

        matriz_q = F(matriz_q) #Volvemos a convertir la matriz a una matriz de F
        matriz_q_rref = matriz_q.row_reduce() #Calculamos la forma escalonada reducida por filas de la matriz q
        H_pub = matriz_q_rref[:n - k] #La clave pública son las n-k primeras filas de la matriz escalonada reducida por filas de q
        if(np.linalg.matrix_rank(H_pub)==(n-k)): #Si la clave pública tiene rango n-k es válida, si no se repite este proceso
            sigue=False

print("Tenemos la clave pública, una matriz de dimensiones ",len(H_pub),"x",{len(H_pub[0])})

#Guardamos la clave en archivos simulando su envio a un transmisor y receptor

#El receptor necesita todos los parámetros y elementos de la clave para desencapsular
clave_privada='Clave.npz'
np.savez("../archivos/"+clave_privada, H_pub=H_pub, n=n, t=t, p=p, d=d, k=k, m=m, delta=delta, h=h, mu=mu, r=r, polinomio_def_L=polinomio_def_L.coefficients(), polinomio_def_F=polinomio_def_F.coefficients(), polinomios_nc_paridad=polinomios_nc_paridad, elementos_aleatorios_eta=elementos_aleatorios_eta, puntos_evaluacion=puntos_evaluacion, g=g.coefficients(), matriz_B2=matriz_B2)
print()
print("La clave privada y los parámetros del criptosistema se han guardado en el archivo ",clave_privada, "de la carpeta archivos")

#El transmisor necesita algunos parámetros y elementos de la clave para encapsular
clave_publica='Clave_pub.npz'
np.savez("../archivos/"+clave_publica, H_pub=H_pub, n=n, t=t, p=p, d=d, k=k, polinomio_def_F=polinomio_def_F.coefficients())
print()
print("La clave pública y los parámetros del criptosistema se han guardado en el archivo ",clave_publica, "de la carpeta archivos")

#Obtenemos el tiempo que ha tardado la generación de la clave y lo mostramos por pantalla
final = time.time() #Finalizamos el tiempo de ejecución del programa
tiempo_transcurrido = final - inicio #Calculamos el tiempo transcurrido durante la ejecución
print("Clave generada en: ",tiempo_transcurrido," segundos.")
print()
