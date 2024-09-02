import galois
import numpy as np
import random
import hashlib
import sys
import time

#OBTENCIÓN DE PARÁMETROS Y DATOS

# Verificar que se pasen dos argumentos
if len(sys.argv) != 2: #Si hay error en los argumentos, se indica cómo debe ejecutarse
    print()
    print("ERROR: Debe ejecutar el archivo con el siguiente comando: python.exe encapsular.py <nombre_archivo_clave_publica>")
    sys.exit(1)

inicio = time.time() #Iniciamos el tiempo de ejecución del programa

# Obtener el de los archivos desde los argumentos
archivo_clave = "../archivos/"+sys.argv[1]

print()
print("Encapsulamiento de un secreto compartido usando la clave previamente generada")


# Cargar los datos desde el archivo de clave .npz
data = np.load(archivo_clave, allow_pickle=True)

# Obtenemos los parámetros del archivo y los convertimos al tipo correspondiente)
n = int(data['n'])
t = int(data['t'])
p = int(data['p'])
d = int(data['d'])
k = int(data['k'])

#Construcción del cuerpo F

#Obtenemos los coeficientes del polinomio irreducible de F y construimos el polinomio.
coefs_def_F = data['polinomio_def_F']
polinomio_def_F=galois.Poly(coefs_def_F,field=galois.GF(2))

#Construimos el cuerpo F
F = galois.GF(pow(p,d), irreducible_poly=polinomio_def_F, repr='poly')
F_elemento = random.choice(F.elements)
print()
print("El cuerpo finito de Galois F es GF(p^d): ",F.properties)
print("Elemento random de F: ", F_elemento)

#Obtenemos la clave pública del archivo de la clave
H_pub = data['H_pub']
H_pub=F(H_pub) #La transformamos a una matriz de F
H_Pub_T=H_pub.T #Obtenemos la matriz traspuesta de la clave pública

print()
print("La clave pública H_pub ha sido cargada correctamente")

print()
print("PARÁMETROS:")
print("n: ",n," t: ",t," p: ",p," d: ",d," k: ",k)


print("Cálculo del secreto compartido...")
print()
#Calculamos el secreto compartido:

secreto_compartido = F([F(0) for _ in range(n)]) #Creamos una lista de n elementos nulos

contador = 0
while contador < t: #Ejecutamos t veces
    pos = random.randint(0, n - 1) #Calculamos una posición random de la lista
    valor_no_nulo = random.choice(F.elements) #Calculamos un elemento random de F
    if (secreto_compartido[pos] == F(0)) and (valor_no_nulo != F(0)): #Si la posición es nula y el elemento no es nulo
        secreto_compartido[pos] = valor_no_nulo #Cambiamos la posición elegida de la lista al valor aleatorio
        contador += 1 #Aumentamos el contado

print("Cálculo del criptograma...")
#Cálculo del criptograma
criptograma=F(secreto_compartido@H_Pub_T) #El criptograma es la matriz en F resultante de multiplicar e secreto compartido por la traspuesta de la clave pública

# Mostrar el criptograma generado por pantalla
print()
print("Criptograma generado para el envio: ",criptograma)

# Guardar el criptograma en un archivo llamado "criptograma.npy"
criptograma_file_name = "criptograma.npy"
np.save("../archivos/"+criptograma_file_name, criptograma)

print()
print("El criptograma se ha guardado en el archivo: ",criptograma_file_name, "de la carpeta archivos")

#Creación de un hash del secreto compartido para testear posteriormente el funcionamiento del sistema

# Crear un objeto hash usando SHA-3 (en este caso, sha3_256)
hash_obj = hashlib.sha3_256()

secreto_str = str(secreto_compartido)

# Alimentar el objeto hash con el texto en bytes
hash_obj.update(secreto_str.encode('utf-8'))

# Obtener el hash resultante en formato hexadecimal
hash_result = hash_obj.hexdigest()

print()
print("El secreto compartido ha generado el siguiente hash: ", hash_result)


#GUARDAR EL HASH DEL SECRETO COMPARTIDO EN UN ARCHIVO PARA COMPROBAR QUE FUNCIONA ENCAPSULADO Y DES()

# Convertir el hash a un objeto numpy (array de cadena de texto)
hash_array = np.array([hash_result])
# Guardar el hash en un archivo usando np.save
file_name = "hash_encapsulado.npy"
np.save("../archivos/"+file_name, hash_array)

print()
print("El hash se ha guardado en el archivo: ",file_name, "de la carpeta archivos")

final = time.time() #Finalizamos el tiempo de ejecución del programa
tiempo_transcurrido = final - inicio #Calculamos el tiempo transcurrido durante la ejecución
print("Secreto compartido encapsulado en: ",tiempo_transcurrido," segundos.")
