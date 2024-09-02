import numpy as np
import sys
import time

# Verificar que se pasen dos argumentos (los nombres de los archivos)
if len(sys.argv) != 3:
    print("Uso: python comparar_hashes.py <archivo_hash1.npy> <archivo_hash2.npy>")
    sys.exit(1)

inicio = time.time() #Iniciamos el tiempo de ejecución del programa

print()
print("Comparación de hashes para comprobar el funcionamiento del ciptosistema.")
print()

# Obtener los nombres de los archivos desde los argumentos
file_encapsulado = sys.argv[1]
file_dencapsulado = sys.argv[2]

# Cargar el hash generado en el encapsulado desde el archivo
hash_encapsulado_array = np.load("../"+file_encapsulado)
hash_encapsulado = hash_encapsulado_array[0]

# Cargar el hash generado en el desencapsulado desde el archiv
hash_encapsulado_array = np.load("../"+file_dencapsulado)
hash_desencapsulado = hash_encapsulado_array[0]

# Mostrar los hashes cargados
print(f"Hash del encapsulado {file_encapsulado}: {hash_encapsulado}")
print(f"Hash del desencapsulado {file_dencapsulado}: {hash_desencapsulado}")
print()

# Comparar los hashes y mostrar el resultado
if hash_encapsulado == hash_desencapsulado:
    print("Los hashes son iguales. El secreto compartido se ha transmitido con éxito")
    print()
    print("¡¡EL CRIPTOSISTEMA FUNCIONA CORRECTAMENTE!!")
    print()
    print("SE ACABÓ :)")
else:
    print("ERROR: Los hashes son diferentes.")

final = time.time() #Finalizamos el tiempo de ejecución del programa
tiempo_transcurrido = final - inicio #Calculamos el tiempo transcurrido durante la ejecución
print()
print("Hashes comparados en: ",tiempo_transcurrido," segundos.")
print()

