
# Criptosistema de McEliece/Niederreiter con Códigos Skew Goppa

## Descripción del Proyecto

Este proyecto implementa un criptosistema de McEliece/Niederreiter utilizando códigos skew Goppa en Python. El sistema incluye la generación de claves, el encapsulado y desencapsulado de secretos compartidos, y la verificación del funcionamiento a través de hashes. 

## Estructura del Repositorio

El repositorio está organizado de la siguiente manera:

- **README.md**: Este archivo con información general del proyecto, instrucciones de instalación y uso.
- **src/**: Carpeta que contiene el código fuente del proyecto.
  - `generar_clave.py`: Genera la clave pública y los parámetros del criptosistema.
  - `encapsular.py`: Construye y encapsula un secreto compartido aleatorio en un criptograma.
  - `desencapsular.py`: Desencapsula el criptograma para obtener el secreto compartido.
  - `comparar_hashes.py`: Compara los hashes generados al encapsular y desencapsular para verificar el funcionamiento del criptosistema.
- **.gitignore**: Indica qué elementos ignorar al hacer commits en el repositorio.
- **Memoria.pdf**: Documento que incluye la memoria del proyecto con toda la información técnica, resultados y conclusiones.

## Instalación del Entorno de Desarrollo

### Visual Studio Code

Visual Studio Code es un editor de código fuente recomendado para el desarrollo de Python. A continuación se detallan las instrucciones para instalarlo en diferentes sistemas operativos:

#### Windows

1. Descarga el instalador desde la [página oficial de Visual Studio Code](https://code.visualstudio.com/Download).
2. Ejecuta el archivo descargado y sigue las instrucciones del asistente de instalación.
3. Abre Visual Studio Code e instala la extensión de Python desde el Marketplace de Visual Studio Code.

Para más detalles, consulta el [tutorial de instalación en Windows](https://code.visualstudio.com/docs/setup/windows).

#### Ubuntu

1. Abre una terminal y ejecuta los siguientes comandos:
    ```bash
    sudo apt-get install wget gpg
    wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
    sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
    echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list > /dev/null
    rm -f packages.microsoft.gpg
    ```
2. Actualiza la caché de paquetes e instala el paquete:
    ```bash
    sudo apt-get update
    sudo apt-get install code
    ```
3. Abre Visual Studio Code e instala la extensión de Python desde el Marketplace.

Para más detalles, consulta el [tutorial de instalación en Ubuntu](https://code.visualstudio.com/docs/setup/linux).

#### macOS

1. Descarga el archivo `.dmg` desde la [página oficial de Visual Studio Code](https://code.visualstudio.com/Download).
2. Abre el archivo descargado y arrastra el icono de Visual Studio Code a la carpeta de Aplicaciones.
3. Abre Visual Studio Code e instala la extensión de Python desde el Marketplace.

Para más detalles, consulta el [tutorial de instalación en macOS](https://code.visualstudio.com/docs/setup/mac).

### Python

Python es el lenguaje de programación utilizado para el desarrollo del proyecto. A continuación se indican las instrucciones para instalarlo:

#### Windows

1. Descarga el instalador desde la [página oficial de Python](https://www.python.org/downloads/).
2. Ejecuta el archivo descargado y asegúrate de marcar la opción **Add Python to PATH** antes de hacer clic en **Install Now**.
3. Verifica la instalación abriendo una terminal (cmd) y ejecutando:
    ```bash
    python --version
    ```

Consulta el [tutorial de instalación en Windows](https://docs.python.org/3/using/windows.html).

#### Ubuntu

1. Verifica si ya tienes Python instalado ejecutando:
    ```bash
    python3 --version
    ```
2. Si no está instalado, abre una terminal y ejecuta:
    ```bash
    sudo apt-get update
    sudo apt-get install python3
    ```
3. Verifica la instalación ejecutando:
    ```bash
    python3 --version
    ```

Consulta el [tutorial de instalación en Ubuntu](https://python-guide-es.readthedocs.io/es/latest/starting/install3/linux.html).

#### macOS

macOS ya viene con una versión de Python preinstalada. Para instalar la última versión estable (3.x):

1. Descarga el instalador desde la [página oficial de Python](https://www.python.org/downloads/macos/).
2. Ejecuta el instalador y sigue las instrucciones.
3. Verifica la instalación ejecutando:
    ```bash
    python3 --version
    ```

Consulta el [tutorial de instalación en macOS](https://kinsta.com/es/base-de-conocimiento/instalar-python/).

## Instalación de las Librerías Necesarias

Este proyecto utiliza varias librerías de Python que no vienen preinstaladas con la distribución estándar. A continuación, se detallan las instrucciones para instalar estas librerías:

### Librerías Preinstaladas

Estas librerías vienen con la distribución estándar de Python:
- `math`: Funciones matemáticas.
- `sys`: Acceso a variables usadas por el intérprete.
- `time`: Funciones relacionadas con el tiempo.
- `random`: Generación de números aleatorios.
- `hashlib`: Algoritmos de hashing y cifrado.

### Librerías a Instalar

Las siguientes librerías deben instalarse manualmente:

- `numpy`: Facilita la computación científica con objetos multidimensionales. [Documentación de numpy](https://numpy.org/doc/stable/)
- `galois`: Extiende `numpy` para trabajar con cuerpos finitos. [Documentación de galois](https://github.com/fastmath/galois)
- `progress`: Muestra barras de progreso en la terminal. [Documentación de progress](https://pypi.org/project/progress/)

#### Instrucciones de Instalación

**Windows:**
1. Abre el símbolo del sistema (Cmd) o PowerShell.
2. Asegúrate de tener `pip` instalado:
    ```bash
    python -m ensurepip --upgrade
    ```
3. Instala las librerías utilizando:
    ```bash
    pip install numpy galois progress
    ```

**Ubuntu:**
1. Abre la terminal.
2. Asegúrate de tener `pip` instalado:
    ```bash
    sudo apt-get install python3-pip
    ```
3. Instala las librerías utilizando:
    ```bash
    pip3 install numpy galois progress
    ```

**macOS:**
1. Abre la terminal.
2. Asegúrate de tener `pip` instalado:
    ```bash
    sudo easy_install pip
    ```
3. Instala las librerías utilizando:
    ```bash
    pip install numpy galois progress
    ```

#### Verificación de Instalación

Para verificar que las librerías se han instalado correctamente, abre una consola de Python e intenta importar cada una de ellas:
```python
import numpy as np
import galois
from progress.bar import Bar
