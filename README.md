# TFM

El desarrollo de la aplicación de Procesamiento de datos de gran dimensionalidad, eda y visualización BDMED se ha llevado a cabo en un ordenador personal con el sistema operativo Windows 10. Por ello, tanto en este apartado, como en el archivo README del repositorio de GitHub que contiene el proyecto, se indican en todos los requisitos y pasos para la instalación de BDMED en entornos con Windows 10. No se garantiza el funcionamiento de la aplicación si no se realiza la instalación paso por paso. Los pasos a seguir para la instalación se detallan en la lista siguiente:

1.	Instalación de Python. La versión utilizada durante la creación de la aplicación es Python 3.8.6, que puede descargarse en el enlace [46].
2.	Instalación de Pip. En el caso de instalar Python en su versión de Windows desde el sitio web oficial, se incluye de manera predeterminada este sistema de gestión de paquetes de Python. Sin embargo, puede instalarse de manera manual siguiendo el procedimiento provisto en su documentación [47]. 
3.	Descarga e instalación de Java 8, que puede llevarse a cabo desde [48].
4.	Descarga de los paquetes de Spark, preconstruido para Apache Hadoop 2.7. Una vez realizada la descarga, se descomprimen los archivos y el directorio padre se traslada a la ruta siguiente: C:\opt\spark. Este paso es importante, ya que suelen surgir problemas y errores vinculados a la errónea ejecución del mismo.
5.	Inclusión del archivo winutils.exe, incluido en el repositorio de GitHub, a la carpeta con los archivos de Spark.
6.	Adición de las siguientes variables de entorno:
a.	HADOOP_HOME: Directorio donde está instalado Spark. Ejemplo: C:\opt\spark\spark-3.0.1-bin-hadoop2.7.
b.	SPARK_HOME: Mismo valor que HADOOP_HOME.
c.	JAVA_HOME: Carpeta de instalación de Java. Ejemplo: C:\Program Files\Java\jre1.8.0_271.
d.	Añadir a la variable PATH el directorio con la carpeta bin de Java. Ejemplo: C:\Program Files\Java\jre1.8.0_271\bin.
7.	Instalación de todos los paquetes requeridos por la aplicación. Para ello se hace uso de Pip, que instala todos los paquetes incluidos en el archivo requirements.txt con el comando pip install -r requirements.txt, en el caso que el archivo de texto se halle en el directorio de trabajo.
