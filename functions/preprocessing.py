import pandas as pd
import numpy as np
from datetime import datetime
import collections
from collections import Counter
import os
import logging

# Funciones usadas

# Función que reemplaza los elementos de una lista usando un diccionario generado en python
def replace_dictionary(list_test, dictionary):
    return [dictionary.get(item, item) for item in list_test]

# Función que calcula la edad de una persona dada su fecha de nacimiento y el día en que 
def calculate_age(born, alta):
    return alta.year - born.year - ((alta.month, alta.day) < (born.month, born.day))

# Función que devuelve el índice de un elemento y si no hay apariciones del mismo devuelve un -1
def index_withoutexception(list_param, i):
    try:
        return list_param.index(i)
    except:
        return -1

def preprocessCMBD(data):
    ##### Creación de diccionarios y mapeo del resto de variables que no son diagnósiticos ########

    # lectura de los datos originales

    # variables seleccionadas (las más interesantes)
    selected_columns = ['HISTORIAmodificada', 'TXTFECNAC', 'SEXO', 'PROCEDE' ,'TXTFECCONT', 'TXTFECING',
                         'TIPING', 'TXTFECALT', 'TIPALT', 'CONTINUIDA', 'TRASH', 'MUNIRESI', 'PAISNAC',
                         'REGFIN', 'UCI', 'DIASUCI', 'C1'] 

    data_final = data[selected_columns]

    # creación de los diccionarios para el posterior mapeo de las variables discretas seleccionadas

    #Financiación (REGFIN)
    dict_REGFIN = {1: 'Sistema Nacional de Salud. Residente en Andalucía', 2: 'Sistema Nacional de Salud. Residente en otra Comunidad',
            3: 'Reconocimiento del derecho de asistencia en Andalucía', 
            4: 'Convenio Unión Europea',
            5: 'Convenio internacional estatal',
            6: 'Instituciones penitenciarias',
            7: 'Mutualidades de los regímenes especiales de la Seguridad Social de funcionarios (MUFACE, ISFAS y MUGEJU). Opción publica',
            8: 'Mutualidades de los regímenes especiales de la Seguridad Social de funcionarios (MUFACE, ISFAS y MUGEJU). Opción privada',
            9: 'Privado particular (personas físicas)',
            10: 'Compañías de seguro sanitario privado. Accidentes de tráfico',
            11: 'Compañías de seguro sanitario privado. Excepto accidentes de tráfico',
            12: 'Mutuas Colaboradoras con la Seguridad Social (Accidentes de trabajo o enfermedad profesional)',
            13: 'Asistencia Sanitaria',
            14: 'Otros'
           }

    data_final['REGFIN'] = replace_dictionary(data_final['REGFIN'], dict_REGFIN)

    #SEXO
    dict_SEXO = {1: 'Hombre', 2: 'Mujer', 3: 'Indeterminado'}
    data_final['SEXO'] = replace_dictionary(data_final['SEXO'], dict_SEXO)

    #Ámbito de procedencia (PROCEDE)
    dict_PROCEDE = {1: 'Urgencias del propio hospital',
            2: 'Consultas del propio hospital',
            3: 'Lista de Espera Quirúrgica',
            4: 'Hospital de día médico del propio hospital',
            5: 'Hospital de día quirúrgico del propio hospital',
            6: 'Nacidos en el hospital',
            7: 'Hospitalización del propio hospital',
            8: 'Hospitalización a domicilio del propio hospital',
            9: 'Procedimientos ambulatorios de especial complejidad del propio hospital',
            10: 'Otro hospital',
            11: 'Por orden judicial',
            12: 'Hospitalización de comunidad terapéutica de salud mental',
            13: 'Centro de Atención Primaria'
           }
    data_final['PROCEDE'] = replace_dictionary(data_final['PROCEDE'], dict_PROCEDE)

    #País de nacimiento (PAISNAC)
    
    logger = logging.getLogger('app_api')

    # se carga el archivo que contiene el diccionario que se va a usar en el mapeo de la variable PAISNAC
    df_PAISNAC = pd.read_csv('./bdmed/functions/mapeoPAISNAC.csv', index_col=0, squeeze=True, header=0, delimiter=";")

    dict_PAISNAC = df_PAISNAC.to_dict()
    data_final['PAISNAC'] = replace_dictionary(data_final['PAISNAC'], dict_PAISNAC)

    #CIRCUNSTANCIA DEL INGRESO O CONTACTO (TIPING)
    dict_TIPING = {1: 'Urgente', 2: 'Programado'}
    data_final['TIPING'] = replace_dictionary(data_final['TIPING'], dict_TIPING)

    #circunstancia del alta (TIPALT)
    dict_TIPALT = {1: 'Destino al domicilio',
            2: 'Traslado a otro hospital',
            3: 'Traslado a residencia social',
            4: 'Alta voluntaria',
            5: 'Defunción',
            6: 'Hospitalización a domicilio',
            7: 'In Extremis',
            8: 'Fuga',
            9: 'Hospitalización de comunidad terapéutica de salud mental',
            10: 'Hospitalización de agudos', 
    }
    data_final['TIPALT'] = replace_dictionary(data_final['TIPALT'], dict_TIPALT)

    #Continuidad asistencial (CONTINUIDA)
    dict_CONTINUIDA = {1: 'No precisa',
            2: 'Ingreso en hospitalización',
            3: 'Hospitalización a domicilio',
            4: 'Hospital de día médico',
            5: 'Hospital de día quirúrgico',
            6: 'Urgencias',
            7: 'Consultas',
            8: 'Comunidad terapéutica de salud mental',
    }
    data_final['CONTINUIDA'] = replace_dictionary(data_final['CONTINUIDA'], dict_CONTINUIDA)
    
    # C1 - diagnóstico principal (nivel 1)
    df_nivel1 = pd.read_csv('./bdmed/functions/diagnosticos2020_nivel1def.csv', index_col=0, squeeze=True, header=0, delimiter=";")
    di1 = df_nivel1.to_dict()
    simplified_column = data_final['C1'].str.slice(0, 3)
    data_level1 = simplified_column.to_frame()
    data_level1 = data_level1.replace({'C1': di1})
    
    data_final['C1'] = data_level1
    
    
    ######### Creación de nuevas variables a partir de las iniciales ###########

    data1620 = data_final
    
    # EDAD

    # Se seleccionan las columnas necesarias
    fecha_nac = data1620['TXTFECNAC']
    fecha_ing = data1620['TXTFECING']
    fecha_alta = data1620['TXTFECALT']
    fecha_cont = data1620['TXTFECCONT']

    # Se calcula la edad del paciente calculando la diferencia entre la fecha del alta médica y la de nacimiento
    edades = []
    for i in range(len(fecha_nac)):
        if (len(str(fecha_nac[i])) == 7):
            fecha_nac_obj = datetime.strptime(( '0' + str(fecha_nac[i])), '%d%m%Y')
        else:
            fecha_nac_obj = datetime.strptime(str(fecha_nac[i]), '%d%m%Y')

        fecha_alta_obj = datetime.strptime(str(fecha_alta[i]), '%d%m%Y %H%M')
        edades.append(calculate_age(fecha_nac_obj, fecha_alta_obj))

    # TIEMPOING

    # Se calcula el tiempo que ha estado ingresado el paciente a partir de la diferencia entre la fecha de alta y
    # la de ingreso, teniendo en cuenta que si proviene de urgencias la fehca de ingreso es la de contacto
    tiempo_ingresado = []
    for i in range(len(fecha_ing)):
        if (len(str(fecha_ing[i])) == 3): #nan case
            fecha_ing_obj = datetime.strptime(str(fecha_cont[i]), '%d%m%Y %H%M')
        else:
            fecha_ing_obj = datetime.strptime(str(fecha_ing[i]), '%d%m%Y %H%M')

        fecha_alta_obj = datetime.strptime(str(fecha_alta[i]), '%d%m%Y %H%M')
        tiempo_ingresado_obj = fecha_alta_obj - fecha_ing_obj
        tiempo_ingresado_days = tiempo_ingresado_obj.days + tiempo_ingresado_obj.seconds/60/60/24
        tiempo_ingresado.append(str(round(tiempo_ingresado_days,2)))

    # MES

    # se cargan en memoria las fechas de ingreso
    fechas_ing = []
    for i in range(len(fecha_ing)):
        if (len(str(fecha_ing[i])) == 3): #nan case
            fecha_ing_obj = datetime.strptime(str(fecha_cont[i]), '%d%m%Y %H%M')
        else:
            fecha_ing_obj = datetime.strptime(str(fecha_ing[i]), '%d%m%Y %H%M')
        fechas_ing.append(fecha_ing_obj)

    # se crea el diccionario con los meses
    meses_numero = []
    dict_meses = {1: 'Enero',
            2: 'Febrero',
            3: 'Marzo',
            4: 'Abril',
            5: 'Mayo',
            6: 'Junio',
            7: 'Julio',
            8: 'Agosto',
            9: 'Septiembre',
            10: 'Octubre',
            11: 'Noviembre',
            12: 'Diciembre'
    }

    for date in fechas_ing:
        meses_numero.append(date.month)

    meses = replace_dictionary(meses_numero, dict_meses)

    # ESTACIONES

    # se fijan los inicios de cada una de las estaciones
    estaciones = []
    inicio_verano = datetime.strptime("21062020", '%d%m%Y')
    inicio_otoño = datetime.strptime("21092020", '%d%m%Y')
    inicio_invierno = datetime.strptime("21122020", '%d%m%Y')
    inicio_primavera = datetime.strptime("21032020", '%d%m%Y')

    # por cada fecha de ingreso, se clasifica basándonos en las fechas de inicio de cada estación
    for date in fechas_ing:
        date_simplified = datetime(2020, date.month, date.day)
        if (date_simplified < inicio_otoño and date_simplified >= inicio_verano):
            estaciones.append('verano')
        elif (date_simplified < inicio_invierno and date_simplified >= inicio_otoño):
            estaciones.append('otoño')
        elif (date_simplified < inicio_verano and date_simplified >= inicio_primavera):
            estaciones.append('primavera')
        else:
            estaciones.append('invierno')

    # AÑO
    # por cada fecha de ingreso, se añade el año de la fecha
    years = []
    for date in fechas_ing:
        years.append(date.year)

    # se añaden las columnas al dataframe
    data1620_mapeados = data_final
    data1620_mapeados['EDAD'] = edades
    data1620_mapeados['TIEMPOING'] = tiempo_ingresado
    data1620_mapeados['MES'] = meses
    data1620_mapeados['MESNUM'] = meses_numero
    data1620_mapeados['ESTACION'] = estaciones
    data1620_mapeados['ANO'] = years
    
    return data1620_mapeados