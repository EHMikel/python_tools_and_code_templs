import openai                                                               # importamos 
import os

#openai.api_key = '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'     # fijamos la clave


def enviar_promt_completions_mode(
        mi_prompt: str, model: str= "gpt-3.5-turbo-instruct", temp= 1, max_tokens: int= 500, 
        probabilidad_acumulada: float= 0.2, frequency_penalty= 0, presence_penalty= 0):
    
    # completions es la seccion de openAI para completar texto
    respuesta = openai.completions.create(
        model= model,                           # indicamos el modelo que vamos a utlizar
        prompt= mi_prompt,                      # el mensaje que le pasamos a chat 
        temperature= temp,                      # regula el grado de aleatoriedad de la respuesta 0 (siempre la misma respuesta), 1 (respuesta con aleatoriedad)
        max_tokens= max_tokens,                 # el maximo de token que queremos generar por cada promt 
        top_p= probabilidad_acumulada,          # delimitamos el universo de tokens de los cuales puede elegir para responder 1 = analiza todos 0.1 solo el 10% con mayor probabilidad, etc

        # se mueven en un rango de -2,2 
        frequency_penalty=frequency_penalty,    # si repiten tokens recibe penalización  
        presence_penalty= presence_penalty      # con que un token aparezca una vez ya recibe penalización
    )

    return respuesta.choices[0].text.strip()    # el indice donde esta la respuesta de nuestro modelo


def enviar_promt_chat_completions_mode(
        mensaje: list, modelo: str = "gpt-4-1106-preview", formato: dict = None, 
        maximo_tokens: int= 500, aleatoriedad: float= 0.5, probabilidad_acumulada: float= 0.5):
     
    respuesta = openai.chat.completions.create(
        messages= mensaje, 
        model= modelo, 
        response_format= None, 
        max_tokens=maximo_tokens, 
        temperature=aleatoriedad, 
        top_p= probabilidad_acumulada
    )

    if formato == {'type': 'json_object'}: 
        return respuesta['choices'][0]['message']['content']
    
    else: return respuesta.choices[0].message.content


def conect_to_bbdd(bbdd_name, user, password, host= 'localhost' ,port= '5432'): 
    from sqlalchemy import create_engine
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{bbdd_name}")

def close_conection(engine): 
    engine.dispose()
    return None


def BBDD_to_text_df(table_name:str, bbdd_name:str,password:str, 
                    user:str, host:str= 'localhost', puerto:str='5432'):
    
    import pandas as pd
    engine = conect_to_bbdd(bbdd_name=bbdd_name, user= user, password=password, host= host, port=puerto)

    try: 
        consulta_sql = f"SELECT * FROM {table_name}"          # crear la consulta
        df = pd.read_sql(consulta_sql, engine)                # read_sql para pasar la consulta a dataframe
        df_text = df.to_markdown()
    
    except Exception as e: 
        print(f"La consulta dio el siguiente error: \n{e}")

    finally: 
        engine.dispose()                                      # cerrar la conexion de forma segura

    return df_text    #, df                # se podria devolver tambien el df original 


def nlp_BBBD_text_df(consulta_nlp:str, bbdd_name:str, password:str, 
                     user:str, puerto:str='5432', host= 'localhost'):
    '''
    Esta función recibe una consulta en lenguaje natural y la formatea a codigo SQL para 
    luego devolver una tabla en formato texto y que la api de chat completions de openai 
    pueda procesar esa tabla.
    '''
    import pandas as pd

    engine = conect_to_bbdd(bbdd_name=bbdd_name, user= user, password=password, host= host, port=puerto)

    try: 

        mi_prompt = [
        {'role': 'system', 'content': 'Eres un asistente experto en bases de datos, que convierte peticiones de lenguaje natural a código SQL.\
                                       DEVUELVES ÚNICAMENTE EL CÓDIGO SQL'},      
        {'role': 'user',   'content': f'{consulta_nlp}'},
        {'role': 'assistant', 'content': 'DAME SOLO EL CÓDIGO SQL SIN NADA MÁS'}
        ]

        respuesta_sql = enviar_promt_chat_completions_mode(
            mensaje=mi_prompt, 
            probabilidad_acumulada=1, 
            aleatoriedad=0)

        consulta_sql = respuesta_sql[7:-4]                    # extraigo la parte de código de la respuesta
        df = pd.read_sql(consulta_sql, engine)                # read_sql para pasar la consulta a dataframe
        df_text = df.to_markdown()
    
    except Exception as e: 
        print(f"La consulta dio el siguiente error: \n{e}")

    finally: 
        engine.dispose()                                      # cerrar la conexion de forma segura

    return df_text    


def get_embedding(texto, model= "text-embedding-ada-002"):
    text = texto.replace('\n', ' ')
    respuesta = openai.embeddings.create(input= text, model= model)
    return respuesta.data[0].embedding


def cosine_similarity(embedding1, embedding2): 
    #print(type(embedding1), type(embedding2))
    import numpy as np
    from numpy.linalg import norm
    cos_sim = np.dot(embedding1, embedding2)/(norm(embedding1)*norm(embedding2))
    return cos_sim

def buscar_metadata_mas_similar(prompt, metadata, n_resulatdos= 5): 
    prompt_embedding = get_embedding(prompt)                                                       # se extrae el embedding de la pregunta del usuario
    metadata['similarity']= metadata['embedding'].apply(lambda x: cosine_similarity(x, prompt_embedding))   # se saca la similitud de la pregunta con las posibles respuestas
    metadata = metadata.sort_values('similarity', ascending= False)                                             # los ordena de mayor similitud a menor
    return metadata.iloc[:n_resulatdos][['metadata_str', 'similitud', 'embedding']]


def store_simple_metadata_in_json_str(engine):
    from   sqlalchemy import MetaData
    import json
    metadata = MetaData()       # MetaData es un contenedor que mantiene información sobre las tablas y modelos en una base de datos.
    metadata.reflect(engine)    # Cargar la información de la base de datos incluyendo 

    tablas_columnas = {table.name: [column.name for column in table.columns] for table in metadata.tables.values()}
    
    return json.dumps(tablas_columnas, indent= 4)
    

def store_full_metadata_in_json_str(engine):
    from   sqlalchemy import MetaData
    import json
    metadata = MetaData()       # MetaData es un contenedor que mantiene información sobre las tablas y modelos en una base de datos.
    metadata.reflect(engine)    # Cargar la información de la base de datos incluyendo 
    
    metadatos = {}
    for table in metadata.tables.values():
            # Por cada tabla, se almacena información detallada de sus columnas en el diccionario 'tablas_columnas'.
            # Se crea una clave en el diccionario para cada nombre de tabla.
        metadatos[table.name] = {                                     
        "columnas": {column.name: {                                   # dentro de cada tabla se crea un diccionario de 
                        "tipo": str(column.type),                     # Tipo de dato de la columna.
                        "nulo": column.nullable,                      # Booleano que indica si la columna acepta valores nulos.
                        "clave_primaria": column.primary_key,         # Booleano que indica si la columna es una clave primaria
                        "clave_foranea": bool(column.foreign_keys)}   # Booleano que indica si la columna es una clave foránea.
                    for column in table.columns}                      # Este bucle interno itera a través de todas las columnas de la tabla.
            }
    return json.dumps(metadatos, indent= 4)


def full_metadata_pipeline(engine):
    from sqlalchemy import MetaData
    import json
    import pandas as pd
    import numpy as np
    import openai

    metadata = MetaData()     # MetaData es un contenedor que mantiene información sobre las tablas y modelos en una base de datos.
    metadata.reflect(engine)  # Cargar la información de la base de datos

    metadatos = {}
    for table in metadata.tables.values():
        metadatos_tabla = []  # Lista para almacenar los metadatos de cada tabla

        for column in table.columns:
            columna_metadatos = {
                "col_name": column.name,                       # se guarda el nombre de la columna
                "type": str(column.type),                      # Tipo de dato de la columna.
                "null": column.nullable,                       # Booleano que indica si la columna acepta valores nulos.
                "primary_key": column.primary_key,             # Booleano que indica si la columna es una clave primaria
                "foreing_key": bool(column.foreign_keys)       # Booleano que indica si la columna es una clave foránea.
            }

            str_md_columna = json.dumps(columna_metadatos)  # se pasan los metadatos a str
            metadatos_tabla.append(str_md_columna)          # Añadir los metadatos y embedding de cada columna a la lista

        metadatos_tabla_str = 'table_name:' + table.name + '\n' + '\n'.join([str_md_col for str_md_col in metadatos_tabla])
        
        md_embedding = np.array(get_embedding(texto=metadatos_tabla_str))  # calculo el embedding de los metadatos
        metadatos[table.name] = [metadatos_tabla_str, md_embedding]        # Añadir los metadatos de la tabla al diccionario

    md_df = pd.DataFrame(metadatos)
    md_df.index = ['metadata_str', 'embedding']     # set_index(keys= ['metadata_str', 'embedding'])
    md_df = md_df.T                                 #.reset_index(drop= True, inplace= True)
    
    return md_df


def simple_metadata_pipeline(engine):

    from   sqlalchemy import MetaData
    import json
    import numpy as np
    import pandas as pd

    metadata = MetaData()       # MetaData es un contenedor que mantiene información sobre las tablas y modelos en una base de datos.
    metadata.reflect(engine)    # Cargar la información de la base de datos incluyendo 
    metadatos = {}

    #tablas_columnas = {table.name: [column.name for column in table.columns] for table in metadata.tables.values()}
    #md_str = '\n'.join([table_name + ':' + columns for table_name, columns in tablas_columnas.items()])

    for table in metadata.tables.values():
        md = {}
        col_names = [column.name for column in table.columns]
        md[table.name] = col_names
        md_str = json.dumps(md)
        md_embedding = np.array(get_embedding(texto= md_str))
        metadatos[table.name] = [md_str, md_embedding]

    
    md_df = pd.DataFrame(metadatos)
    md_df.index = ['metadata_str', 'embedding']     # set_index(keys= ['metadata_str', 'embedding'])
    md_df = md_df.T                                 #.reset_index(drop= True, inplace= True)
    
    return md_df

# def store_metadata_df_to_pickle(bbdd_name, user, password, host= 'localhost', port= '5432'):

#     import os
#     path = 'data/'
#     name = bbdd_name
#     formato = '.pickle'

#     if not os.path.exists(path + name + '_metadata' + formato):

#         engine = conect_to_bbdd(bbdd_name=name, user= user, password=password, host= host, port= port)
#         metadata_df = metadata_pipeline(engine)
#         close_conection(engine)
#         # display(metadata_df)
#         # print(metadata_df.info)
#         metadata_df.to_pickle(path + name + '_metadata')

#     else: 
#         print(f'el archivo ya existe en: {path}, con el nombre: {name}_metadata{formato}')

#     return None


def nlp_to_BBBD_with_metadata_to_text_df(
        consulta_nlp:str, bbdd_name:str, password:str, 
        user:str, puerto:str='5432', host= 'localhost'):
    '''
    Esta función recibe una consulta en lenguaje natural y la formatea a codigo SQL para 
    luego devolver una tabla en formato texto y que la api de chat completions de openai 
    pueda procesar esa tabla. PROPORCIONA INFO DE LOS METADATOS DE LA BBDD AL ASISTENTE
    '''
    import pandas as pd
    import re
    
    try: 
        # creacion del motor de conexion a la BBDD via SQL para postgresql
        engine = conect_to_bbdd(bbdd_name=bbdd_name, user= user, password=password, host= host, port=puerto)

        metadatos_json_str = store_full_metadata_in_json_str(engine= engine)

        mi_prompt = [
        {'role': 'system', 'content': f'Eres un asistente experto en bases de datos, que convierte peticiones de lenguaje natural a código SQL.\
                                       AQUI TIENES LO METADATOS DE LA BASE DE DATOS: \n{metadatos_json_str}'},      
        {'role': 'user',   'content': f'{consulta_nlp}'},
        {'role': 'assistant', 'content': 'Devuélveme SOLO EL CODIGO SQL y en este formato: \n```sql\nEL CODIGO SQL;\n```'}
        ]

        respuesta_sql = enviar_promt_chat_completions_mode(
            mensaje=mi_prompt, 
            probabilidad_acumulada=1, 
            aleatoriedad=0)
        
        regex_pattern = r"```sql\n(.*?;)\n```" #r"```sql\n([A-Z].*?);"
        coincidencia = re.search(regex_pattern, respuesta_sql, re.DOTALL)

        if coincidencia:
            codigo_sql = coincidencia.group(1).strip()  # .strip() para eliminar espacios extra
            print(codigo_sql)
        else:
            raise KeyError("no se encontro codigo sql en la consulta")

        # codigo_sql = respuesta_sql[7:-4]                  # extraigo la parte de código de la respuesta
        df = pd.read_sql(codigo_sql, engine)                # read_sql para pasar la consulta a dataframe
        df_text = df.to_markdown()
    
    except Exception as e: 
        print(f"La consulta dio el siguiente error: \n{e}")

    engine.dispose()                                      # cerrar la conexion de forma segura

    return df_text    
