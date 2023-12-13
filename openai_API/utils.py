import openai                                                               # importamos 
import os

openai.api_key = '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'     # fijamos la clave


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


def BBDD_to_text_df(table_name:str, bbdd_name:str, user:str = 'postgres', password:str= '123', puerto:str='5432'):
    import pandas as pd
    from sqlalchemy import create_engine

    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@localhost:{puerto}/{bbdd_name}")

    try: 
        consulta_sql = f"SELECT * FROM {table_name}"          # crear la consulta
        df = pd.read_sql(consulta_sql, engine)                # read_sql para pasar la consulta a dataframe
        df_text = df.to_markdown()
    
    except Exception as e: 
        print(f"La consulta dio el siguiente error: \n{e}")

    finally: 
        engine.dispose()                                      # cerrar la conexion de forma segura

    return df_text    #, df                # se podria devolver tambien el df original 

def nlp_BBBD_text_df(consulta_nlp:str, bbdd_name:str, user:str = 'postgres', password:str= '123', puerto:str='5432'):
    '''
    Esta función recibe una consulta en lenguaje natural y la formatea a codigo SQL para 
    luego devolver una tabla en formato texto y que la api de chat completions de openai 
    pueda procesar esa tabla.
    '''
    import pandas as pd
    from sqlalchemy import create_engine

    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@localhost:{puerto}/{bbdd_name}")

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

def nlp_to_BBBD_with_metadata_to_text_df(
        consulta_nlp:str, bbdd_name:str, user:str = 'postgres', 
        password:str= '123', puerto:str='5432'):
    '''
    Esta función recibe una consulta en lenguaje natural y la formatea a codigo SQL para 
    luego devolver una tabla en formato texto y que la api de chat completions de openai 
    pueda procesar esa tabla. PROPORCIONA INFO DE LOS METADATOS DE LA BBDD AL ASISTENTE
    '''
    import pandas as pd
    from  sqlalchemy import create_engine, MetaData
    import json
    import re
    
    try: 
        engine = create_engine(f"postgresql+psycopg2://{user}:{password}@localhost:{puerto}/{bbdd_name}")
        metadata = MetaData()
        metadata.reflect(engine)
        tablas_columnas= {table.name: [column.name for column in table.columns] for table in metadata.tables.values()}
        tablas_columnas_json_str = json.dumps(tablas_columnas, indent= 4)

        mi_prompt = [
        {'role': 'system', 'content': f'Eres un asistente experto en bases de datos, que convierte peticiones de lenguaje natural a código SQL.\
                                       AQUI TIENES LO METADATOS DE LA BASE DE DATOS: \n{tablas_columnas_json_str}'},      
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
