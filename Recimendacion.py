import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Generando un conjunto de productos más amplio
productos = {
    "comida_perros": "alimentacion",
    "comida_gatos": "alimentacion",
    "juguete_gatos": "juguetes",
    "juguete_perros": "juguetes",
    "camita_perros": "descanso",
    "camita_gatos": "descanso",
    "arena_gatos": "higiene",
    "correa_perros": "accesorios",
    "rascador_gatos": "juguetes",
    "bol_comida_gatos": "alimentacion",
    "collar_perros": "accesorios",
    "caseta_perros": "descanso",
    "cama_gatos": "descanso",
    "limpiador_arenero": "higiene",
    "comederos_automáticos": "alimentacion",
    "juguete_inteligente_perros": "juguetes",
    "juguete_inteligente_gatos": "juguetes"
}

# Simulando datos de clientes y sus calificaciones/compras (1-5)
datos = {
    "cliente1": {"comida_perros": 5, "correa_perros": 4, "juguete_perros": 3, "camita_perros": 5},
    "cliente2": {"juguete_gatos": 5, "arena_gatos": 4, "rascador_gatos": 3, "cama_gatos": 4},
    "cliente3": {"camita_perros": 5, "juguete_gatos": 3, "bol_comida_gatos": 4, "collar_perros": 3},
    "cliente4": {"comida_gatos": 5, "bol_comida_gatos": 4, "arena_gatos": 3, "rascador_gatos": 2},
    "cliente5": {"comida_perros": 4, "comederos_automáticos": 5, "juguete_inteligente_perros": 4, "caseta_perros": 3},
    "cliente6": {"comida_gatos": 3, "limpiador_arenero": 4, "juguete_inteligente_gatos": 5, "cama_gatos": 3},
    "cliente7": {"comida_perros": 2, "correa_perros": 3, "camita_perros": 5, "collar_perros": 4},
    "cliente8": {"comida_gatos": 5, "arena_gatos": 3, "bol_comida_gatos": 4, "rascador_gatos": 3},
    "cliente9": {"juguete_perros": 4, "correa_perros": 5, "caseta_perros": 3, "juguete_inteligente_perros": 4},
    "cliente10": {"comida_gatos": 4, "camita_gatos": 5, "limpiador_arenero": 4, "juguete_inteligente_gatos": 4},
    # Añadir más clientes para mayor diversidad
}

# Transformando los datos en una lista de triples (cliente, producto, calificación)
data_list = []
for cliente, productos_comprados in datos.items():
    for producto, rating in productos_comprados.items():
        data_list.append((cliente, producto, rating))

# Definiendo el formato del dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(pd.DataFrame(data_list, columns=["usuario", "item", "rating"]), reader)

# Dividiendo en conjuntos de entrenamiento y prueba
trainset, testset = train_test_split(data, test_size=0.25)

# Usando SVD (Singular Value Decomposition) para el modelo de recomendación
algo = SVD()
algo.fit(trainset)

# Predicción de calificaciones para el conjunto de prueba
predictions = algo.test(testset)

# Evaluación del modelo
print(f'RMSE: {accuracy.rmse(predictions)}')

# Ejemplo de recomendación para un cliente específico
cliente = "cliente4"
todos_los_productos = productos.keys()
compras_cliente = datos[cliente].keys()
productos_no_comprados = [producto for producto in todos_los_productos if producto not in compras_cliente]

# Recomendaciones
recomendaciones = [(producto, algo.predict(cliente, producto).est) for producto in productos_no_comprados]
recomendaciones.sort(key=lambda x: x[1], reverse=True)

print(f"Recomendaciones para {cliente}: {[r[0] for r in recomendaciones]}")
