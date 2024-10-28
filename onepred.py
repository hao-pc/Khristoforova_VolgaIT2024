from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
import numpy as np

# Инициализация модели для векторизации текста
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Инициализация клиента и создание коллекции
client = chromadb.Client()
collection = client.get_or_create_collection("test_addresses")

# Создание DataFrame с 10 примерами адресов
data = {
    'house_uuid': [f'id_{i}' for i in range(4)],
    'house_full_address': [
        "Ульяновская обл, Ульяновск г, Карла Либкнехта ул, 21",
        "Ульяновская обл, Ульяновск г, Карла Либкнехта ул, 23А",
        "Ульяновская обл, Ульяновск г, Карла Либкнехта ул, 23",
        "Ульяновская обл, Ульяновск г, Карла Либкнехта ул, 25"
    ]
}
addresses_df = pd.DataFrame(data)

# Векторизация и добавление в коллекцию
address_texts = addresses_df['house_full_address'].tolist()
embeddings = [model.encode(addr).astype(np.float32) for addr in address_texts]
ids = addresses_df['house_uuid'].tolist()

# Добавление всех адресов в коллекцию за один раз
collection.add(
    documents=address_texts,
    embeddings=embeddings,
    ids=ids
)


def find_most_similar_address(comment):
    comment_embedding = model.encode(comment).astype(np.float32)
    # Поиск ближайшего совпадения
    results = collection.query(
        query_embeddings=[comment_embedding],
        n_results=1
    )

    # Извлечение первого совпадения
    best_uuid = results['ids'][0][0]
    best_address = results['documents'][0][0]
    similarity_score = results['distances'][0][0]

    return best_uuid, best_address, similarity_score


# Пример использования
comment = "Карла Либкхнета 23-25"
best_uuid, best_address, score = find_most_similar_address(comment)
print(f"Комментарий: {comment}")
print(f"Наиболее похожий адрес: {best_address} (UUID: {best_uuid}, Сходство: {score})")
