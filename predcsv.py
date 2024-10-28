from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
import numpy as np
import re

# Инициализация модели для векторизации текста
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Инициализация клиента и создание коллекции
client = chromadb.Client()
collection = client.get_or_create_collection("test_addresses")

# Загрузка адресов из файла с указанием разделителя и кодировки
addresses_df = pd.read_csv('volgait2024-semifinal-addresses.csv', sep=';', encoding='iso-8859-1')
address_texts = addresses_df['house_full_address'].tolist()
embeddings = [model.encode(addr).astype(np.float32) for addr in address_texts]
ids = addresses_df['house_uuid'].tolist()

# Добавление всех адресов в коллекцию партиями
batch_size = 5000  # Устанавливаем размер партии
for i in range(0, len(address_texts), batch_size):
    batch_texts = address_texts[i:i + batch_size]
    batch_embeddings = embeddings[i:i + batch_size]
    batch_ids = ids[i:i + batch_size]
    collection.add(
        documents=batch_texts,
        embeddings=batch_embeddings,
        ids=batch_ids
    )

# Загрузка комментариев из файла с указанием разделителя и кодировки
comments_df = pd.read_csv('volgait2024-semifinal-task.csv', sep=';', encoding='iso-8859-1')

# Функция для поиска наиболее похожего адреса по одному комментарию
def find_most_similar_address(comment):
    comment_embedding = model.encode(comment).astype(np.float32)
    results = collection.query(
        query_embeddings=[comment_embedding],
        n_results=1
    )
    best_uuid = results['ids'][0][0]
    best_address = results['documents'][0][0]
    similarity_score = results['distances'][0][0]
    return best_uuid, best_address, similarity_score

# Функция для поиска адресов по диапазону и перечислению
def find_similar_addresses(comment):
    unique_results = {}

    # Разбиваем комментарий на отдельные адреса
    individual_matches = re.findall(r"(\d+)(?:-(\d+))?", comment)
    for match in individual_matches:
        start = int(match[0])
        end = int(match[1]) if match[1] else start

        for house_number in range(start, end + 1):
            temp_comment = re.sub(r"\d+(-\d+)?", str(house_number), comment, count=1)
            uuid, address, score = find_most_similar_address(f"Карла Либкнехта {house_number}")

            if uuid not in unique_results or unique_results[uuid][1] < score:
                unique_results[uuid] = (address, score)

    return list(unique_results.keys())

# Обработка всех комментариев и сохранение результатов
results = []
for _, row in comments_df.iterrows():
    shutdown_id = row['shutdown_id']
    comment = row['comment']
    matching_uuids = find_similar_addresses(comment)
    house_uuids = ",".join(matching_uuids)

    results.append({
        "shutdown_id": shutdown_id,
        "house_uuids": house_uuids
    })

# Создание DataFrame с результатами и запись в CSV
results_df = pd.DataFrame(results)
results_df.to_csv('volgait2024-semifinal-result.csv', sep=';', index=False, encoding='iso-8859-1')


