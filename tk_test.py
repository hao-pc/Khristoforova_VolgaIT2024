import tkinter as tk
from tkinter import scrolledtext
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

# Создание DataFrame с примерами адресов
data = {
    'house_uuid': [f'id_{i}' for i in range(11)],
    'house_full_address': [
        "Ульяновская обл, Ульяновск г, Карла Либкнехта ул, 21",
        "Ульяновская обл, Ульяновск г, Карла Либкнехта ул, 23А",
        "Ульяновская обл, Ульяновск г, Карла Либкнехта ул, 23",
        "Ульяновская обл, Ульяновск г, Карла Либкнехта ул, 25",
        "Ульяновская обл, Ульяновск г, Карла Либкнехта ул, 21",
        "Ульяновская обл, Ульяновск г, Пушкинская ул, 10",
        "Ульяновская обл, Ульяновск г, Гончарова ул, 7",
        "Ульяновская обл, Ульяновск г, Гончарова ул, 12",
        "Ульяновская обл, Ульяновск г, Ленина ул, 5",
        "Ульяновская обл, Ульяновск г, Ленина ул, 7",
        "Ульяновская обл, Ульяновск г, Гагарина ул, 8"
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
    individual_matches = re.findall(r"(\d+)(?:-(\d+))?", comment)  # Найти номера и диапазоны
    for match in individual_matches:
        start = int(match[0])
        end = int(match[1]) if match[1] else start  # Если есть диапазон, то конец — это второй номер, иначе — первый

        for house_number in range(start, end + 1):
            uuid, address, score = find_most_similar_address(f"Карла Либкнехта {house_number}")

            # Сохраняем только уникальные адреса с наибольшим сходством
            if uuid not in unique_results or unique_results[uuid][1] < score:
                unique_results[uuid] = (address, score)

    return [(uuid, address, score) for uuid, (address, score) in unique_results.items()]


# Функция для обработки комментария и вывода результатов
def process_comment():
    comment = comment_entry.get()
    matches = find_similar_addresses(comment)
    result_text.delete(1.0, tk.END)
    for match in matches:
        uuid, address, score = match
        result_text.insert(tk.END, f"Адрес: {address} (UUID: {uuid}, Сходство: {score:.4f})\n")


# Создание окна приложения
root = tk.Tk()
root.title("Поиск адресов")
root.geometry("500x400")

# Поле ввода для комментария
tk.Label(root, text="Введите комментарий:").pack(pady=5)
comment_entry = tk.Entry(root, width=50)
comment_entry.pack(pady=5)

# Кнопка для поиска
search_button = tk.Button(root, text="Найти адреса", command=process_comment)
search_button.pack(pady=10)

# Поле для вывода результатов
tk.Label(root, text="Результаты:").pack(pady=5)
result_text = scrolledtext.ScrolledText(root, width=60, height=15)
result_text.pack(pady=5)

# Запуск приложения
root.mainloop()
