# Христофорова Алёна, VolgaIT 2024

![](https://img.shields.io/badge/Python-3.12.x-yellow)

## Содержание
  - [Описание проекта:](#описание-проекта)
  - [Руководство пользователя:](#руководство-пользователя)

## Описание проекта:
Задачей полуфинала предлагается разработка алгоритма, который по заданным доступным адресам домов и текстовому описанию отключения максимально точно определял бы перечень домов, которые были упомянуты.

Моё решение подразумевает получения на вход комментария, предобработка (обработка диапазонов, перечислений, обработка излишних данных) и сравнение с адресами посредством семантического поиска. Для него применялась [модель](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) paraphrase-multilingual-mpnet-base-v2, предназначенная для решения такой задачи (Sentence Simularity). Остановилась именно на этой модели (пыталась использовать также paraphrase-multilingual-MiniLM-L12-v2), т.к. она сочетает высокую точность совпадения и хорошую скорость. 

## Руководство пользователя:
В репозитории можно проследить последовательный ход решения вплоть до финального скрипта. Необходимые зависимости для проверки можно загрузить с помощью requirements.txt

onepred.py - реализация алгоритма семантического поиска для комментариев, где содержится только 1 адрес

multipred.py - реализация алгоритма семантического поиска для комментариев, где содержатся сколько угодно адресов, включая перечисления и диапазоны, обработка излишних данных (по типу д=100, пз/х)

tk_test.py - GUI на Tkinter, позволяющая вписать свой комментарий и поискать по имеющимся в программе адресам 

predcsv.py - реализация алгоритма для csv, предоставленных в задании, а также выдающая на выход csv volgait2024-semifinal-result.csv. Требует файлы volgait2024-semifinal-addresses.csv и volgait2024-semifinal-task.csv. Может занимать очень большое количество времени, поэтому для теста на небольшом количестве данных рекомендуется самостоятельно сократить количество данных в файлах.
