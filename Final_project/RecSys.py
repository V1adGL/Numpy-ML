import os
from loguru import logger
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# Логирование
logger.add("log/chroma_ranking.log", format="{time} {level} {message}", level="DEBUG", rotation="100 KB", compression="zip")


def load_documents_from_folder(folder_path):
    """
    Функция считывает файлы формата PDF и docx

    :param folder_path: Путь до папки с файлами

    :return documents: Список считанных файлов
    """
    documents = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif file.endswith(".docx"):
                loader = Docx2txtLoader(path)
            else:
                continue
            for doc in loader.load():
                doc.metadata["source"] = path
                documents.append(doc)
    return documents


def get_chroma_db(size, overlap):
    """
    Функция создает векторное представление данных из загруженных файлов.
    Если добавляются новые файлы - создается и добавляется к базе эмбеддинг новых файлов.

    :param size: Число символов в чанке.
    :param overlap: Число символов перекрытия чанков.

    :return db: Эмбеддинг файлов
    """

    db_path = f"chroma_db_exp/chunk{size}_overlap{overlap}"
    collection_name = "doc_collection"
    model_id = 'intfloat/multilingual-e5-large'
    embeddings = HuggingFaceEmbeddings(model_name=model_id, model_kwargs={'device': 'cpu'})

    # Инициализация базы
    logger.debug("Загружаем или создаём ChromaDB")
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=db_path,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Получаем уже добавленные источники
    try:
        existing = db.get(include=["metadatas"])
        existing_sources = {meta.get("source") for meta in existing["metadatas"] if "source" in meta}
    except Exception as e:
        logger.warning(f"Не удалось получить существующие источники: {e}")
        existing_sources = set()

    # Загружаем документы
    documents = load_documents_from_folder("docs")
    new_documents = [doc for doc in documents if doc.metadata.get("source") not in existing_sources]

    if not new_documents:
        logger.info("Нет новых документов для добавления.")
        return db

    # Разбиваем новые документы на чанки
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = text_splitter.split_documents(new_documents)

    # Добавляем в базу
    logger.info(f"Добавляем {len(chunks)} новых чанков из {len(new_documents)} документов.")
    db.add_documents(chunks)

    return db


def search_similar_documents(query, db, k=3):
    """
    Функция поиска наиболее релевантного документа(файла) через косинусное расстояние.

    :param query: Запрос пользователя
    :param db: Эмбеддинг файлов
    :param k: Количество самых похожих чанков для обработки

    :return: Список самых похожих документов(файлов) в порядке уменьшения соответсвия запросу
    """
    logger.debug(f"Поиск похожих документов по запросу: {query}")
    results = db.similarity_search_with_score(query, k=k)
    doc_scores = {}  # Словарь для хранения оценок для документов
    seen_sources = set()  # Множество для отслеживания уже добавленных источников

    for doc, score in results:
        source = doc.metadata.get("source")
        if source and source not in seen_sources:
            seen_sources.add(source)
            if source not in doc_scores:
                doc_scores[source] = []
            doc_scores[source].append(score)

    # Агрегируем оценки для каждого документа (например, среднее)
    aggregated_results = []
    for source, scores in doc_scores.items():
        avg_score = sum(scores) / len(scores)  # Средняя оценка
        aggregated_results.append({
            "source": source,
            "score": avg_score
        })

    # Сортируем документы по итоговой оценке
    aggregated_results.sort(key=lambda x: x['score'], reverse=True)

    return aggregated_results


if __name__ == "__main__":
    db = get_chroma_db(512, 0)
    query = ''' What is the core idea behind Generative Adversarial Networks (GANs)?''' # Запрос пользователя
    results = search_similar_documents(query, db, k=3)

    if results:
        print("\nНаиболее релевантный документ:")
        print(f"Источник: {results[0]['source']}, Оценка схожести: {results[0]['score']:.3f}")
        if len(results) > 1:
            print("\nВозможно подойдут:")
            for r in results[1:3]:
                print(f"Источник: {r['source']}, Оценка схожести: {r['score']:.3f}")
    else:
        print("Ничего не найдено.")

