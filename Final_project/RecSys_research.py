import os
import csv
from typing import List, Dict
from loguru import logger
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Задание параметров для исследования
CHUNK_SIZES = [1024, 512, 2048]
CHUNK_OVERLAPS = [0, 100, 200]
K_VALUES = [3, 5, 10, 25]
MODEL_ID = 'intfloat/multilingual-e5-large'
DOCS_PATH = "docs"

# Словарь с вопросами и явным указанием на файл, по которому задан вопрос
GROUND_TRUTH = {
    "What is the Adam method based on?":
        ["docs/Adam Optimizer.pdf"],
    "How does Adam's bias correction mechanism work, and why is it important for optimization?":
        ["docs/Adam Optimizer.pdf"],
    "How does the Transformer architecture eliminate the need for recurrent or convolutional layers in sequence modeling?":
        ["docs/Attention Is All You Need.pdf"],
    "What are the two key pre-training tasks used in BERT, and how do they contribute to downstream task performance?":
        ["docs/BERT.pdf"],
    "How does unCLIP differ from traditional text-to-image generation models like GLIDE or DALL-E?":
        ["docs/DALL-E 2.pdf"],
    "How does the residual learning framework differ from traditional deep neural network training?":
        ["docs/Deep Residual Learning.pdf"],
    "What is the core idea behind Generative Adversarial Networks (GANs)?":
        ["docs/GANs.pdf"],
    "Why does the global optimum of the GAN framework occur when p_g = p_data?":
        ["docs/GANs.pdf"],
    "What methods were used to evaluate and mitigate test set contamination in the benchmarks?":
        ["docs/GPT-3.pdf"],
    "What biases were identified in GPT-3 regarding gender, race, and religion, and how were these biases measured?":
        ["docs/GPT-3.pdf"],
    "What are the main advantages of Latent Diffusion Models (LDMs) over traditional pixel-based diffusion models?":
        ["docs/Stable Diffusion.pdf"],
    "What limitations do the authors identify for LDMs, particularly in tasks requiring high pixel-level precision?":
        ["docs/Stable Diffusion.pdf"],
    "What is the purpose of the weighted loss function, and how is it calculated?":
        ["docs/U-Net.pdf"],
    "What is the purpose of the weighted loss function in the U-Net, and how is it calculated?":
        ["docs/U-Net.pdf"]
}

# Логирование
os.makedirs("log", exist_ok=True)
logger.add("log/experiment.log",
           format="{time} {level} {message}",
           level="INFO",
           rotation="500 KB",
           compression="zip")

def load_documents_from_folder(folder_path: str):
    """
    Функция считывает файлы формата PDF и docx.

    :param folder_path: Путь к папке с файлами.

    :return documents: Список считаных файлов.
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

def build_chroma_db(documents, chunk_size, chunk_overlap, db_dir):
    """
    Функция создает векторные представления данных из загруженных файлов при различных параметрах эмбеддинга.
    Если добавляются новые файлы - создается и добавляется к базе эмбеддинг новых файлов.

    :param documents: Список считаных файлов.
    :param chunk_size: Размер одного чанка в количестве символов
    :param chunk_overlap: Размер перекрытия чанков в количестве символов
    :param db_dir: Путь до папки с файлами для эмбеддинга

    :return db: Векторное представление файлов
    """
    # Создаём эмбеддинги
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_ID, model_kwargs={'device': 'cpu'})

    # Инициализируем или загружаем Chroma DB
    db = Chroma(
        collection_name="doc_collection",
        embedding_function=embeddings,
        persist_directory=db_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Получаем уже сохранённые источники
    try:
        existing = db.get(include=["metadatas"])
        existing_sources = {meta.get("source") for meta in existing["metadatas"] if "source" in meta}
    except Exception as e:
        existing_sources = set()

    # Отбираем только новые документы
    new_documents = [doc for doc in documents if doc.metadata.get("source") not in existing_sources]

    if not new_documents:
        print("Нет новых документов для добавления.")
        return db

    print(f"Найдено {len(new_documents)} новых документов. Добавляем в базу...")

    # Разбиваем на чанки
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(new_documents)

    # Добавляем в базу
    db.add_documents(chunks)

    return db

def precision_at_k(predicted: List[str], ground_truth: List[str]) -> float:
    """
    Функция определяет долю чанков, соответсвующих введенному запросу пользователя, от общего числа чанков, отобраных
    программой как наиболее релевантные запросу пользователя.

    :param predicted: Список отобранных чанков
    :param ground_truth: Список путей к файлам, соответсвующих запросу пользователя (задается явно)

    :return: Доля правильно определенных чанков в общем числе отобранных
    """
    if not predicted:
        return 0.0
    gt_normalized = set(os.path.normpath(path) for path in ground_truth)
    pred_normalized = [os.path.normpath(os.path.relpath(p)) for p in predicted]

    relevant = sum(1 for src in pred_normalized if src in gt_normalized)
    return relevant / len(predicted)

def run_experiments():
    """
    Функция запускает программу и записываи результаты работы в .csv файл для дальнейшего исследования.
    """
    logger.info("Начинаем загрузку документов")
    documents = load_documents_from_folder(DOCS_PATH)
    logger.info(f"Загружено документов: {len(documents)}")

    results = []

    for chunk_size in CHUNK_SIZES:
        for chunk_overlap in CHUNK_OVERLAPS:
            db_dir = f"chroma_db_exp/chunk{chunk_size}_overlap{chunk_overlap}"
            os.makedirs(db_dir, exist_ok=True)

            logger.info(f"Создаём базу с chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
            db = build_chroma_db(documents, chunk_size, chunk_overlap, db_dir)

            for k in K_VALUES:
                for query, gt_sources in GROUND_TRUTH.items():
                    logger.debug(f"Поиск: query='{query}' | k={k}")
                    results_with_scores = db.similarity_search_with_score(query, k=k)
                    predicted_sources = [doc.metadata.get("source") for doc, _ in results_with_scores]
                    prec = precision_at_k(predicted_sources, gt_sources)

                    logger.info(f"chunk={chunk_size}, overlap={chunk_overlap}, k={k}, query='{query}', precision={prec:.3f}")

                    if results_with_scores:
                        top_doc, top_score = results_with_scores[0]
                        print("\n" + "-" * 80)
                        print(f" Запрос: {query}")
                        print(f" Топ-1 документ: {top_doc.metadata.get('source')}")
                        print(f" Оценка (score): {top_score:.4f}")
                        print(f" Похожие документы:")
                        for i, (doc, score) in enumerate(results_with_scores[1:3], start=2):
                            print(f"   {i}. {doc.metadata.get('source')} (score={score:.4f})")
                        print("-" * 80 + "\n")
                    else:
                        print(f"\n[!] Запрос '{query}': нет результатов.\n")

                    results.append({
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "k": k,
                        "query": query,
                        "precision": round(prec, 3)
                    })

    out_file = "experiment_results.csv"
    with open(out_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_size", "chunk_overlap", "k", "query", "precision"])
        writer.writeheader()
        writer.writerows(results)

    logger.success(f"Эксперименты завершены. Результаты сохранены в {out_file}")

if __name__ == "__main__":
    run_experiments()
