from bs4 import BeautifulSoup
import requests
import re
import os
from pathlib import Path
from PyPDF2 import PdfFileMerger
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Function to download PDF from URL
def download_pdf(url, output_path):
    response = requests.get(url, stream=True, timeout=300)
    with open(output_path, 'wb') as f:
        f.write(response.content)

# Function to create a folder if it doesn't exist
def create_folder(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def clean_filename(name):
    remove_characters = "[],/\\:.;\"'?!*"
    return name.translate(str.maketrans(remove_characters, len(remove_characters) * " ")).strip()


def get_hathitrust_info(book_url):
    response = requests.get(book_url)
    soup = BeautifulSoup(response.text, "html.parser")
    pages = int(soup.find("section", {'class': 'd--reader--viewer'})['data-total-seq'])
    name = soup.find('meta', {'property': 'og:title'})['content'][:55]
    return name, pages


def download_and_merge_pdf(book_url):
    book_name, total_pages = get_hathitrust_info(book_url)
    book_name = clean_filename(book_name)

    create_folder(book_name)
    output_folder = f"{os.getcwd()}/{book_name}/"

    bar = tqdm(total=total_pages, desc=f"Downloading pages for {book_name}")

    for page_number in range(1, total_pages + 1):
        pdf_url = f'https://babel.hathitrust.org/cgi/imgsrv/download/pdf?id={id_book};orient=0;size=100;seq={page_number};attachment=0'
        pdf_path = f'{output_folder}page{page_number}.pdf'

        download_pdf(pdf_url, pdf_path)

        while os.path.getsize(pdf_path) < 6000:
            download_pdf(pdf_url, pdf_path)

        bar.update(1)

    bar.close()

    merge_pdfs(output_folder, book_name)

    
    train_neural_network(output_folder)


def merge_pdfs(folder, output_name):
    pdf_files = sorted(os.listdir(folder), key=lambda x: (int(re.sub('\D', '', x)), x))
    pdf_list = [folder + file for file in pdf_files if file.endswith(".pdf")]

    merger = PdfFileMerger()

    for pdf in pdf_list:
        merger.append(open(pdf, 'rb'))

    output_path = f"{folder}{output_name}_output.pdf"

    with open(output_path, "wb") as fout:
        merger.write(fout)

    merger.close()


def train_neural_network(data_folder):
 
    X, y = prepare_data(data_folder)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


def prepare_data(data_folder):
  
    X = tf.random.normal((100, 10))  # 100 samples with 10 features each
    y = tf.random.uniform((100, 1), 0, 2, dtype=tf.int32)  # Binary labels (0 or 1)
    return X, y

def build_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    book_url = "https://babel.hathitrust.org/cgi/pt?id=txu.059173023561817"
    id_book = re.findall('id=(\w*\.\d*)|$', book_url)[0]

    download_and_merge_pdf(book_url)
