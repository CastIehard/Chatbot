import os
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from langchain_community.embeddings import LlamafileEmbeddings
import re
from PyPDF2 import PdfReader
import xml.etree.ElementTree as ET
import subprocess


EMBEDDING_NAME = "models\\gte-large.Q8_0.gguf"
EMBEDDING_PORT = 1111
FILES_PATH = 'Files'
INDEX_PATH = 'Index/faiss_index.pkl'


def start_server(model_name: str, port: int, embedding: bool = False, threads: int = os.cpu_count()-2, verbose: bool = False, browser: bool = False, output: bool = False) -> None:
    """
    Starts the LlamaFile server with the specified model and configuration.

    Parameters:
    model_name (str): The path to the model file to be used by the server.
    port (int): The port number on which the server will run.
    embedding (bool): Flag indicating whether to enable embedding functionality. Default is False.
    threads (int): Number of threads to use during generation. Default is the number of available CPU cores.
    verbose (bool): Flag indicating whether to print verbose output. Default is False.
    browser (bool): Flag indicating whether to open the server in a web browser. Default is False.
    output (bool): Flag indicating whether to print the server output. Default is False.

    The function starts the server in a background process and suppresses the output.
    No value is returned from this function.
    """

    command = [
        "llamafile-0.8.12.exe",
        "--server",
        "--model", model_name,
        "--port", str(port),
        "--threads", str(threads),
    ]

    if embedding:
        command.append("--embedding")

    if verbose:
        command.append("--verbose")

    if not browser:
        command.append("--nobrowser")

    # Print the command as string
    print(f"\n\nStarting the Server:\n\n{' '.join(command)}")

    # Determine the stdout and stderr parameters based on the output flag
    if output:
        subprocess.Popen(command)
    else:
        with open(os.devnull, 'w') as devnull:
            subprocess.Popen(command, stdout=devnull, stderr=devnull)

def extract_text_from_file(file_path, file_format):
    match file_format:
        case '.pdf':
            text = ''
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text()
            return text
        case '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        case '.xml':
            tree = ET.parse(file_path)
            root = tree.getroot()
            text = ET.tostring(root, encoding='utf-8', method='text').decode('utf-8')
            return text
        case _:
            raise ValueError(f"Unsupported file format: {file_format}")

def split_text(text, file_format, max_chunk_size=5000, overlap_size=500):
    def further_split(chunks, symbol):
        new_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size:
                sub_chunks = re.split(symbol, chunk)
                for sub_chunk in sub_chunks:
                    if len(sub_chunk) > max_chunk_size:
                        for i in range(0, len(sub_chunk), max_chunk_size - overlap_size):
                            new_chunks.append(sub_chunk[i:i + max_chunk_size])
                    else:
                        new_chunks.append(sub_chunk)
            else:
                new_chunks.append(chunk)
        return new_chunks

    if file_format == '.pdf':
        # Try to split by chapter patterns
        chunks = re.split(r'\n\d+\s', text)
        if any(len(chunk) > max_chunk_size for chunk in chunks):
            chunks = further_split(chunks, r'\n\d+\.\d+\s')
    
    elif file_format == '.md':
        chunks = text.split('# ')
        if any(len(chunk) > max_chunk_size for chunk in chunks):
            chunks = further_split(chunks, '## ')
        if any(len(chunk) > max_chunk_size for chunk in chunks):
            chunks = further_split(chunks, '\n')
    elif file_format == '.xml':
        # Split by <section> tags first
        chunks = re.split(r'(?=<section)', text)
        if any(len(chunk) > max_chunk_size for chunk in chunks):
            chunks = further_split(chunks, '(?=<subsection)')
        if any(len(chunk) > max_chunk_size for chunk in chunks):
            chunks = further_split(chunks, '(?=<title>)')
        if any(len(chunk) > max_chunk_size for chunk in chunks):
            chunks = further_split(chunks, '(?=<para>|<listitem>)')

    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            for i in range(0, len(chunk), max_chunk_size - overlap_size):
                final_chunks.append(chunk[i:i + max_chunk_size])
        else:
            final_chunks.append(chunk)
    return final_chunks

def get_all_file_paths(FILES_PATH, extensions):
    file_paths = []
    for root, _, files in os.walk(FILES_PATH):
        for file in files:
            if file.endswith(extensions):
                file_paths.append(os.path.join(root, file))
    return file_paths

def create_faiss_index(vectors):
    vectors_np = np.array(vectors).astype('float32')
    faiss.normalize_L2(vectors_np)
    index = faiss.IndexFlatIP(vectors_np.shape[1])
    index.add(vectors_np)
    return vectors_np, index

def save_index(texts_with_paths, vectors, index, INDEX_PATH):
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, 'wb') as f:
        pickle.dump((texts_with_paths, vectors, index), f)

def main():
    # Start the embedder server
    start_server(model_name=EMBEDDING_NAME, port=EMBEDDING_PORT, embedding=True, output=False)
    
    # Initialize the embedder
    embedder = LlamafileEmbeddings(base_url=f"http://localhost:{EMBEDDING_PORT}")
    
    # Get all file paths
    file_paths = get_all_file_paths(FILES_PATH, extensions=('.pdf', '.md', '.xml'))
    
    print(f"\nFound {len(file_paths)} Files\n")

    texts = {}
    for file_path in tqdm(file_paths, desc="Extracting texts from files"):
        file_format = os.path.splitext(file_path)[1]
        text = extract_text_from_file(file_path, file_format)
        chunks = split_text(text, file_format)
        
        # Save chunks along with their file path
        texts[file_path] = chunks
        print(f"\nExtracted Text from {file_path}")

    print(f"\nDone extracting, start Embedding using {EMBEDDING_NAME}")
    
    # Embed the texts
    all_vectors = []
    all_texts_with_paths = []

    for file_path, chunks in tqdm(texts.items(), desc="Embedding texts"):
        for chunk in chunks:
            vector = embedder.embed_query(chunk)
            all_vectors.append(vector)
            # Store the chunk text along with its file path
            all_texts_with_paths.append((file_path, chunk))

    # Create FAISS index
    vectors_np, index = create_faiss_index(all_vectors)
    
    # Save the FAISS index, texts, and file paths
    save_index(all_texts_with_paths, vectors_np, index, INDEX_PATH)
    print("FAISS index created and saved successfully.")

if __name__ == "__main__":
    main()