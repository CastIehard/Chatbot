# This Script is used to run the chatbot in the console. It uses the LLM and RAG model to generate responses to user queries.

# LLM Outputs are green, hidden prompts are red, souces are blue

import time
import pickle
import faiss
import numpy as np
from langchain_community.llms.llamafile import Llamafile
from langchain_community.embeddings import LlamafileEmbeddings
import subprocess
import os
import re

# Configuration
REWRITE_RAG_QUERIES = False
ENABLE_RAG = False
RAG_THRESH = 0.2
RAG_N_RETURN = 1
LLM_NAME = "models\\qwen2-1_5b-instruct-q4_k_m.gguf"
LLM_PORT = 1112
EMBEDDING_NAME = "models\\qwen2-1_5b-instruct-q4_k_m.gguf"
EMBEDDING_PORT = 1111
SYSTEM_PROMPT = "Dies ist ein Gespräch zwischen User und Chatbot, einem freundlichen Chatbot der bei der allen Fragen hilft."
HISTORY_MSG_N = 0
STOP_WORDS = ["Chatbot:", "User:", "</s>", "<|eot_id|>"]


def start_server(
    model_name: str,
    port: int,
    embedding: bool = False,
    threads: int = os.cpu_count() - 2,
    verbose: bool = False,
    browser: bool = False,
    output: bool = False,
) -> None:
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
        "llamafile-0.8.13.exe",
        "--server",
        "--model",
        model_name,
        "--port",
        str(port),
        "--threads",
        str(threads),
    ]

    if embedding:
        command.append("--embedding")

    if verbose:
        command.append("--verbose")

    if not browser:
        command.append("--nobrowser")

    # Print the command as string
    print(f"\033[33m\n\nStarting the Server:\n\n{' '.join(command)}\033[0m")

    # Determine the stdout and stderr parameters based on the output flag
    if output:
        subprocess.Popen(command)
    else:
        with open(os.devnull, "w") as devnull:
            subprocess.Popen(command, stdout=devnull, stderr=devnull)


# Initialize the LLM
def initialize_llm():
    start_server(model_name=LLM_NAME, port=LLM_PORT, embedding=False)
    return Llamafile(
        base_url=f"http://localhost:{LLM_PORT}",
        streaming=True,
        temperature=0,
        top_k=40,
        top_p=0.95,
        min_p=0.05,
        n_predict=-1,
        n_keep=0,
        tfs_z=1,
        typical_p=1,
        repeat_penalty=1.1,
        repeat_last_n=64,
        penalize_nl=True,
        presence_penalty=0,
        frequency_penalty=0,
        mirostat=2,
        mirostat_tau=5,
        mirostat_eta=0.1,
    )


# Initialize RAG
def initialize_rag():
    start_server(model_name=EMBEDDING_NAME, port=EMBEDDING_PORT, embedding=True)
    embedder = LlamafileEmbeddings(base_url=f"http://localhost:{EMBEDDING_PORT}")

    with open("Index/faiss_index.pkl", "rb") as f:
        all_chunks, vectors_np, index = pickle.load(f)
    return embedder, all_chunks, vectors_np, index


# Query Index
def query_index(
    embedder, index, all_chunks, query_text, n_return, RAG_THRESH=RAG_THRESH
):
    # Embed the query text
    query_vector = embedder.embed_query(query_text)
    query_vector = np.array([query_vector]).astype("float32")
    faiss.normalize_L2(query_vector)

    # Search the index for the closest matches
    D, I = index.search(query_vector, index.ntotal)  # Search entire index

    cosine_distances = 1 - D
    results = [(all_chunks[I[0][i]], cosine_distances[0][i]) for i in range(len(I[0]))]

    # Sort results by cosine distance (ascending order)
    results = sorted(results, key=lambda x: x[1])

    # Filter based on RAG_THRESH and return the top n_return results
    filtered_results = [
        (file_path, text_chunk, dist)
        for (file_path, text_chunk), dist in results
        if dist < RAG_THRESH
    ]

    return filtered_results[:n_return]


# Generate Prompt
def generate_prompt(context, conversation, query):
    return f"""
{SYSTEM_PROMPT}
{context}
User: Hallo, bitte antworte mir in Deutsch!
Chatbot: Gerne, wie kann ich dir helfen?
{conversation}
User: {query}
Chatbot: """


# Stream Response
def stream_response(llm, prompt, output=True):
    answer = ""
    for chunk in llm.stream(prompt, stop=STOP_WORDS):
        # check if junk is "Chatbot:" then delete it
        if output:
            print(f"\033[92m{chunk}\033[0m", end="")
        answer += chunk
    return answer


# Function to generate a query based on the chat history
def generate_query_from_history(context, llm, query):
    prompt = f"""
    This is a conversation between a user and Chatbot, a friendly chatbot. Chatbot rephrases questions so they can be understood without any prior context. If the context is relevant to the question, Chatbot incorporates the context into the question to make it clear. If the context is not relevant, Chatbot returns the question as is.

    User: Please rephrase the following questions so they can be understood independently of the context. If the context is relevant, include it in the question. If the context is not relevant, just return the question.
    Chatbot: Sure, send me the context and the original question.

    User: Context: "What is an apple?" Original question: "What color is it?"
    Chatbot: What color is an apple?

    User: Context: "The house is red and big." Original question: "How do I drive a car correctly?"
    Chatbot: How do I drive a car correctly?

    User: Context: "{context}" Original question: "{query}"
    Chatbot:
    """
    print(f"\033[33m\nPrompt: {prompt}\033[0m")

    generated_query = stream_response(llm, prompt, output=False)

    print(f"\033[33m\nGenerated Query: {generated_query}\033[0m")
    return generated_query.strip()


# Main Chat Loop
def main_chat_loop():
    llm = initialize_llm()
    embedder, all_chunks, vectors_np, index = (None, None, None, None)

    if ENABLE_RAG:
        embedder, all_chunks, vectors_np, index = initialize_rag()

    chat_history = []
    conversation = None

    while True:
        query = input("\n\nFrage: ")
        start_time = time.time()

        conversation = "\n".join(chat_history)

        if len(chat_history) > 0 and REWRITE_RAG_QUERIES:
            context = chat_history[-2]
            context = re.sub(r"(?i)User:\s*", "", context).strip()
            generated_query = generate_query_from_history(context, llm, query)
        else:
            generated_query = query

        if len(chat_history) > 2 * HISTORY_MSG_N:
            chat_history = chat_history[-2 * HISTORY_MSG_N :]

        context = ""
        file_paths = []  # Initialize list to store file paths
        if ENABLE_RAG:
            results = query_index(
                embedder, index, all_chunks, generated_query, RAG_N_RETURN
            )
            if len(results) > 0:
                context = "\n\n".join([text for file_path, text, _ in results])
                context = f"\nBenutze folgende Informationen, um auf die Benutzeranfrage zu antworten: \n\n###\n{context}\n###\n\n"
                file_paths = list(
                    set(file_path for file_path, _, _ in results)
                )  # Extract and deduplicate file paths
            else:
                context = "Es wurden keine Informationen gefunden, um auf die Benutzeranfrage zu antworten. Teile dies dem Benutzer mit und bitte ihn, die Frage zu präzisieren. ANTWORTE NICHT AUF DIE FRAGE!"

        prompt = generate_prompt(context, conversation, generated_query)
        print(f"\033[33m\nPrompt: {prompt}\033[0m\n\n")
        print(f"\033[92mChatbot:\033[0m", end="")
        answer = stream_response(llm, prompt)

        if file_paths:
            print(f"\033[94mQuelle:{'\n'.join(file_paths)}\033[0m")

        chat_history.append(f"User: {query}")
        chat_history.append(f"Chatbot: {answer.strip()}")

        time_taken = time.time() - start_time
        characters_per_second = len(answer) / time_taken
        print(
            f"\033[33m\n\nCharacters per second: {characters_per_second} Time taken: {time_taken}\033[0m"
        )


if __name__ == "__main__":
    main_chat_loop()
