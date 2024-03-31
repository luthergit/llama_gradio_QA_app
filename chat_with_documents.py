import gradio as gr
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma
import os

model_path = "./llama_model_7b/llama-2-7b-chat.Q4_K_M.gguf"

def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)

    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)

    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        print(f'Loading {file}')
        loader = TextLoader(file)

    else:
        print('Document format is not supported!')
        return None
        
    data = loader.load()
    return data

def chunk_data(data, chunk_size=156, chunk_overlap = 20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter #general for generic text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks, chunk_size=512):

    embeddings = LlamaCppEmbeddings(model_path=model_path)
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def ask_and_get_answer(vector_store, q, k=3):

    from langchain.chains import RetrievalQA
    from langchain_community.llms import LlamaCpp
    from langchain import PromptTemplate, LLMChain
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import (
        StreamingStdOutCallbackHandler,
    )
    # model_path = "./llama_model/nous-hermes-llama2-13b.Q4_0.gguf" # <-------- enter your model path here 
    model_path = "./llama_model_7b/llama-2-7b-chat.Q4_K_M.gguf" # <-------- enter your model path here 


    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 2048  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_ctx = 500

    ## for gpu
    # llm = LlamaCpp(
    #     model_path=model_path,
    #     n_gpu_layers=n_gpu_layers,
    #     n_batch=n_batch,
    #     callback_manager=callback_manager,
    #     verbose=False,
    #     n_ctx=n_ctx,
    #     temperature=0
    # )

    # Uncomment the code below if you want to run inference on CPU
    llm = LlamaCpp(
        model_path=model_path,
        # n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=False,
        n_ctx=n_ctx,
        temperature=0
    )

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    answer = chain.run(q)
    return answer

def upload_and_process_file(file, chunk_size=512):

    # Open the file and read its contents
    data = load_document(file)
    
    # print("Document data:", data)  # Debugging print statement
    chunks = chunk_data(data, chunk_size=chunk_size)
    # print("Chunks:", chunks)  # Debugging print statement
    vector_store = create_embeddings(chunks, chunk_size=chunk_size)
    return vector_store


def qa_interface(file, question, chunk_size=512, k=3, previous_questions_and_answers=[]):
    answer = None
    
    if file is not None:
        vector_store = upload_and_process_file(file, chunk_size=chunk_size)
    
    if question:
        answer = ask_and_get_answer(vector_store, question, k)
        
        # Append the question and answer to the beginning of the list of previous questions and answers
        previous_questions_and_answers.insert(0, (question, answer))
        
    # Concatenate all previous questions and answers into a single string for display
    previous_qa_text = "\n".join([f"Question: {q}\nAnswer: {a}\n{'-'*100}" for q, a in previous_questions_and_answers])
    
    return answer, previous_qa_text, previous_questions_and_answers



title = "LLM Question and Answering Application V1🥋"
description = "LLAMA Question and Answering Chatbot"
iface = gr.Interface(fn=qa_interface, 
                     inputs=[gr.File(label="Upload a file"), 
                             gr.Textbox(label="Ask a question about the content of your file:"),
                             gr.Number(label="Chunk size:", value=512, minimum=100, maximum=2048), 
                             gr.Number(label="K value:", value=3, minimum=1, maximum=20)], 
                     outputs=[gr.Textbox(label="LLM Answer:"), 
                              gr.Textbox(label="Chat History:")],
                     title = title,
                    description = description
                    )

iface.launch(server_name = "0.0.0.0", server_port = 8000)
