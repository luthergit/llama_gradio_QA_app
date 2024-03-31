#pull in python image as a basis for your docker image
FROM --platform=arm64 python:3.10.13

#create a working dir called 'app' in my docker
WORKDIR /app

#Get the local llama model
RUN wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true
#copy the relevant files in the app folder
COPY chat_with_documents.py requirements.txt /app/

#copy the relevant files in llama folder in the app folder 
RUN mkdir /app/llama_model_7b 
# RUN mv llama-2-7b-chat.Q4_K_M.gguf /app/llama_model_7b

#install the requirements from the txt file 
RUN pip install -r requirements.txt

RUN mv "llama-2-7b-chat.Q4_K_M.gguf?download=true" "llama-2-7b-chat.Q4_K_M.gguf"
RUN mv llama-2-7b-chat.Q4_K_M.gguf /app/llama_model_7b

# export and exposes to local server in docker container
EXPOSE 8000

#run the file
CMD [ "python","chat_with_documents.py"]

#Next step: command shift p