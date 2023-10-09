import os
import chainlit as cl
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings 
import utils as utils
import importlib
importlib.reload(utils)


ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")



prompt_template = """
Eres un lector experto de documentos desde hace muchos años. Si no sabes la respuesta, simplemente menciona que no la sabes, no intentes inventar una respuesta.
SIEMPRE devuelve todas tus respuestas en Español, es necesario traducir sus respuestas al Español sobre los documentos.
El ejemplo de su respuesta debería ser:

Contexto: {context}
Pregunta: {question}

Sólo devuelva la respuesta útil y nada más.
Respuesta Alternativa:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt


def create_retrieval_qa_bot(
    model_name="./model/all-MiniLM-L6-v2",
    persist_dir=DB_DIR,
    device="cpu",
):
    """
    This function creates a retrieval-based question-answering bot.

    Parameters:
        model_name (str): The name of the model to be used for embeddings.
        persist_dir (str): The directory to persist the database.
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').

    Returns:
        RetrievalQA: The retrieval-based question-answering bot.

    Raises:
        FileNotFoundError: If the persist directory does not exist.
        SomeOtherException: If there is an issue with loading the embeddings or the model.
    """

    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"No directory found at {persist_dir}")

    try:
        embedding_function = SentenceTransformerEmbeddings(
            model_name=model_name, 
            model_kwargs={'device':device})
        
    except Exception as e:
        raise Exception(
            f"Failed to load embeddings with model name {model_name}: {str(e)}"
        )

    vectorStore =  Chroma(persist_directory=persist_dir, embedding_function=embedding_function)

    qa_prompt = (
        set_custom_prompt()
    )  # Assuming this function exists and works as expected

    try:
        qa_chain = utils.createConversationalChain(vectorStore, qa_prompt)
        # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to create retrieval QA chain: {str(e)}")

    return qa_chain


def retrieve_bot_answer(query):
    """
    Retrieves the answer to a given query using a QA bot.

    This function creates an instance of a QA bot, passes the query to it,
    and returns the bot's response.

    Args:
        query (str): The question to be answered by the QA bot.

    Returns:
        dict: The QA bot's response, typically a dictionary with response details.
    """
    qa_bot_instance = create_retrieval_qa_bot()
    bot_response = qa_bot_instance({"query": query})
   
    return bot_response


@cl.on_chat_start
async def initialize_bot():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    qa_chain = create_retrieval_qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to Chat With Documents using Llama2 and LangChain."
    )
    await welcome_message.update()

    cl.user_session.set("chain", qa_chain)


@cl.on_message
async def process_chat_message(message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the user's session,
    sets up a callback handler for the bot's response, and executes the bot's
    call method with the given message and callback. The bot's answer and source
    documents are then extracted from the response.
    """
    qa_chain = cl.user_session.get("chain")
    callback_handler = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    callback_handler.answer_reached = True
    response = await qa_chain.acall(message, callbacks=[callback_handler])
    bot_answer = response["result"]

    #print('response',response)
    cl.Message(bot_answer).send()
#chainlit run app.py -w