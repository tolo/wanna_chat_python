import os
import requests
from urllib.parse import urlencode

from dotenv import load_dotenv, find_dotenv

from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms import Ollama



persist_directory = './vector_store/'

class ChatService:
    def __init__(self):
        _ = load_dotenv(find_dotenv()) # read local .env file

        isExist = os.path.exists(persist_directory)
        self.vector_db = self.__setup_vector_db()

        print(f"Persist directory exists: {isExist}")
        if not isExist:
            self.ingest()

        self.llm = self.__setup_llm()

        self.chat_store = {}
        self.qa_chain = self.__setup_qa_chain()

        # testMsg = self.add_message("test_id", "Hello")
        # print(f"Test message: {testMsg}")
        #
        # print(f"Test history: {self.get_history('test_id')}")


    def get_history(self, chat_id) -> list[dict]:
        history = self.__get_session_history(chat_id)
        print(f"History for {chat_id} : {history}")
        if len(history.messages) == 0:
            history.add_message(AIMessage("Hello, I'm your friendly htmx AI-teacher! Ask away!"))
            self.chat_store[chat_id] = history

        return [{"type": msg.type, "text": msg.content} for msg in history.messages]

    def ask_question(self, chat_id, question) -> str:
        print(f"Question from {chat_id} : {question}")

        sim_result = self.vector_db.max_marginal_relevance_search(question, k=5)
        print(f"Similarity search result: {sim_result}")

        return self.qa_chain.invoke(
            {"input": question},
            config={
                "configurable": {"session_id": chat_id}
            },
        )["answer"]

    def ingest(self):
        print("Ingesting...")

        docs = self.__load_and_split()

        print(f"Ingesting documents ({len(docs)})...")

        self.vector_db.add_documents(docs)
        self.vector_db.persist()

    def __load_and_split(self) -> list[Document]:
        # Load files
        # loaders = [
        #     TextLoader("./data/some_file.txt"),
        # ]

        # Load web pages
        loaders = [
            WebBaseLoader("https://hypermedia.systems/hypermedia-systems/"),
            WebBaseLoader("https://v1.htmx.org/docs/"),
            WebBaseLoader("https://v1.htmx.org/essays/hypermedia-on-whatever-youd-like/"),
            WebBaseLoader("https://v1.htmx.org/essays/a-response-to-rich-harris/"),
            WebBaseLoader("https://v1.htmx.org/essays/when-to-use-hypermedia/"),
            WebBaseLoader("https://v1.htmx.org/essays/does-hypermedia-scale/"),
            WebBaseLoader("https://v1.htmx.org/essays/spa-alternative/"),
        ]

        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        print("Initializing text splitter...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=64
        )

        splits = splitter.split_documents(docs)
        print(f"Documents split into {len(splits)} chunks")
        return splits

    def __setup_vector_db(self) -> Chroma:
        openai_embeddings_model: str = "text-embedding-3-small"
        print(f"Using OpenAI embeddings model: {openai_embeddings_model}")
        embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model=openai_embeddings_model)

        # #hf_embeddings_model: str = "Salesforce/SFR-Embedding-Mistral" # 7B params, 26GB mem use, 4096 dim, 67.56 avg score
        # hf_embeddings_model: str = "WhereIsAI/UAE-Large-V1" # #335M params, 1.25GB mem use, 1024 dim, 64.64 avg score
        # #hf_embeddings_model: str = "avsolatorio/GIST-all-MiniLM-L6-v2" # 23M params, 0.08GB mem use, 384 dim, 59 avg score
        #
        # #os.environ["TOKENIZERS_PARALLELISM"] = "true"
        # print(f"Using HuggingFace embedding model: {hf_embeddings_model}")
        # embeddings = HuggingFaceEmbeddings(
        #     model_name=hf_embeddings_model,
        #     cache_folder="./model_cache",
        #     #show_progress=True,
        # )

        return Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

    def __setup_llm(self) -> BaseChatModel:
        # meta-llama/Meta-Llama-3-8B-Instruct

        # OpenAI
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            #openai_api_base="...",
            #model_name="gpt-3.5-turbo",
            model_name="gpt-4-turbo",
            temperature=0.1,
        )

        # Anthropic (Claude)
        from langchain_anthropic import ChatAnthropic
        # llm = ChatAnthropic(
        #     model='claude-3-opus-20240229',
        #     temperature=0.1,
        # )

        # Ollama
        llm = Ollama(model="llama3", verbose=True)

        # Models from HuggingFace
        # from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        # llm = HuggingFacePipeline.from_model_id(
        #     model_id="microsoft/phi-2",
        #     #model_id="macadeliccc/laser-dolphin-mixtral-2x7b-dpo",
        #     #model_id="mistralai/Mistral-7B-Instruct-v0.2",
        #     #model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        #     task="text-generation",
        #     pipeline_kwargs={"max_new_tokens": 10},
        # )

        # GPT4All (if running locally, and GPT4All is installed)
        # llm = GPT4All(
        #     model="/Users/tobias/Library/Application Support/nomic.ai/GPT4All/Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf",
        #     max_tokens=2048,
        # )
        print(f"Initialized LLM {llm}")
        return llm

    def __setup_qa_chain(self) -> RunnableWithMessageHistory:
        ### Contextualize question ###
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.vector_db.as_retriever(), contextualize_q_prompt
        )

        ### Answer question ###
        qa_system_prompt = """
        You are an experienced software architect, expert at answering questions based on provided sources and giving helpful advice. 
        Using the below provided context, answer the user's question to the best of your ability using only the resources provided. 
        Be concise and stick to the subject! If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        When responding, do it in the style of Erlich Bachman from the TV series Silicon Valley.  
        
        Use the following pieces of retrieved context to answer the question:
        {context}
        """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        ### Statefully manage chat history ###
        return RunnableWithMessageHistory(
            rag_chain,
            self.__get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
            #kwargs={"verbose": True},
            chain_type_kwargs={"verbose": True }
        )

    def __get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.chat_store:
            self.chat_store[session_id] = ChatMessageHistory()
        return self.chat_store[session_id]
