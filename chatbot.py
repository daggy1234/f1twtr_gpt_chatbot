from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
import os

def make_chain():
    model = ChatOpenAI(temperature="0")
    embedding = OpenAIEmbeddings()

    vector_store = Chroma(
        embedding_function=embedding,
        collection_name="f1_tweets", 
        persist_directory="./chroma"
    )

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        # verbose=True,
    )


if __name__ == "__main__":
    # os.environ["OPENAI_API_KEY"] = ""
    chain = make_chain()
    chat_history = []

    while True:
        question = input("Question: ")
        # I don't want to use history as it fixates and expects every question to be a followup
        response = chain({"question": question, "chat_history": []})
        answer = response["answer"]
        source = response["source_documents"]
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        print("\n\nSources:\n")
        for document in source:
            print(f"Author: {document.metadata['author']}")
        print(f"Answer: {answer}")
