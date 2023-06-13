from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
import re
import json
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_data():
    doc_chunks = []
    with open("accounts_to_scrape.json","r") as file:
        ff = json.load(file)
    for k,v in ff.items():
        for itm in v:
            file_path = f"./data/{itm.lower()}.csv"
            df = pd.read_csv(file_path,sep="	")
            data = df['tweet'].to_list()
            data = [re.sub(r'https:\/\/t.co\/[^\s]+','', d.replace("http://", "https://")) + '\n' for d in data if not d.startswith("RT")]
            data = "\n".join(data)
            data = re.sub(r"(?<!\n)\n(?!\n)", " ", data)
            data = re.sub(r"\n{2,}", "\n", data)
            metadata = {
                "category": k,
                "author": df['name'][0],
                "username": itm
            }
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=0
            )
            chunks = text_splitter.split_text(data)
            for i,chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                    "chunk": i,
                    **metadata
                    }
                )
                doc_chunks.append(doc)
    return doc_chunks


if __name__ == "__main__":
	# os.environ["OPENAI_API_KEY"] = ""
	chunked_list = load_data()
	embeddings = OpenAIEmbeddings()
	vector_store = Chroma.from_documents(chunked_list,embeddings,collection_name="f1_tweets", persist_directory="./chroma")
	vector_store.persist()



