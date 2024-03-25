import random

import torch
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from datasets import load_dataset

torch.set_default_device("cuda")


def download_dataset():
    # Download the cosmopedia Openstax split from HuggingFace
    data = load_dataset("HuggingFaceTB/cosmopedia", "openstax", split="train")
    data.to_csv("datasets/Openstax.csv")
    pass


def load_from_csv_datset_and_split():
    loader = CSVLoader(file_path="datasets/Openstax.csv")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    return docs


def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": False},
    )
    return embeddings


def get_llm_model():
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceM4/tiny-random-LlamaForCausalLM",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceM4/tiny-random-LlamaForCausalLM", trust_remote_code=True
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_tensors="pt",
        max_length=512,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )


def main():
    # If the data does not exceed, download the cosmopedia
    # openstax split and save it as csv

    # download_dataset()

    llm_pipeline = get_llm_model()
    embeddings = get_embedding_model()
    print("Loaded the LLM Model and the Embedding Model")

    docs = load_from_csv_datset_and_split()
    print("Generated the Documents from the dataset")

    # Let us sample 1000 documents for the experimentation
    docs = random.sample(docs, 1000)
    vector_db = FAISS.from_documents(docs, embeddings)
    print("Converted the documents into embeddings")

    # Setuo a pipeline for feeding the augmented query to the LLM model
    llm = HuggingFacePipeline(
        pipeline=llm_pipeline,
        model_kwargs={"temperature": 0.7, "max_length": 512},
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_db.as_retriever()
    )

    # Invoke the pipeline with a query
    qa.invoke("Write an educational story for young children.")


if __name__ == "__main__":
    main()
