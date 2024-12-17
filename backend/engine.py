# from llama_index.packs.raft_dataset import RAFTDatasetPack
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import ServiceContext, set_global_service_context, Settings

from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  BitsAndBytesConfig,
  pipeline,
)
import psycopg2
from sqlalchemy import make_url
# from sentence_transformers import SentenceTransformer
import torch
# import wandb
from sqlalchemy import create_engine, text
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy.exc import OperationalError

from llama_index.core.node_parser import MarkdownNodeParser #, MarkdownElementNodeParser
# from llama_index.core import SimpleDirectoryReader
# from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
import os
import json

EMBEDDING_MODEL_PATH = "models/embedding_model/mpnet-base-v2"
LLM_PATH = "models/llms/vietcuna-3b-v2"
TOKENIZER_PATH = "tokenizers/vi-gemma-function-calling"
load_dotenv("./.env")

POSTGRES_USER=os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD=os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST=os.getenv("POSTGRES_HOST")
POSTGRES_PORT=os.getenv("POSTGRES_PORT")
VECTOR_DATABASE=os.getenv("VECTOR_DATABASE")
SCHEMA_NAME=os.getenv("SCHEMA_NAME")
EMBED_DIM=os.getenv("EMBED_DIM")
VECTOR_TABLE_NAME=os.getenv("VECTOR_TABLE_NAME")

def load_embedding_model(model_name: str):
    return HuggingFaceEmbedding(model_name=model_name, device='cpu')

def load_llm(model_path: str, tokenizer_path: str) -> HuggingFaceLLM:
    # Kích hoạt mô hình cơ sở độ chính xác 4 bit đang tải
    # use_4bit = True

    # # Tính toán kiểu dữ liệu cho các mô hình cơ sở 4 bit
    # bnb_4bit_compute_dtype = "float16"

    # # Kiểu lượng tử hóa (fp4 hoặc nf4)
    # bnb_4bit_quant_type = "nf4"

    # # Kích hoạt lượng tử hóa lồng nhau cho cơ sở 4 bit mô hình (lượng tử hóa kép)
    # use_nested_quant = False

    # compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    # bnb_config = BitsAndBytesConfig(
    # load_in_4bit=use_4bit,
    # bnb_4bit_quant_type=bnb_4bit_quant_type,
    # bnb_4bit_compute_dtype=compute_dtype,
    # bnb_4bit_use_double_quant=use_nested_quant,
    # )
    llm = HuggingFaceLLM(model_name=model_path,
                        tokenizer_name=tokenizer_path,
                        #  model_kwargs={"quantization_config": bnb_config},
                        device_map="auto"
                        )

    return llm

# embed_model = load_embedding_model(model_name=EMBEDDING_MODEL_NAME)
# Settings.embed_model = embed_model

# load sotaysinhvien
# documents = FlatReader().load_data(Path("data/sotay.md"))

def creating_database(connection_string: str):
    connection_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}"
    db_name = VECTOR_DATABASE
    conn = psycopg2.connect(connection_string)
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {db_name}")
        c.execute(f"CREATE DATABASE {db_name}")

def load_documents(docpath:str):
   return FlatReader().load_data(Path(docpath))

def create_vector_store():
    engine = create_engine(f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{VECTOR_DATABASE}")
    try:
        with engine.connect() as conn:
            # check if the schema exists
            result = conn.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = :schema_name
                );
                """), {"schema_name": SCHEMA_NAME}).scalar()
            if result:
                print(f"Schema '{SCHEMA_NAME}' exist, skipping creating table.")
                # conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA_NAME} CASCADE;"))
            else:
                print(f"Schema '{SCHEMA_NAME}' does not exist. Creating it...")

            # # Check if the table exists
            # result = conn.execute(text(f"""
            #     SELECT EXISTS (
            #         SELECT FROM information_schema.tables 
            #         WHERE table_schema = :schema_name AND table_name = :table_name
            #     );
            # """), {"table_name": table_name, "schema_name": db_params["schema"]}).scalar()
            # print(result)
            # if result:
            #     print(f"Table '{table_name}' exists. Cleaning it...")
            #     conn.execute(text(f"TRUNCATE TABLE {table_name};"))
            # else:
            #     print(f"Table '{table_name}' does not exist. Creating it...")

            # Create a new table for vector store
            vector_store = PGVectorStore.from_params(
                database=VECTOR_DATABASE,
                host=POSTGRES_HOST,
                password=POSTGRES_PASSWORD,
                port=POSTGRES_PORT,
                user=POSTGRES_USER,
                table_name=VECTOR_TABLE_NAME,  # Use unprefixed name; PGVectorStore adds 'data_' automatically
                embed_dim=int(EMBED_DIM),
                schema_name=SCHEMA_NAME
            )
            # vector_store._initialize()
            # print(f"Schema '{SCHEMA_NAME}' and table '{VECTOR_TABLE_NAME}' has been created.")
            return vector_store
    except OperationalError as e:
        print(f"Error: {e}")
        print("Make sure the database exists and connection parameters are correr.")

def load_index(persist_dir="./storage_context"):
    vector_store = create_vector_store()
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir, vector_store=vector_store)

    with open('doc_to_id.json', 'r') as f:
        doc_to_id = json.load(f)


    index = load_index_from_storage(storage_context=storage_context, index_id=doc_to_id['sotaysinhvien'])
    print(index.index_id)
    # index = VectorStoreIndex.from_documents(
    #     documents, storage_context=storage_context, show_progress=True, transformations=[MarkdownNodeParser()]
    # )
    return index
# print(type(EMBED_DIM))
# query = "Trường đại học PHENIKAA được thành lập vào năm nào?"
