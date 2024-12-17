from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import psycopg2
from sqlalchemy import make_url
from sqlalchemy import create_engine, text
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy.exc import OperationalError
from llama_index.readers.file import FlatReader
from pathlib import Path
from dotenv import load_dotenv
import os
import json

class Data:
    def __init__(self, db_params, storage_context_path):
        self.db_params = db_params
        self.storage_context_path = storage_context_path
    def create_vector_store(self):
        engine = create_engine(f"postgresql://{self.db_params['user']}:{self.db_params['password']}@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['vector_database']}")
        try:
            with engine.connect() as conn:
                # check if the schema exists
                result = conn.execute(text(f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_schema = :schema_name
                    );
                    """), {"schema_name": self.db_params['schema_name']}).scalar()
                if result:
                    print(f"Schema '{self.db_params['schema_name']}' exist, skipping creating table.")
                    # conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA_NAME} CASCADE;"))
                else:
                    print(f"Schema '{self.db_params['schema_name']}' does not exist. Creating it...")

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
                    database=self.db_params['vector_database'],
                    host=self.db_params['host'],
                    password=self.db_params['password'],
                    port=self.db_params['port'],
                    user=self.db_params['user'],
                    table_name=self.db_params['vector_table_name'],  # Use unprefixed name; PGVectorStore adds 'data_' automatically
                    embed_dim=int(self.db_params['embed_dim']),
                    schema_name=self.db_params['schema_name']
                )
                # vector_store._initialize()
                # print(f"Schema '{SCHEMA_NAME}' and table '{VECTOR_TABLE_NAME}' has been created.")
                return vector_store
        except OperationalError as e:
            print(f"Error: {e}")
            print("Make sure the database exists and connection parameters are correr.")
    
    def ingest(self):
        vector_store = self.create_vector_store()
        storage_context = StorageContext.from_defaults(persist_dir=self.storage_context_path, vector_store=vector_store)

        with open('doc_to_id.json', 'r') as f:
            doc_to_id = json.load(f)

        index = load_index_from_storage(storage_context=storage_context, index_id=doc_to_id['sotaysinhvien'])
        return index