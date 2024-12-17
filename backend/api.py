from fastapi import FastAPI, WebSocket
from llama_index.core import VectorStoreIndex, Settings, load_index_from_storage
from llama_index.readers.file import FlatReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.agent.react import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core import ChatPromptTemplate
from rag.data import Data
from rag.rag import RAG
from dotenv import load_dotenv
import os
from pydantic import BaseModel

class StreamedMessage(BaseModel):
    content: str


load_dotenv("./.env")
# POSTGRES_USER=os.getenv("POSTGRES_USER")
# POSTGRES_PASSWORD=os.getenv("POSTGRES_PASSWORD")
# POSTGRES_HOST=os.getenv("POSTGRES_HOST")
# POSTGRES_PORT=os.getenv("POSTGRES_PORT")
# VECTOR_DATABASE=os.getenv("VECTOR_DATABASE")
# SCHEMA_NAME=os.getenv("SCHEMA_NAME")
# EMBED_DIM=os.getenv("EMBED_DIM")
# VECTOR_TABLE_NAME=os.getenv("VECTOR_TABLE_NAME")
config = {"db_params":  {
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT"),
        "vector_database": os.getenv("VECTOR_DATABASE"),
        "schema_name": os.getenv("SCHEMA_NAME"),
        "embed_dim": os.getenv("EMBED_DIM"),
        "vector_table_name": os.getenv("VECTOR_TABLE_NAME")
    }
}

app = FastAPI()

# data = Data()
rag = RAG(config=config)

# print(Settings.llm)
@app.get("/")
def read_root():
    return {"message": "Welcome to the LlamaIndex-powered chatbot! Connect via WebSocket to chat"}

@app.websocket("/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    chat_engine = await rag.get_chat_engine()
    
    # await websocket.send_text("Hello client!")
    
    # try:
    while True:
        user_input = await websocket.receive_text()
        # print(type(data))
        user_message = str(user_input)
        streaming_chat_response: StreamingAgentChatResponse = (
            await chat_engine.astream_chat(user_message)
        )
        # messsage = [ChatMessage(role="user", content=data)]
        # streaming_chat_response = await chat_engine.astream_chat(data)
        response_str = ""
        async for text in streaming_chat_response.async_response_gens():
            response_str += text
            await websocket.send_text(StreamedMessage(content=response_str))
    # except Exception as e:
    #     print(f"Error: {e}")
    #     await websocket.close()

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)