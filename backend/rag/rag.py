from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.react import ReActAgent
from llama_index.core import ChatPromptTemplate
from engine import load_llm, load_embedding_model, LLM_PATH, EMBED_DIM, EMBEDDING_MODEL_PATH
from rag.data import Data
# from get_response_synth import get_custom_response_synth
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.prompts.prompts import RefinePrompt, QuestionAnswerPrompt
from llama_index.core.prompts.prompt_type import PromptType
# from llama_index.core.response_synthesizers.factory import get_response_synthesizer
from llama_index.core import get_response_synthesizer
from llama_index.core.llms import ChatMessage, MessageRole


# def get_custom_response_synth(llm) -> BaseSynthesizer:
#     qa_template_str = """
#     ### Instruction and Input:
#     Dựa vào ngữ cảnh/tài liệu sau:
#     {context_str}
#     Hãy trả lời câu hỏi: {query_str}

#     ### Response:
    
#     """
    
#     qa_prompt = QuestionAnswerPrompt(
#         template=qa_template_str,
#         prompt_type=PromptType.QUESTION_ANSWER,
#     )
    
#     return get_response_synthesizer(
#         text_qa_template=qa_prompt,
#         structured_answer_filtering=False,
#     )

class RAG:
    def __init__(self, config):
        self.config = config
        self.embedder = None
        self.llm = None
        self.index = None
        self.chat_engine = None

        text_qa_template_msgs = [
            ChatMessage(
                role=MessageRole.USER,
                content="""
                        ### Instruction and Input:
                        Dựa vào ngữ cảnh/tài liệu sau:
                        {context_str}
                        Hãy trả lời câu hỏi: {query_str}

                        ### Response:
    
                        """
            )
        ]
        self.text_qa_template = ChatPromptTemplate(text_qa_template_msgs)

    def load_embedder(self):
        if not self.embedder:
            self.embedder = load_embedding_model(model_name=EMBEDDING_MODEL_PATH)
            Settings.embed_model = self.embedder
        # return embed_model
    
    def load_llm(self):
        if not self.llm:
            self.llm = load_llm(model_path=LLM_PATH, tokenizer_path=LLM_PATH)
            Settings.llm = self.llm
        
    def ingest(self):
        if not self.index:
            # print(self.config['db_params']['vector_table_name'])
            data = Data(db_params=self.config['db_params'], storage_context_path="storage_context")
            self.load_embedder()
            self.load_llm()
            self.index = data.ingest()
            # print(index.index_id)
        # return index
    
    async def get_chat_engine(self):
        if not self.chat_engine:
            self.ingest()
            self.chat_engine = self.index.as_chat_engine()
            # query_engine_tool = QueryEngineTool(
            #     query_engine=query_engine,
            #     metadata=ToolMetadata(
            #         name="sotaysinhvien",
            #         description="Cung cấp thông tin về sổ tay sinh viên của trường Đại học Phenikaa"
            #     ),

            # )
        # ToolMetadata()
            # self.chat_engine = ReActAgent.from_tools(
            #     [query_engine_tool],
            #     llm=Settings.llm,
            #     verbose=True,
            # )
        return self.chat_engine