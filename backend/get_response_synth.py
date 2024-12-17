from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.prompts.prompts import RefinePrompt, QuestionAnswerPrompt
from llama_index.core.prompts.prompt_type import PromptType
# from llama_index.core.response_synthesizers.factory import get_response_synthesizer
from llama_index.core import get_response_synthesizer
def get_custom_response_synth() -> BaseSynthesizer:
    qa_template_str = """
    ### Instruction and Input:
    Dựa vào ngữ cảnh/tài liệu sau:
    {context_str}
    Hãy trả lời câu hỏi: {query_str}

    ### Response:
    
    """
    
    qa_prompt = QuestionAnswerPrompt(
        template=qa_template_str,
        prompt_type=PromptType.QUESTION_ANSWER,
    )
    
    return get_response_synthesizer(
        text_qa_template=qa_prompt,
        structured_answer_filtering=False,
    )