"""
Entity Extraction Module
========================

This module provides a function to extract key entities from either plain text
or conversational text using a LangChain-compatible chat model. The output is
a structured list of entities, including subjects, objects, and participants.

Classes:
--------
- Output: Pydantic model defining the structure of the extraction output.

Functions:
----------
- get_entities: Extracts entities from input text or conversation using a language model.
"""

from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel


class Output(BaseModel):
    """
    Structured output model for extracted entities.
    """
    entities: List[str] = Field(desc="THOROUGH list of key entities")


PROMPT_ENTITIES = """Extract key entities from the source text.
Extracted entities are subjects or objects.
This is for an extraction task, please be THOROUGH and accurate to the reference text.
<Source Text>
{SOURCE_TEXT}
</Source Text>
<ADDITIONAL CONTEXT>
{CONTEXT}
</ADDITIONAL CONTEXT>
"""

PROMPT_ENTITIES_CONVERSATION = """Extract key entities from the conversation. Extracted entities are
subjects or objects. Consider both explicit entities and participants in the conversation.
This is for an extraction task, please be THOROUGH and accurate.
<Source Text>
{SOURCE_TEXT}
</Source Text>
<ADDITIONAL CONTEXT>
{CONTEXT}
</ADDITIONAL CONTEXT>
"""


def get_entities(llm: BaseChatModel,
                 input_data: str,
                 is_conversation: bool = False,
                 context: str="") -> List[str]:
    """
    Extract key entities from the provided text or conversation.

    Uses a structured output parser with a LangChain-compatible chat model to extract
    entities such as subjects, objects, or conversation participants from the input.

    :param llm: A chat-based language model implementing the 
                BaseChatModel interface (e.g., ChatOpenAI).
    :type llm: BaseChatModel
    :param input_data: The source text or conversation string from which to extract entities.
    :type input_data: str
    :param is_conversation: If True, the input is treated as a conversation; 
                            otherwise, as plain text.
    :type is_conversation: bool, optional
    :return: A list of extracted entity strings.
    :rtype: List[str]
    """
    prompt_text = PROMPT_ENTITIES_CONVERSATION if is_conversation else PROMPT_ENTITIES
    prompt_template = PromptTemplate(
        input_variables=["SOURCE_TEXT", "CONTEXT"], template=prompt_text)
    prompt = prompt_template.format(SOURCE_TEXT=input_data, CONTEXT=context)
    model = llm.with_structured_output(Output)
    result = model.invoke(prompt)
    return result.entities
