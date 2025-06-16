"""
Module for extracting subject-predicate-object relations from text or conversation using a
LangChain-compatible chat model with structured output parsing.

Supports strict entity-typed extraction with fallback and optional post-correction to ensure
subjects and objects match known entities.

Main function: `get_relations(...)` → returns List of (subject, predicate, object) triples.
"""

from typing import List, Tuple, Literal
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain.prompts import PromptTemplate


BASE_RELATION_PROMPT = """Extract subject-predicate-object` triples from the source text.
Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
This is for an extraction task, please be thorough, accurate, and faithful to the reference text.

<SOURCE TEXT>
{SOURCE_TEXT}
</SOURCE TEST>

<ENTITIES LIST>
{ENTITIES}
</ENTITIES LIST>

<ADDITIONAL CONTEXT>
{CONTEXT}
</ADDITIONAL CONTEXT>
"""


BASE_RELATION_CONVERSATION_PROMPT = """Extract subject-predicate-object triples from the
conversation, including:
1. Relations between concepts discussed
2. Relations between speakers and concepts (e.g. user asks about X)
3. Relations between speakers (e.g. assistant responds to user)
Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
This is for an extraction task, please be thorough, accurate, and faithful to the reference text. 

<CONVERSATION>
{SOURCE_TEXT}
</CONVERSATION>

<ENTITIES LIST>
{ENTITIES}
</ENTITIES LIST>

<ADDITIONAL CONTEXT>
{CONTEXT}
</ADDITIONAL CONTEXT>
"""


def get_relations(llm: BaseChatModel,
                  input_data: str,
                  entities: list[str],
                  is_conversation: bool = False,
                  context: str = "") -> List[Tuple[str, str, str]]:
    """
    Extracts subject-predicate-object relations from the input using structured prompting.

    :param llm: The language model used for structured output parsing.
    :param input_data: Input text or conversation.
    :param entities: List of known entity strings used for constraint typing.
    :param is_conversation: Whether the input is a conversation.
    :param context: Optional context to guide the model.
    :return: List of (subject, predicate, object) relation triples.
    """
    # Base prompt
    base_prompt = BASE_RELATION_CONVERSATION_PROMPT if is_conversation else BASE_RELATION_PROMPT

    prompt_template = PromptTemplate(input_variables=["SOURCE_TEXT", "CONTEXT", "ENTITIES"],
                                     template=base_prompt)
    prompt = prompt_template.format(SOURCE_TEXT=input_data,
                                    CONTEXT=context,
                                    ENTITIES=", ".join(entities))

    try:

        class Relation(BaseModel):
            """Strict relation with entity-typed subject/object"""
            subject: Literal[tuple(entities)]
            predicate: str
            object: Literal[tuple(entities)]

        class Output(BaseModel):
            """Output"""
            relations: List[Relation] = Field(
                ..., description="List of subject-predicate-object tuples. Be thorough.")

        model = llm.with_structured_output(Output)
        result = model.invoke(prompt)
        return [(r.subject, r.predicate, r.object) for r in result.relations]

    except Exception as _:
        # Relaxed fallback
        class Relation(BaseModel):
            """Relation"""
            subject: str
            predicate: str
            object: str

        class Output(BaseModel):
            """Output"""
            relations: List[Relation] = Field(
                ..., description="List of subject-predicate-object tuples. Be thorough.")

        model = llm.with_structured_output(Output)
        result = model.invoke(prompt)

        # Fix relations prompt
        fix_prompt = """Fix the relations so that every subject and object of the relations are
        exact matches to an entity. Keep the predicate the same. The meaning of every relation 
        should stay faithful to  the reference text. If you cannot maintain the meaning of the 
        original relation relative to the source text, then do not return it.

<SOURCE TEXT>
{SOURCE_TEXT}
</SOURCE TEXT>

<ENTITIES>
{ENTITIES}
</ENTITIES>

<RELATIONS>
{RELATIONS}
</RELATIONS>
"""

        fix_template = PromptTemplate(input_variables=["SOURCE_TEXT", "ENTITIES", "RELATIONS"],
                                      template=fix_prompt)
        fix_prompt_final = fix_template.format(SOURCE_TEXT=input_data,
                                               ENTITIES=", ".join(entities),
                                               RELATIONS=result.relations)

        class FixOutput(BaseModel):
            """Output Fixed Relations"""
            fixed_relations: List[Relation] = Field(
                ..., description="Filtered and corrected relations")

        model = llm.with_structured_output(FixOutput)
        fix_res = model.invoke(fix_prompt_final)

        # Post-validation
        good_relations = [
            r for r in fix_res.fixed_relations
            if r.subject in entities and r.object in entities
        ]
        return [(r.subject, r.predicate, r.object) for r in good_relations]
