"""Prompt templates designed to keep answers grounded in retrieved context."""
from langchain_core.prompts import ChatPromptTemplate

NO_ANSWER = "I don't know based on the provided documents."

SYSTEM = f"""You are a careful assistant that answers questions strictly from the \
context provided below.

Rules:
1. Use ONLY the information in <context>. Do not use outside knowledge.
2. If the context does not contain the answer, reply exactly: "{NO_ANSWER}"
3. Keep answers concise. Quote or paraphrase the context faithfully.
4. After the answer, cite the sources you used as [source: <file>, page <n>].

<context>
{{context}}
</context>"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", "{question}")]
)
