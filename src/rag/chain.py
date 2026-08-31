"""Conversational RAG chain: history-aware retrieval + grounded generation
with inline source citations."""
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.retrievers import BaseRetriever

from src import config

# create_stuff_documents_chain's default document formatting shows the LLM
# ONLY page_content, never metadata — so without this, an instruction to
# "cite [source: <name>]" has no real source name to draw on, and the model
# fabricates a plausible-looking one from whatever text happens to be in the
# chunk (a real, reproduced bug — see README's Corrective RAG section). This
# tags every passage with its actual source before the model ever sees it.
DOCUMENT_PROMPT = PromptTemplate.from_template("[source: {source}]\n{page_content}")

CONTEXTUALIZE_PROMPT = (
    "Given a chat history and the latest user question, which might reference "
    "context in the chat history, rewrite it as a standalone question that can "
    "be understood without the chat history. Do NOT answer the question, just "
    "reformulate it if needed, otherwise return it as-is."
)

ANSWER_PROMPT = (
    "You are a knowledgeable assistant helping someone understand what's in "
    "the retrieved context below. Answer using ONLY that context — every "
    "factual claim must still be traceable to it, and each passage is "
    "prefixed with its real [source: ...] tag; cite that EXACT tag right "
    "after the claim it supports. Never invent, paraphrase, or guess a "
    "source name; only copy a tag that's actually shown.\n\n"
    "Within that constraint, write like you're actually explaining it to "
    "someone, not filling out a form: full sentences, a natural tone, and "
    "enough detail to be genuinely useful. If the context has more relevant "
    "detail than a one-line answer would use — conditions, exceptions, "
    "related specifics — include it rather than trimming to the bare "
    "minimum. Don't pad with restatement or filler, and don't add anything "
    "the context doesn't support; the goal is a fuller, more complete answer "
    "grounded in what's actually there, not a longer one for its own sake.\n\n"
    "Context may include both the user's own documents and live web search "
    "results — treat both as valid sources; web tags are URLs, so the user can "
    "tell a claim came from the internet rather than their own material.\n\n"
    "If the context does not contain the answer, say so plainly and "
    "naturally instead of guessing.\n\n"
    "Context:\n{context}"
)


def get_llm(temperature: float = 0.1) -> BaseChatModel:
    config.require_credentials()

    if config.LLM_PROVIDER == "vertexai":
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(
            model=config.GENERATION_MODEL,
            project=config.VERTEX_PROJECT_ID,
            location=config.VERTEX_LOCATION,
            temperature=temperature,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.GENERATION_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=temperature,
    )


def build_conversational_rag_chain(retriever: BaseRetriever):
    """Returns (retrieval_chain, document_chain). retrieval_chain does the
    full history-aware retrieve-then-generate pass; document_chain is also
    exposed on its own so a self-correction loop (see self_correction.py)
    can regenerate an answer from the SAME already-retrieved context —
    e.g. after a hallucination grade fails — without paying for (and
    getting the exact same result from) a fresh, deterministic retrieval."""
    llm = get_llm()

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(
        llm, answer_prompt, document_prompt=DOCUMENT_PROMPT
    )

    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    return retrieval_chain, document_chain
