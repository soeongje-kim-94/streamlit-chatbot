from langchain_core.chat_history import InMemoryChatMessageHistory

store = {}


def format_history(history: InMemoryChatMessageHistory):
    return "\n".join(f"{m.type.upper()}: {m.content}" for m in history.messages)


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


def is_first_chat(session_id: str):
    return session_id not in store
