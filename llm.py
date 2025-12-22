
from langchain_core.runnables import RunnableLambda
from langsmith import Client

import chain
from history import get_history
from question import get_popular_questions, get_questions_by_interests
from user import get_user_interests


def create_langsmith_client(api_key):
    return Client(api_key=api_key)


def add_user_message(message, session_id="default"):
    history = get_history(session_id)
    history.add_user_message(message)

    return message


def add_ai_message(message, session_id="default"):
    history = get_history(session_id)
    history.add_ai_message(message)

    return message


def recommand_initial_questions():
    interests = get_user_interests(1)

    if len(interests) != 0:
        return get_questions_by_interests(interests)

    return get_popular_questions()


def recommend_contextual_questions():
    return


def get_ai_response(user_input, session_id="default"):
    dictionary_chain = chain.get_dictionary_chain()
    rag_chain = chain.get_rag_chain(session_id)
    tax_chain = dictionary_chain | RunnableLambda(lambda x: add_user_message(x, session_id)) | rag_chain

    return tax_chain.stream(user_input, config={"configurable": {"session_id": session_id}})
