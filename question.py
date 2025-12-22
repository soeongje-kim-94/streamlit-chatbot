import chain

def get_popular_questions(k=3) -> list:
    return [
        "소득은 어떻게 구분되나요?",
        "원천징수 영수증은 언제 발급받을 수 있나요?",
        "종합소득세 계산을 어떻게 하나요?",
    ]


def get_questions_by_interests(interests: list, k=3) -> list:
    initial_question_chain = chain.get_initial_question_chain()
    
    return initial_question_chain.invoke({"interests": interests, "k": k})
