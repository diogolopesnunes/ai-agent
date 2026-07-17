from workflow.graph import support_graph


def test_empty_question_is_rejected():
    result = support_graph.invoke({"question": "   "})
    assert "vazia" in result["answer"]["answer"].lower()


def test_unknown_question_abstains():
    result = support_graph.invoke({
        "question": "Qual é o resultado da loteria de outro planeta?"
    })

    assert result["answer"]["needs_human_review"] is True
    assert result["answer"]["sources"] == []


def test_write_action_is_only_proposed():
    result = support_graph.invoke({
        "question": "O sistema caiu e ninguém consegue trabalhar."
    })

    if result.get("proposed_action"):
        assert result["requires_approval"] is True