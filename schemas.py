from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    source: str
    chunk_id: str | None = None


class AssistantAnswer(BaseModel):
    answer: str = Field(description="Resposta em português brasileiro")
    sources: list[SourceReference]
    urgency: str
    classifier_confidence: float
    needs_human_review: bool
    proposed_action: str | None = None