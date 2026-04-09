from dataclasses import dataclass
from pydantic import BaseModel


# ── Dataclasses (used by parser-based output) ────────────────────────────────

@dataclass
class QAPair:
    question: str
    answer: str

    def __repr__(self):
        return f"QAPair(question={self.question!r}, answer={self.answer!r})"


@dataclass
class Triplet:
    subject: str
    relation: str
    object: str

    def __repr__(self):
        return f"({self.subject}, {self.relation}, {self.object})"


@dataclass
class RetrievalResult:
    index: int | None   # 0-based index into passages list; None if no passage answers
    reasoning: str
    raw: str            # full model output

    def __repr__(self):
        return f"RetrievalResult(index={self.index}, reasoning={self.reasoning!r})"


# ── Pydantic schemas (used by outlines JSON mode) ────────────────────────────

class BulletsOutput(BaseModel):
    bullets: list[str]

class QAPairSchema(BaseModel):
    question: str
    answer: str

class QAPairsOutput(BaseModel):
    pairs: list[QAPairSchema]

class QuestionOutput(BaseModel):
    question: str

class QuestionsListOutput(BaseModel):
    questions: list[str]

class FactOutput(BaseModel):
    fact: str

class AnswerOutput(BaseModel):
    answer: str

class RephraseOutput(BaseModel):
    text: str

class ContinuationOutput(BaseModel):
    text: str

class TripletSchema(BaseModel):
    subject: str
    relation: str
    object: str

class TripletsOutput(BaseModel):
    triplets: list[TripletSchema]

class ComparisonOutput(BaseModel):
    comparison: str

class RetrievalOutput(BaseModel):
    passage_index: int | None
    reasoning: str
