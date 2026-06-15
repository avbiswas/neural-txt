from dataclasses import dataclass
from typing import Annotated
from pydantic import BaseModel, Field


ShortText = Annotated[str, Field(max_length=512)]
LongText = Annotated[str, Field(max_length=2048)]


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


@dataclass
class RankedResponse:
    index: int           # 0-based index into the input responses
    response: str
    score: float

    def __repr__(self):
        return f"RankedResponse(index={self.index}, score={self.score:.4f}, response={self.response!r})"


@dataclass
class ReasonedOutput:
    output: object
    reasoning: str
    raw: str

    def __repr__(self):
        return f"ReasonedOutput(output={self.output!r}, reasoning={self.reasoning!r})"


# ── Pydantic schemas (used by outlines JSON mode) ────────────────────────────

class BulletsOutput(BaseModel):
    bullets: list[ShortText]

class QAPairSchema(BaseModel):
    question: ShortText
    answer: LongText

class QAPairsOutput(BaseModel):
    pairs: list[QAPairSchema]

class QuestionOutput(BaseModel):
    question: ShortText

class QuestionsListOutput(BaseModel):
    questions: list[ShortText]

class FactOutput(BaseModel):
    fact: ShortText

class AnswerOutput(BaseModel):
    answer: LongText

class RephraseOutput(BaseModel):
    text: LongText

class ContinuationOutput(BaseModel):
    text: LongText

class TripletSchema(BaseModel):
    subject: ShortText
    relation: ShortText
    object: ShortText

class TripletsOutput(BaseModel):
    triplets: list[TripletSchema]

class ComparisonOutput(BaseModel):
    comparison: LongText

class RetrievalOutput(BaseModel):
    passage_index: int | None
    reasoning: ShortText
