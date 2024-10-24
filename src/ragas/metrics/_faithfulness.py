from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field, RootModel

from ragas.dataset_schema import SingleTurnSample
from ragas.llms.output_parser import RagasOutputParserOld, get_json_format_instructions
from ragas.metrics.base import (
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
    get_segmenter,
)
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


class HasSegmentMethod(t.Protocol):
    def segment(self, text) -> t.Any: ...


logger = logging.getLogger(__name__)


# TODO: Remove this!!!
class Statements(BaseModel):
    sentence_index: int = Field(
        ..., description="Index of the sentence from the statement list"
    )
    simpler_statements: t.List[str] = Field(..., description="the simpler statements")


class StatementsAnswers(RootModel):
    root: t.List[Statements]


_statements_output_instructions = get_json_format_instructions(StatementsAnswers)
_statements_output_parser = RagasOutputParserOld(pydantic_object=StatementsAnswers)

########################################################


class FaithfulnessStatements(BaseModel):
    question: str = Field(description="The question to answer")
    answer: str = Field(description="The answer to the question")
    sentences: t.Dict[int, str] = Field(
        description="A mapping of sentence index to the sentence"
    )


class SentenceComponents(BaseModel):
    sentence_index: int = Field(description="The index of the sentence")
    simpler_statements: t.List[str] = Field(
        description="A list of simpler statements that can be directly inferred from the context"
    )


class SentencesSimplified(BaseModel):
    sentences: t.List[SentenceComponents] = Field(
        description="A list of sentences and their simpler versions"
    )


# examples
example_input_1 = FaithfulnessStatements(
    question="アルベルト・アインシュタインとはどんな人物で、何が最もよく知られているか？",
    answer="彼はドイツ生まれの理論物理学者であり、史上最も偉大で影響力のある物理学者の一人として広く認められている。相対性理論を発展させたことで知られるが、量子力学理論の発展にも重要な貢献をした。",
    sentences={
        0: "彼はドイツ生まれの理論物理学者であり、史上最も偉大で影響力のある物理学者の一人として広く認められている。",
        1: "相対性理論を発展させたことで知られるが、量子力学理論の発展にも重要な貢献をした。",
    },
)

example_output_1 = SentencesSimplified(
    sentences=[
        SentenceComponents(
            sentence_index=0,
            simpler_statements=[
                "アルベルト・アインシュタインはドイツ生まれの理論物理学者である。",
                "アルベルト・アインシュタインは、史上最も偉大で最も影響力のある物理学者の一人として認められている。",
            ],
        ),
        SentenceComponents(
            sentence_index=1,
            simpler_statements=[
                "アルベルト・アインシュタインは相対性理論の開発で最もよく知られている。",
                "アルベルト・アインシュタインもまた、量子力学の理論の発展に重要な貢献をした。",
            ],
        ),
    ]
)


class LongFormAnswerPrompt(PydanticPrompt[FaithfulnessStatements, SentencesSimplified]):
    instruction = "\nGiven a question, an answer, and sentences from the answer analyze the complexity of each sentence given under 'sentences' and break down each sentence into one or more fully understandable statements while also ensuring no pronouns are used in each statement. Format the outputs in JSON."
    input_model = FaithfulnessStatements
    output_model = SentencesSimplified
    examples = [(example_input_1, example_output_1)]


class StatementFaithfulnessAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")


class NLIStatementOutput(BaseModel):
    statements: t.List[StatementFaithfulnessAnswer]


class NLIStatementInput(BaseModel):
    context: str = Field(..., description="The context of the question")
    statements: t.List[str] = Field(..., description="The statements to judge")


class NLIStatementPrompt(PydanticPrompt[NLIStatementInput, NLIStatementOutput]):
    instruction = "Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context."
    input_model = NLIStatementInput
    output_model = NLIStatementOutput
    examples = [
        (
            NLIStatementInput(
                context="""ジョンはXYZ大学の学生である。コンピュータ・サイエンスの学位を取得しようとしている。今学期は、データ構造、アルゴリズム、データベース管理など、いくつかのコースを履修している。ジョンは勤勉な学生で、かなりの時間を勉強に費やし、課題をこなしている。図書館に遅くまで残ってプロジェクトに取り組むことも多い。""",
                statements=[
                    "ジョンは生物学を専攻している。",
                    "ジョンは人工知能のコースを取っています。",
                    "ジョンは熱心な学生です。",
                    "ジョンはアルバイトをしている。",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="ジョンは生物学を専攻している。",
                        reason="ジョンの専攻はコンピューターサイエンスと明記されている。生物学を専攻していることを示唆する情報はない。",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="ジョンは人工知能のコースを取っています。",
                        reason="contextにはジョンが現在受講しているコースが記されているが、人工知能については触れられていない。したがって、ジョンがAIに関するコースを受講しているとは推論できない。",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="ジョンは熱心な学生です。",
                        reason="contextによれば、彼はかなりの時間を勉強と課題の完成に費やしている。さらに、彼はしばしば遅くまで図書館に残ってプロジェクトに取り組んでいることが書かれており、これは献身を意味する。",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="ジョンはアルバイトをしている。",
                        reason="ジョンがアルバイトをしているという情報は、contextにはない。",
                        verdict=0,
                    ),
                ]
            ),
        ),
        (
            NLIStatementInput(
                context="光合成は、植物、藻類、ある種のバクテリアが光エネルギーを化学エネルギーに変換するために用いるプロセスである。",
                statements=[
                    "アルベルト・アインシュタインは天才だった。",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="アルベルト・アインシュタインは天才だった。",
                        reason="contextとstatementは無関係である",
                        verdict=0,
                    )
                ]
            ),
        ),
    ]


@dataclass
class Faithfulness(MetricWithLLM, SingleTurnMetric):
    name: str = "faithfulness"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
                "retrieved_contexts",
            }
        }
    )
    nli_statements_message: PydanticPrompt = field(default_factory=NLIStatementPrompt)
    statement_prompt: PydanticPrompt = field(default_factory=LongFormAnswerPrompt)
    sentence_segmenter: t.Optional[HasSegmentMethod] = None
    max_retries: int = 1
    _reproducibility: int = 1

    @property
    def reproducibility(self):
        return self._reproducibility

    @reproducibility.setter
    def reproducibility(self, value):
        if value < 1:
            logger.warning("reproducibility cannot be less than 1, setting to 1")
            value = 1
        elif value % 2 == 0:
            logger.warning(
                "reproducibility level cannot be set to even number, setting to odd"
            )
            value += 1
        self._reproducibility = value

    def __post_init__(self):
        if self.sentence_segmenter is None:
            language = self.nli_statements_message.language
            self.sentence_segmenter = get_segmenter(language=language, clean=False)

    async def _create_verdicts(
        self, row: t.Dict, statements: t.List[str], callbacks: Callbacks
    ) -> NLIStatementOutput:
        assert self.llm is not None, "llm must be set to compute score"

        contexts_str: str = "\n".join(row["retrieved_contexts"])
        verdicts = await self.nli_statements_message.generate(
            data=NLIStatementInput(context=contexts_str, statements=statements),
            llm=self.llm,
            callbacks=callbacks,
        )

        return verdicts

    async def _create_statements(
        self, row: t.Dict, callbacks: Callbacks
    ) -> SentencesSimplified:
        assert self.llm is not None, "llm is not set"
        assert self.sentence_segmenter is not None, "sentence_segmenter is not set"

        text, question = row["response"], row["user_input"]
        sentences = self.sentence_segmenter.segment(text)
        sentences_with_index = {
            i: sentence
            for i, sentence in enumerate(sentences)
            if sentence.strip().endswith(".")
        }

        statements_simplified = await self.statement_prompt.generate(
            llm=self.llm,
            data=FaithfulnessStatements(
                question=question, answer=text, sentences=sentences_with_index
            ),
            callbacks=callbacks,
        )
        return statements_simplified

    def _compute_score(self, answers: NLIStatementOutput):
        # check the verdicts and compute the score
        faithful_statements = sum(
            1 if answer.verdict else 0 for answer in answers.statements
        )
        num_statements = len(answers.statements)
        if num_statements:
            score = faithful_statements / num_statements
        else:
            logger.warning("No statements were generated from the answer.")
            score = np.nan

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self: t.Self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        statements_simplified = await self._create_statements(row, callbacks)
        if statements_simplified is None:
            return np.nan

        # unwrap the statements
        statements = []
        for component in statements_simplified.sentences:
            statements.extend(component.simpler_statements)

        verdicts = await self._create_verdicts(row, statements, callbacks)
        return self._compute_score(verdicts)


@dataclass
class FaithfulnesswithHHEM(Faithfulness):
    name: str = "faithfulness_with_hhem"  # type: ignore
    device: str = "cpu"
    batch_size: int = 10

    def __post_init__(self):
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(
                "Huggingface transformers must be installed to use this feature, try `pip install transformers`"
            )
        self.nli_classifier = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", trust_remote_code=True
        )
        self.nli_classifier.to(self.device)
        super().__post_init__()

    def _create_pairs(
        self, row: t.Dict, statements: t.List[str]
    ) -> t.List[t.Tuple[str, str]]:
        """
        create pairs of (question, answer) from the row
        """
        premise = "\n".join(row["retrieved_contexts"])
        pairs = [(premise, statement) for statement in statements]
        return pairs

    def _create_batch(
        self, pairs: t.List[t.Tuple[str, str]]
    ) -> t.Generator[t.List[t.Tuple[str, str]], None, None]:
        length_of_pairs = len(pairs)
        for ndx in range(0, length_of_pairs, self.batch_size):
            yield pairs[ndx : min(ndx + self.batch_size, length_of_pairs)]

    async def _ascore(self: t.Self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        statements_simplified = await self._create_statements(row, callbacks)
        if statements_simplified is None:
            return np.nan

        statements = []
        for components in statements_simplified.sentences:
            statements.extend(components.simpler_statements)
        assert isinstance(statements, t.List), "statements must be a list"

        scores = []
        pairs = self._create_pairs(row, statements)
        for input_pairs in self._create_batch(pairs):  # to avoid OOM
            batch_scores = (
                self.nli_classifier.predict(input_pairs).cpu().detach().round()
            )
            scores += batch_scores
        return sum(scores) / len(scores)


faithfulness = Faithfulness()
