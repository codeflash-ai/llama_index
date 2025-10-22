"""Dataset generation from documents."""
from __future__ import annotations

import asyncio
import re
from typing import List, Optional

from llama_index.core import Document, ServiceContext, SummaryIndex
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.ingestion import run_transformations
from llama_index.core.llama_dataset import (
    CreatedBy,
    CreatedByType,
    LabelledRagDataExample,
    LabelledRagDataset,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.node import KeywordNodePostprocessor
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.core.prompts.mixin import (
    PromptDictType,
    PromptMixin,
    PromptMixinType,
)
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeWithScore,
    TransformComponent,
)
from llama_index.core.settings import (
    Settings,
    llm_from_settings_or_context,
    transformations_from_settings_or_context,
)

DEFAULT_QUESTION_GENERATION_PROMPT = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge.
generate only questions based on the below query.
{query_str}
"""


class RagDatasetGenerator(PromptMixin):
    """Generate dataset (question/ question-answer pairs) \
    based on the given documents.

    NOTE: this is a beta feature, subject to change!

    Args:
        nodes (List[Node]): List of nodes. (Optional)
        service_context (ServiceContext): Service Context.
        num_questions_per_chunk: number of question to be \
        generated per chunk. Each document is chunked of size 512 words.
        text_question_template: Question generation template.
        question_gen_query: Question generation query.

    """

    def __init__(
        self,
        nodes: List[BaseNode],
        llm: Optional[LLM] = None,
        num_questions_per_chunk: int = 3,
        text_question_template: Optional[BasePromptTemplate] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        question_gen_query: Optional[str] = None,
        metadata_mode: MetadataMode = MetadataMode.NONE,
        show_progress: bool = False,
        workers: int = DEFAULT_NUM_WORKERS,
        # deprecated
        service_context: Optional[ServiceContext] = None,
    ) -> None:
        """Init params."""
        self._llm = llm or llm_from_settings_or_context(Settings, service_context)
        self.text_question_template = text_question_template or PromptTemplate(
            DEFAULT_QUESTION_GENERATION_PROMPT
        )
        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self.question_gen_query = (
            question_gen_query
            or f"You are a Teacher/Professor. Your task is to setup {num_questions_per_chunk} questions for an upcoming quiz/examination. The questions should be diverse in nature across the document. Restrict the questions to the context information provided."
        )
        self.nodes = nodes
        self._metadata_mode = metadata_mode
        self._show_progress = show_progress
        self._workers = workers

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        llm: Optional[LLM] = None,
        transformations: Optional[List[TransformComponent]] = None,
        num_questions_per_chunk: int = 3,
        text_question_template: Optional[BasePromptTemplate] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        question_gen_query: Optional[str] = None,
        required_keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
        show_progress: bool = False,
        workers: int = DEFAULT_NUM_WORKERS,
        # deprecated
        service_context: Optional[ServiceContext] = None,
    ) -> RagDatasetGenerator:
        """Generate dataset from documents."""
        llm = llm or llm_from_settings_or_context(Settings, service_context)
        transformations = transformations or transformations_from_settings_or_context(
            Settings, service_context
        )

        nodes = run_transformations(
            documents, transformations, show_progress=show_progress
        )

        # use node postprocessor to filter nodes
        required_keywords = required_keywords or []
        exclude_keywords = exclude_keywords or []
        node_postprocessor = KeywordNodePostprocessor(
            llm=llm,
            service_context=service_context,
            required_keywords=required_keywords,
            exclude_keywords=exclude_keywords,
        )
        node_with_scores = [NodeWithScore(node=node) for node in nodes]
        node_with_scores = node_postprocessor.postprocess_nodes(node_with_scores)
        nodes = [node_with_score.node for node_with_score in node_with_scores]

        return cls(
            nodes=nodes,
            llm=llm,
            service_context=service_context,
            num_questions_per_chunk=num_questions_per_chunk,
            text_question_template=text_question_template,
            text_qa_template=text_qa_template,
            question_gen_query=question_gen_query,
            show_progress=show_progress,
            workers=workers,
        )

    async def _agenerate_dataset(
        self,
        nodes: List[BaseNode],
        labelled: bool = False,
    ) -> LabelledRagDataset:
        """Node question generator."""

        # Materialize Document objects first in a single step to prevent repeated computation and object initialization costs deep in loops.
        documents = [
            Document(
                text=node.get_content(metadata_mode=self._metadata_mode),
                metadata=node.metadata,
                excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                relationships=node.relationships,
            )
            for node in nodes
        ]

        # Build SummaryIndex objects via list comprehension for maximal speed and locality.
        # This avoids repeated attribute access and enables batch construction.
        summary_indices: List[SummaryIndex] = [
            SummaryIndex.from_documents([doc]) for doc in documents
        ]

        # Precompute query_engines for question generation
        query_engines = [
            index.as_query_engine(
                llm=self._llm,
                text_qa_template=self.text_question_template,
                use_async=True,
            )
            for index in summary_indices
        ]

        # Submit all question-generation queries at once for maximal throughput
        query_tasks = [
            engine.aquery(self.question_gen_query) for engine in query_engines
        ]
        responses = await run_jobs(query_tasks, self._show_progress, self._workers)

        # Process question strings more efficiently
        cleaned_questions_list = []
        for response in responses:
            result = str(response).strip().split("\n")
            cleaned_questions_list.append(_clean_questions(result))

        # Precompute values reused in all examples to avoid per-iteration overhead
        model_name = self._llm.metadata.model_name
        created_by = CreatedBy(type=CreatedByType.AI, model_name=model_name)
        reference_contexts = [node.text for node in nodes]  # List of texts, stays in node order

        examples: List[LabelledRagDataExample] = []

        if labelled:
            # For each node, generate QA results for each question
            # Generate all QA engines and qr_tasks up front in a flat structure for better batching
            # This reduces costly repeated object construction.
            qa_query_engines = [
                index.as_query_engine(
                    llm=self._llm,
                    text_qa_template=self.text_qa_template,
                )
                for index in summary_indices
            ]
            # Build all <task,node_idx,question> tuples for batch answer generation
            qr_task_tuples = [
                (engine.aquery(question), node_idx, question)
                for node_idx, (engine, questions) in enumerate(zip(qa_query_engines, cleaned_questions_list))
                for question in questions
            ]
            qr_tasks = [tpl[0] for tpl in qr_task_tuples]
            # Batch run answer generation for ALL questions, for maximal parallelism
            answer_responses_flat: List[RESPONSE_TYPE] = await run_jobs(
                qr_tasks, self._show_progress, self._workers
            )

            # To maintain the node_idx/question mapping, pair answer_responses back to question and context
            idx = 0
            for (task, node_idx, question) in qr_task_tuples:
                answer_response = answer_responses_flat[idx]
                example = LabelledRagDataExample(
                    query=question,
                    reference_answer=str(answer_response),
                    reference_contexts=[reference_contexts[node_idx]],
                    reference_answer_by=created_by,
                    query_by=created_by,
                )
                examples.append(example)
                idx += 1
        else:
            # Store empty answers for all generated questions
            # Directly zip node indices and questions for optimal appending
            for node_idx, questions in enumerate(cleaned_questions_list):
                for question in questions:
                    examples.append(
                        LabelledRagDataExample(
                            query=question,
                            reference_answer="",
                            reference_contexts=[reference_contexts[node_idx]],
                            reference_answer_by=None,
                            query_by=created_by,
                        )
                    )

        return LabelledRagDataset(examples=examples)

    async def agenerate_questions_from_nodes(self) -> LabelledRagDataset:
        """Generates questions but not the reference answers."""
        return await self._agenerate_dataset(self.nodes, labelled=False)

    async def agenerate_dataset_from_nodes(self) -> LabelledRagDataset:
        """Generates questions for each document."""
        return await self._agenerate_dataset(self.nodes, labelled=True)

    def generate_questions_from_nodes(self) -> LabelledRagDataset:
        """Generates questions but not the reference answers."""
        return asyncio.run(self.agenerate_questions_from_nodes())

    def generate_dataset_from_nodes(self) -> LabelledRagDataset:
        """Generates questions for each document."""
        return asyncio.run(self.agenerate_dataset_from_nodes())

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "text_question_template": self.text_question_template,
            "text_qa_template": self.text_qa_template,
        }

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "text_question_template" in prompts:
            self.text_question_template = prompts["text_question_template"]
        if "text_qa_template" in prompts:
            self.text_qa_template = prompts["text_qa_template"]


def _clean_questions(result: List[str]) -> List[str]:
    # Remove leading numbers/dots/etc from questions and filter empty
    return [
        question
        for question in (
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ) if question
    ]
