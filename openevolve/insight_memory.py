"""
Three-stage insight memory for cross-problem retrieval and guidance.

Stage 1: Raw evolution memory (factual events, reports, traces, errors).
Stage 2: Local insight memory (LLM-derived insight tied to Stage 1 evidence).
Stage 3: Clustered insight memory (incremental clustering over Stage 2 embeddings).
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb

from openevolve.config import InsightMemoryConfig
from openevolve.embedding import EmbeddingClient
from openevolve.utils.metrics_utils import get_fitness_score

logger = logging.getLogger(__name__)

_MEMORY_VERSION = 2
_META_STATE_ID = "meta_state"


@dataclass
class RawEvolutionRecord:
    record_id: str
    timestamp: float
    step_index: int
    problem_id: str
    task_id: str
    source_component: str
    source_file: Optional[str]
    event_type: str
    raw_report_text: str
    raw_payload: Dict[str, Any]
    execution_context: Dict[str, Any]
    stage2_links: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawEvolutionRecord":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        filtered.setdefault("stage2_links", [])
        filtered.setdefault("source_file", None)
        return cls(**filtered)


@dataclass
class LocalInsightRecord:
    insight_id: str
    timestamp: float
    step_index: int
    problem_id: str
    task_id: str
    stage1_evidence_ids: List[str]
    insight_text: str
    code_locations: List[str]
    embedding: List[float]
    embedding_model: str
    stage3_cluster_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalInsightRecord":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        filtered.setdefault("stage3_cluster_ids", [])
        filtered.setdefault("metadata", {})
        return cls(**filtered)


@dataclass
class InsightCluster:
    cluster_id: str
    member_insight_ids: List[str]
    representative_embedding: List[float]
    summary: Optional[str]
    created_timestamp: float
    created_step_index: int
    updated_timestamp: float
    updated_step_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.member_insight_ids)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["cluster_size"] = self.size
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InsightCluster":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        filtered.setdefault("metadata", {})
        return cls(**filtered)


class ThreeStageMemoryStorage:
    """Persistent storage and index maintenance for all three memory stages via ChromaDB."""

    def __init__(self, config: InsightMemoryConfig):
        self.config = config
        self.memory_name = _sanitize_memory_name(config.memory_name)
        self.storage_path = self._resolve_storage_path(config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.storage_path))
        prefix = self.memory_name
        self.stage1_collection = self.client.get_or_create_collection(name=f"{prefix}_stage1")
        self.stage2_collection = self.client.get_or_create_collection(name=f"{prefix}_stage2")
        self.stage3_collection = self.client.get_or_create_collection(name=f"{prefix}_stage3")
        self.meta_collection = self.client.get_or_create_collection(name=f"{prefix}_meta")

        # Initialize meta state once.
        if self.meta_collection.count() == 0:
            self._set_meta(
                {
                    "version": _MEMORY_VERSION,
                    "memory_name": self.memory_name,
                    "step_counter": 0,
                    "legacy_migrated": False,
                }
            )

    def _resolve_storage_path(self, storage_path: Optional[str]) -> Path:
        if storage_path:
            return Path(storage_path).expanduser().resolve()
        return Path.home() / ".openevolve" / "chroma" / self.memory_name

    def _set_meta(self, payload: Dict[str, Any]) -> None:
        self.meta_collection.upsert(
            ids=[_META_STATE_ID],
            documents=[json.dumps(payload, ensure_ascii=True)],
            metadatas=[{"version": int(payload.get("version", _MEMORY_VERSION))}],
        )

    def _get_meta(self) -> Dict[str, Any]:
        result = self.meta_collection.get(ids=[_META_STATE_ID], include=["documents"])
        docs = result.get("documents", [])
        if not docs:
            default = {
                "version": _MEMORY_VERSION,
                "memory_name": self.memory_name,
                "step_counter": 0,
                "legacy_migrated": False,
            }
            self._set_meta(default)
            return default

        doc = docs[0]
        try:
            parsed = json.loads(doc) if isinstance(doc, str) else {}
        except Exception:
            parsed = {}

        parsed.setdefault("version", _MEMORY_VERSION)
        parsed.setdefault("memory_name", self.memory_name)
        parsed.setdefault("step_counter", 0)
        parsed.setdefault("legacy_migrated", False)
        return parsed

    def _upsert_record(self, collection, record_id: str, payload: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        collection.upsert(
            ids=[record_id],
            documents=[json.dumps(payload, ensure_ascii=True)],
            metadatas=[metadata],
        )

    def _get_record(self, collection, record_id: str) -> Optional[Dict[str, Any]]:
        result = collection.get(ids=[record_id], include=["documents"])
        docs = result.get("documents", [])
        if not docs:
            return None
        doc = docs[0]
        if not isinstance(doc, str):
            return None
        try:
            data = json.loads(doc)
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _list_records(self, collection) -> List[Dict[str, Any]]:
        if collection.count() == 0:
            return []
        result = collection.get(include=["documents"])
        docs = result.get("documents", [])
        records: List[Dict[str, Any]] = []
        for doc in docs:
            if not isinstance(doc, str):
                continue
            try:
                payload = json.loads(doc)
            except Exception:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records

    def next_step(self) -> int:
        meta = self._get_meta()
        meta["step_counter"] = int(meta.get("step_counter", 0)) + 1
        step = int(meta["step_counter"])
        self._set_meta(meta)
        return step

    def set_step_floor(self, step_value: int) -> None:
        meta = self._get_meta()
        current = int(meta.get("step_counter", 0))
        if step_value > current:
            meta["step_counter"] = int(step_value)
            self._set_meta(meta)

    def get_step_counter(self) -> int:
        meta = self._get_meta()
        return int(meta.get("step_counter", 0))

    def add_stage1(self, record: RawEvolutionRecord) -> None:
        payload = record.to_dict()
        metadata = {
            "step_index": int(record.step_index),
            "timestamp": float(record.timestamp),
            "problem_id": str(record.problem_id),
            "task_id": str(record.task_id),
            "event_type": str(record.event_type),
            "source_component": str(record.source_component),
        }
        self._upsert_record(self.stage1_collection, record.record_id, payload, metadata)

    def add_stage2(self, record: LocalInsightRecord) -> None:
        payload = record.to_dict()
        metadata = {
            "step_index": int(record.step_index),
            "timestamp": float(record.timestamp),
            "problem_id": str(record.problem_id),
            "task_id": str(record.task_id),
            "embedding_model": str(record.embedding_model or ""),
        }
        self._upsert_record(self.stage2_collection, record.insight_id, payload, metadata)

    def update_stage2(self, record: LocalInsightRecord) -> None:
        self.add_stage2(record)

    def upsert_stage3(self, cluster: InsightCluster) -> None:
        payload = cluster.to_dict()
        metadata = {
            "created_step_index": int(cluster.created_step_index),
            "updated_step_index": int(cluster.updated_step_index),
            "cluster_size": int(cluster.size),
        }
        self._upsert_record(self.stage3_collection, cluster.cluster_id, payload, metadata)

    def remove_all_stage3(self) -> None:
        ids = self.stage3_collection.get().get("ids", [])
        if ids:
            self.stage3_collection.delete(ids=ids)

    def get_stage1(self, record_id: str) -> Optional[RawEvolutionRecord]:
        data = self._get_record(self.stage1_collection, record_id)
        return RawEvolutionRecord.from_dict(data) if data else None

    def get_stage2(self, insight_id: str) -> Optional[LocalInsightRecord]:
        data = self._get_record(self.stage2_collection, insight_id)
        return LocalInsightRecord.from_dict(data) if data else None

    def get_stage3(self, cluster_id: str) -> Optional[InsightCluster]:
        data = self._get_record(self.stage3_collection, cluster_id)
        return InsightCluster.from_dict(data) if data else None

    def list_stage1(self) -> List[RawEvolutionRecord]:
        return [RawEvolutionRecord.from_dict(x) for x in self._list_records(self.stage1_collection)]

    def list_stage2(self) -> List[LocalInsightRecord]:
        return [LocalInsightRecord.from_dict(x) for x in self._list_records(self.stage2_collection)]

    def list_stage3(self) -> List[InsightCluster]:
        return [InsightCluster.from_dict(x) for x in self._list_records(self.stage3_collection)]

    def set_legacy_migrated(self) -> None:
        meta = self._get_meta()
        meta["legacy_migrated"] = True
        self._set_meta(meta)

    def was_legacy_migrated(self) -> bool:
        meta = self._get_meta()
        return bool(meta.get("legacy_migrated", False))


class EmbeddingProvider:
    """Embedding provider with per-model caching and reuse support."""

    def __init__(self, config: InsightMemoryConfig):
        self.config = config
        self.model_name = config.embedding_model
        self.client = EmbeddingClient(
            model_name=config.embedding_model,
            api_base=config.embedding_api_base,
            api_key=config.embedding_api_key,
        )
        self._cache: Dict[Tuple[str, str], List[float]] = {}

    def embed_text(
        self,
        text: str,
        existing_embedding: Optional[List[float]] = None,
        existing_model_name: Optional[str] = None,
    ) -> List[float]:
        text = (text or "").strip()
        if not text:
            return []

        if existing_embedding and existing_model_name == self.model_name:
            return existing_embedding

        cache_key = (self.model_name, text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self.client.get_embedding(text)
        if isinstance(embedding, tuple):
            embedding = embedding[0]
        embedding = [float(v) for v in (embedding or [])]
        self._cache[cache_key] = embedding
        return embedding


class RawEvolutionMemory:
    """Stage 1 memory: stores faithful raw evolution events and reports."""

    def __init__(self, storage: ThreeStageMemoryStorage):
        self.storage = storage

    def create_record(
        self,
        *,
        step_index: int,
        problem_id: str,
        task_id: str,
        source_component: str,
        source_file: Optional[str],
        event_type: str,
        raw_payload: Dict[str, Any],
        execution_context: Dict[str, Any],
        raw_report_text: Optional[str] = None,
    ) -> RawEvolutionRecord:
        record = RawEvolutionRecord(
            record_id=str(uuid.uuid4()),
            timestamp=time.time(),
            step_index=step_index,
            problem_id=problem_id,
            task_id=task_id,
            source_component=source_component,
            source_file=source_file,
            event_type=event_type,
            raw_report_text=raw_report_text or json.dumps(raw_payload, ensure_ascii=True),
            raw_payload=raw_payload,
            execution_context=execution_context,
        )
        self.storage.add_stage1(record)
        return record

    def link_stage2(self, stage1_id: str, stage2_id: str) -> None:
        record = self.storage.get_stage1(stage1_id)
        if record is None:
            return
        if stage2_id not in record.stage2_links:
            record.stage2_links.append(stage2_id)
            self.storage.add_stage1(record)

    def list_by_problem(self, problem_id: str) -> List[RawEvolutionRecord]:
        return [r for r in self.storage.list_stage1() if r.problem_id == problem_id]


class LocalInsightMemory:
    """Stage 2 memory: creates local insights linked to Stage 1 evidence."""

    def __init__(
        self,
        storage: ThreeStageMemoryStorage,
        embedding_provider: EmbeddingProvider,
        llm_ensemble,
        config: InsightMemoryConfig,
    ):
        self.storage = storage
        self.embedding_provider = embedding_provider
        self.llm_ensemble = llm_ensemble
        self.config = config

    async def generate_problem_insight(self, report: Dict[str, Any], fitness_score: float) -> str:
        system_message = (
            "You summarize performance bottlenecks at a high level. "
            "Do not mention code, file names, or line details. "
            "Focus on algorithmic or architectural issues and the main challenge."
        )
        user_message = (
            "You are generating a localized insight from one raw evolution record. "
            "Write 2-4 sentences that identify the local bottleneck and challenge.\n\n"
            f"Fitness score: {fitness_score:.4f}\n"
            f"Raw report: {json.dumps(report, ensure_ascii=True)}"
        )

        response = await self.llm_ensemble.generate_with_context(
            system_message=system_message,
            messages=[{"role": "user", "content": user_message}],
        )
        response_text = (response or "").strip()
        if response_text:
            return response_text
        return _fallback_problem_insight(report)

    async def generate_change_insight(
        self,
        report: Dict[str, Any],
        parent_fitness: float,
        child_fitness: float,
        fitness_delta: float,
        changes_summary: str,
    ) -> str:
        system_message = (
            "You summarize a localized implementation change at a high level. "
            "Do not mention specific identifiers. "
            "Use terms like caching, pruning, vectorization, data structure changes, or recursion reduction."
        )
        user_message = (
            "You are generating a localized insight from one raw evolution record. "
            "Write 1-2 sentences describing the local change type and impact.\n\n"
            f"Parent fitness: {parent_fitness:.4f}\n"
            f"Child fitness: {child_fitness:.4f}\n"
            f"Fitness delta: {fitness_delta:+.4f}\n"
            f"Changes summary: {changes_summary}\n"
            f"Raw report: {json.dumps(report, ensure_ascii=True)}"
        )

        response = await self.llm_ensemble.generate_with_context(
            system_message=system_message,
            messages=[{"role": "user", "content": user_message}],
        )
        response_text = (response or "").strip()
        if response_text:
            return response_text
        return _fallback_change_insight(changes_summary, fitness_delta)

    def create_insight(
        self,
        *,
        step_index: int,
        problem_id: str,
        task_id: str,
        stage1_evidence_ids: List[str],
        insight_text: str,
        code_locations: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        insight_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        embedding_model: Optional[str] = None,
    ) -> LocalInsightRecord:
        record = LocalInsightRecord(
            insight_id=insight_id or str(uuid.uuid4()),
            timestamp=time.time(),
            step_index=step_index,
            problem_id=problem_id,
            task_id=task_id,
            stage1_evidence_ids=list(stage1_evidence_ids),
            insight_text=(insight_text or "").strip(),
            code_locations=list(code_locations or []),
            embedding=list(embedding or []),
            embedding_model=embedding_model or "",
            metadata=metadata or {},
        )

        record.embedding = self.embedding_provider.embed_text(
            record.insight_text,
            existing_embedding=record.embedding,
            existing_model_name=record.embedding_model,
        )
        record.embedding_model = self.embedding_provider.model_name

        self.storage.add_stage2(record)
        return record


class ClusterManager:
    """Clustering engine over Stage 2 insights using cosine similarity."""

    def __init__(self, config: InsightMemoryConfig):
        self.config = config

    def compute_cluster_representation(
        self, member_embeddings: Sequence[List[float]]
    ) -> List[float]:
        if not member_embeddings:
            return []

        centroid = _vector_average(member_embeddings)
        mode = self.config.cluster_representation_mode

        if mode == "average":
            return centroid

        similarities = [_cosine_similarity(vec, centroid) for vec in member_embeddings]

        if mode == "similarity_weighted":
            weights = [max(0.001, s) for s in similarities]
            return _vector_weighted_average(member_embeddings, weights)

        if mode == "attention_weighted":
            temperature = max(0.01, float(self.config.cluster_attention_temperature))
            scaled = [temperature * s for s in similarities]
            max_scaled = max(scaled) if scaled else 0.0
            exp_values = [math.exp(v - max_scaled) for v in scaled]
            denom = sum(exp_values)
            if denom <= 0.0:
                return centroid
            weights = [v / denom for v in exp_values]
            return _vector_weighted_average(member_embeddings, weights)

        return centroid


class ClusteredInsightMemory:
    """Stage 3 memory: maintains and updates clusters over Stage 2 insights."""

    def __init__(
        self,
        storage: ThreeStageMemoryStorage,
        cluster_manager: ClusterManager,
        config: InsightMemoryConfig,
        llm_ensemble,
    ):
        self.storage = storage
        self.cluster_manager = cluster_manager
        self.config = config
        self.llm_ensemble = llm_ensemble

    async def assign_or_create_cluster(self, stage2_record: LocalInsightRecord) -> InsightCluster:
        existing_clusters = self.storage.list_stage3()
        best_cluster: Optional[InsightCluster] = None
        best_similarity = -1.0

        for cluster in existing_clusters:
            if (
                self.config.max_cluster_members is not None
                and len(cluster.member_insight_ids) >= self.config.max_cluster_members
            ):
                continue

            similarity = _cosine_similarity(
                stage2_record.embedding, cluster.representative_embedding
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        now = time.time()
        threshold = float(self.config.clustering_similarity_threshold)
        if best_cluster and best_similarity >= threshold:
            if stage2_record.insight_id not in best_cluster.member_insight_ids:
                best_cluster.member_insight_ids.append(stage2_record.insight_id)
            best_cluster.updated_timestamp = now
            best_cluster.updated_step_index = stage2_record.step_index
            best_cluster.metadata["last_assignment_similarity"] = best_similarity
            best_cluster.metadata["cluster_size"] = len(best_cluster.member_insight_ids)

            self._refresh_cluster_representation(best_cluster)
            await self._refresh_cluster_summary(best_cluster)
            self.storage.upsert_stage3(best_cluster)
            self._ensure_bidirectional_link(stage2_record, best_cluster.cluster_id)
            return best_cluster

        cluster = InsightCluster(
            cluster_id=str(uuid.uuid4()),
            member_insight_ids=[stage2_record.insight_id],
            representative_embedding=list(stage2_record.embedding),
            summary=None,
            created_timestamp=now,
            created_step_index=stage2_record.step_index,
            updated_timestamp=now,
            updated_step_index=stage2_record.step_index,
            metadata={
                "cluster_size": 1,
                "representation_mode": self.config.cluster_representation_mode,
            },
        )
        await self._refresh_cluster_summary(cluster)
        self.storage.upsert_stage3(cluster)
        self._ensure_bidirectional_link(stage2_record, cluster.cluster_id)
        return cluster

    async def full_recluster(self, step_index: int) -> None:
        insights = sorted(self.storage.list_stage2(), key=lambda x: (x.timestamp, x.insight_id))

        # Reset Stage 2 cluster links.
        for insight in insights:
            insight.stage3_cluster_ids = []
            self.storage.update_stage2(insight)

        self.storage.remove_all_stage3()

        for insight in insights:
            await self.assign_or_create_cluster(insight)

        self.storage.set_step_floor(step_index)

    def _refresh_cluster_representation(self, cluster: InsightCluster) -> None:
        embeddings: List[List[float]] = []
        for insight_id in cluster.member_insight_ids:
            insight = self.storage.get_stage2(insight_id)
            if insight and insight.embedding:
                embeddings.append(insight.embedding)

        cluster.representative_embedding = self.cluster_manager.compute_cluster_representation(
            embeddings
        )
        cluster.metadata["cluster_size"] = len(cluster.member_insight_ids)

    async def _refresh_cluster_summary(self, cluster: InsightCluster) -> None:
        if not self.config.generate_cluster_summary:
            return

        members = [self.storage.get_stage2(m) for m in cluster.member_insight_ids]
        member_texts = [m.insight_text for m in members if m and m.insight_text]
        if not member_texts:
            return

        if self.llm_ensemble is None:
            cluster.summary = " ; ".join(member_texts[:3])
            return

        system_message = (
            "You summarize a cluster of related local optimization insights. "
            "Return a concise practical summary in 2-3 sentences."
        )
        user_message = (
            "Cluster member insights:\n- " + "\n- ".join(member_texts[:10])
        )

        response = await self.llm_ensemble.generate_with_context(
            system_message=system_message,
            messages=[{"role": "user", "content": user_message}],
        )
        summary = (response or "").strip()
        if summary:
            cluster.summary = summary

    def _ensure_bidirectional_link(self, insight: LocalInsightRecord, cluster_id: str) -> None:
        if cluster_id not in insight.stage3_cluster_ids:
            insight.stage3_cluster_ids.append(cluster_id)
            self.storage.update_stage2(insight)


class MemoryRetriever:
    """Hierarchical retrieval: clusters -> members -> LLM synthesis."""

    def __init__(
        self,
        storage: ThreeStageMemoryStorage,
        embedding_provider: EmbeddingProvider,
        llm_ensemble,
        config: InsightMemoryConfig,
    ):
        self.storage = storage
        self.embedding_provider = embedding_provider
        self.llm_ensemble = llm_ensemble
        self.config = config

    def retrieve(
        self,
        query_text: str,
        *,
        top_clusters: Optional[int] = None,
        top_members_per_cluster: Optional[int] = None,
    ) -> Dict[str, Any]:
        query_embedding = self.embedding_provider.embed_text(query_text)
        if not query_embedding:
            return {
                "query_embedding": [],
                "clusters": [],
                "insights": [],
            }

        cluster_limit = max(1, int(top_clusters or self.config.retrieval_top_clusters))
        member_limit = max(
            1,
            int(top_members_per_cluster or self.config.retrieval_top_members_per_cluster),
        )

        ranked_clusters: List[Tuple[float, InsightCluster]] = []
        for cluster in self.storage.list_stage3():
            sim = _cosine_similarity(query_embedding, cluster.representative_embedding)
            ranked_clusters.append((sim, cluster))
        ranked_clusters.sort(key=lambda x: x[0], reverse=True)
        selected_clusters = ranked_clusters[:cluster_limit]

        selected_insights: List[Dict[str, Any]] = []
        for cluster_sim, cluster in selected_clusters:
            member_candidates: List[Tuple[float, LocalInsightRecord]] = []
            for insight_id in cluster.member_insight_ids:
                insight = self.storage.get_stage2(insight_id)
                if not insight or not insight.embedding:
                    continue
                sim = _cosine_similarity(query_embedding, insight.embedding)
                member_candidates.append((sim, insight))
            member_candidates.sort(key=lambda x: x[0], reverse=True)

            for member_sim, insight in member_candidates[:member_limit]:
                evidence_records = [
                    self.storage.get_stage1(stage1_id)
                    for stage1_id in insight.stage1_evidence_ids
                    if self.storage.get_stage1(stage1_id) is not None
                ]
                selected_insights.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "cluster_similarity": cluster_sim,
                        "insight_id": insight.insight_id,
                        "insight_similarity": member_sim,
                        "insight_text": insight.insight_text,
                        "code_locations": insight.code_locations,
                        "stage1_evidence": [r.to_dict() for r in evidence_records],
                    }
                )

        return {
            "query_embedding": query_embedding,
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "similarity": s,
                    "size": len(c.member_insight_ids),
                    "summary": c.summary,
                }
                for s, c in selected_clusters
            ],
            "insights": selected_insights,
        }

    async def synthesize_recommendation(
        self,
        query_text: str,
        retrieval: Dict[str, Any],
    ) -> str:
        insights = retrieval.get("insights", [])
        if not insights:
            return ""

        evidence_lines: List[str] = []
        for item in insights:
            insight_line = (
                f"insight={item['insight_text']} | "
                f"cluster_sim={item['cluster_similarity']:.4f} | "
                f"insight_sim={item['insight_similarity']:.4f}"
            )
            evidence_lines.append(insight_line)

            if self.config.include_stage1_evidence_in_retrieval:
                for evidence in item.get("stage1_evidence", [])[:2]:
                    evidence_lines.append(
                        "stage1="
                        + json.dumps(
                            {
                                "event_type": evidence.get("event_type"),
                                "source_component": evidence.get("source_component"),
                                "step_index": evidence.get("step_index"),
                                "raw_report_text": evidence.get("raw_report_text", "")[:400],
                            },
                            ensure_ascii=True,
                        )
                    )

        if self.llm_ensemble is None:
            return "\n".join(
                [
                    "Use these retrieved local insights as actionable guidance:",
                    *[f"- {line}" for line in evidence_lines[:10]],
                ]
            )

        system_message = (
            "You are synthesizing actionable guidance from retrieved local insights and raw evidence. "
            "Produce concise recommendations for the next code mutation attempt. "
            "Ground recommendations in evidence and avoid unsupported claims."
        )
        user_message = (
            f"Current query/problem:\n{query_text}\n\n"
            "Retrieved evidence:\n"
            + "\n".join(f"- {line}" for line in evidence_lines[:30])
            + "\n\nReturn 3-6 concrete recommendations."
        )

        response = await self.llm_ensemble.generate_with_context(
            system_message=system_message,
            messages=[{"role": "user", "content": user_message}],
        )
        return (response or "").strip()

    async def build_context(
        self,
        query_text: str,
        *,
        top_clusters: Optional[int] = None,
        top_members_per_cluster: Optional[int] = None,
    ) -> str:
        retrieval = self.retrieve(
            query_text,
            top_clusters=top_clusters,
            top_members_per_cluster=top_members_per_cluster,
        )
        insights = retrieval.get("insights", [])
        if not insights:
            return ""

        synthesis = await self.synthesize_recommendation(query_text, retrieval)

        lines = ["## Evolution Memory (Three-Stage)"]
        lines.append("Retrieved local insights from similar clusters:")
        for item in insights:
            lines.append(
                f"- cluster={item['cluster_id']} | "
                f"cluster_sim={item['cluster_similarity']:.4f} | "
                f"insight_sim={item['insight_similarity']:.4f} | "
                f"insight={item['insight_text']}"
            )

        if synthesis:
            lines.append("")
            lines.append("Synthesis for current mutation:")
            lines.append(synthesis)

        return "\n".join(lines).strip() + "\n"


class ThreeStageMemoryManager:
    """Orchestrates Stage 1/2/3 writes, reclustering, and retrieval."""

    def __init__(
        self,
        config: InsightMemoryConfig,
        llm_ensemble,
        feature_dimensions: List[str],
        *,
        problem_id: str,
        task_id: str,
        source_file: Optional[str] = None,
    ):
        self.config = config
        self.llm_ensemble = llm_ensemble
        self.feature_dimensions = feature_dimensions
        self.problem_id = problem_id
        self.task_id = task_id
        self.source_file = source_file

        self.storage = ThreeStageMemoryStorage(config)
        self.embedding_provider = EmbeddingProvider(config)
        self.raw_memory = RawEvolutionMemory(self.storage)
        self.local_memory = LocalInsightMemory(
            self.storage,
            self.embedding_provider,
            llm_ensemble,
            config,
        )
        self.clustered_memory = ClusteredInsightMemory(
            self.storage,
            ClusterManager(config),
            config,
            llm_ensemble,
        )
        self.retriever = MemoryRetriever(
            self.storage,
            self.embedding_provider,
            llm_ensemble,
            config,
        )

        self.problem_insight: Optional[str] = None

        if self.config.migration_from_legacy_enabled:
            self._migrate_legacy_if_needed()

    def _legacy_storage_path(self) -> Path:
        if self.config.storage_path:
            return Path(self.config.storage_path).expanduser().resolve()
        memory_name = _sanitize_memory_name(self.config.memory_name)
        return Path.home() / ".openevolve" / "chroma" / memory_name

    def _migrate_legacy_if_needed(self) -> None:
        if self.storage.was_legacy_migrated() or self.storage.list_stage2():
            return

        legacy_path = self._legacy_storage_path()
        if not legacy_path.exists():
            self.storage.set_legacy_migrated()
            return

        try:
            client = chromadb.PersistentClient(path=str(legacy_path))
            collection_name = f"{_sanitize_memory_name(self.config.memory_name)}_problem"
            collection = client.get_collection(collection_name)
            payload = collection.get(include=["metadatas", "documents", "embeddings"])
        except Exception as exc:
            logger.info("No legacy insight memory detected for migration: %s", exc)
            self.storage.set_legacy_migrated()
            return

        ids = payload.get("ids", []) or []
        documents = payload.get("documents", []) or []
        metadatas = payload.get("metadatas", []) or []
        embeddings = payload.get("embeddings", []) or []

        imported = 0
        max_step = self.storage.get_step_counter()

        for idx, entry_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            problem_text = documents[idx] if idx < len(documents) else ""
            embedding = embeddings[idx] if idx < len(embeddings) else []

            report_raw = metadata.get("report", "{}")
            try:
                report = json.loads(report_raw) if isinstance(report_raw, str) else report_raw
            except Exception:
                report = {"legacy_report": report_raw}

            step_index = int(max_step + 1)
            max_step = step_index

            raw_record = self.raw_memory.create_record(
                step_index=step_index,
                problem_id=self.problem_id,
                task_id=self.task_id,
                source_component="legacy_migration",
                source_file=self.source_file,
                event_type="legacy_import",
                raw_payload={
                    "legacy_entry_id": entry_id,
                    "legacy_report": report,
                    "legacy_metadata": metadata,
                },
                execution_context={"migration": True},
                raw_report_text=json.dumps(report, ensure_ascii=True),
            )

            change_insight = metadata.get("change_insight", "")
            text = f"Problem insight: {problem_text}. Change insight: {change_insight}".strip()
            if not text:
                continue

            local_record = self.local_memory.create_insight(
                step_index=step_index,
                problem_id=self.problem_id,
                task_id=self.task_id,
                stage1_evidence_ids=[raw_record.record_id],
                insight_text=text,
                metadata={
                    "legacy_entry_id": entry_id,
                    "fitness_delta": metadata.get("fitness_delta", 0.0),
                },
                insight_id=str(uuid.uuid4()),
                embedding=[float(v) for v in (embedding or [])],
                embedding_model=self.config.embedding_model,
            )
            self.raw_memory.link_stage2(raw_record.record_id, local_record.insight_id)

            # Assign with async-free fallback for migration startup.
            cluster = self._assign_cluster_sync(local_record)
            if cluster and cluster.cluster_id not in local_record.stage3_cluster_ids:
                local_record.stage3_cluster_ids.append(cluster.cluster_id)
                self.storage.update_stage2(local_record)
            imported += 1

        self.storage.set_step_floor(max_step)
        self.storage.set_legacy_migrated()
        logger.info("Imported %d legacy insight entries into three-stage memory", imported)

    def _assign_cluster_sync(self, stage2_record: LocalInsightRecord) -> Optional[InsightCluster]:
        # Used only for startup migration to avoid async dependency before event loop is available.
        existing_clusters = self.storage.list_stage3()
        best_cluster: Optional[InsightCluster] = None
        best_similarity = -1.0

        for cluster in existing_clusters:
            if (
                self.config.max_cluster_members is not None
                and len(cluster.member_insight_ids) >= self.config.max_cluster_members
            ):
                continue
            similarity = _cosine_similarity(stage2_record.embedding, cluster.representative_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        now = time.time()
        if best_cluster and best_similarity >= self.config.clustering_similarity_threshold:
            if stage2_record.insight_id not in best_cluster.member_insight_ids:
                best_cluster.member_insight_ids.append(stage2_record.insight_id)
            members = [
                self.storage.get_stage2(mid).embedding
                for mid in best_cluster.member_insight_ids
                if self.storage.get_stage2(mid) is not None
            ]
            best_cluster.representative_embedding = ClusterManager(
                self.config
            ).compute_cluster_representation(members)
            best_cluster.updated_timestamp = now
            best_cluster.updated_step_index = stage2_record.step_index
            best_cluster.metadata["cluster_size"] = len(best_cluster.member_insight_ids)
            self.storage.upsert_stage3(best_cluster)
            return best_cluster

        cluster = InsightCluster(
            cluster_id=str(uuid.uuid4()),
            member_insight_ids=[stage2_record.insight_id],
            representative_embedding=list(stage2_record.embedding),
            summary=None,
            created_timestamp=now,
            created_step_index=stage2_record.step_index,
            updated_timestamp=now,
            updated_step_index=stage2_record.step_index,
            metadata={"cluster_size": 1, "representation_mode": self.config.cluster_representation_mode},
        )
        self.storage.upsert_stage3(cluster)
        return cluster

    async def record_problem_insight(
        self, report: Dict[str, Any], metrics: Dict[str, Any]
    ) -> Optional[str]:
        if not self.config.enabled:
            return None

        step = self.storage.next_step()
        raw_record = self.raw_memory.create_record(
            step_index=step,
            problem_id=self.problem_id,
            task_id=self.task_id,
            source_component="controller",
            source_file=self.source_file,
            event_type="baseline_evaluation",
            raw_payload={"report": report, "metrics": metrics},
            execution_context={"phase": "baseline"},
            raw_report_text=json.dumps(report, ensure_ascii=True),
        )

        fitness_score = get_fitness_score(metrics, self.feature_dimensions)
        insight_text = await self.local_memory.generate_problem_insight(report, fitness_score)
        local_record = self.local_memory.create_insight(
            step_index=step,
            problem_id=self.problem_id,
            task_id=self.task_id,
            stage1_evidence_ids=[raw_record.record_id],
            insight_text=insight_text,
            metadata={
                "kind": "problem",
                "fitness_score": fitness_score,
            },
        )

        self.raw_memory.link_stage2(raw_record.record_id, local_record.insight_id)
        cluster = await self.clustered_memory.assign_or_create_cluster(local_record)
        if cluster.cluster_id not in local_record.stage3_cluster_ids:
            local_record.stage3_cluster_ids.append(cluster.cluster_id)
            self.storage.update_stage2(local_record)

        await self._maybe_recluster(step)

        self.problem_insight = insight_text
        return insight_text

    async def record_change_insight(
        self,
        parent_metrics: Dict[str, Any],
        child_metrics: Dict[str, Any],
        report: Dict[str, Any],
        changes_summary: str,
        baseline_fitness: float,
    ) -> Optional[str]:
        if not self.config.enabled:
            return None

        step = self.storage.next_step()

        parent_fitness = get_fitness_score(parent_metrics, self.feature_dimensions)
        child_fitness = get_fitness_score(child_metrics, self.feature_dimensions)
        fitness_delta = child_fitness - baseline_fitness

        raw_record = self.raw_memory.create_record(
            step_index=step,
            problem_id=self.problem_id,
            task_id=self.task_id,
            source_component="process_parallel",
            source_file=self.source_file,
            event_type="child_evaluation",
            raw_payload={
                "parent_metrics": parent_metrics,
                "child_metrics": child_metrics,
                "report": report,
                "changes_summary": changes_summary,
                "baseline_fitness": baseline_fitness,
                "parent_fitness": parent_fitness,
                "child_fitness": child_fitness,
                "fitness_delta": fitness_delta,
            },
            execution_context={"phase": "evolution_iteration"},
            raw_report_text=json.dumps(report, ensure_ascii=True),
        )

        insight_text = await self.local_memory.generate_change_insight(
            report=report,
            parent_fitness=parent_fitness,
            child_fitness=child_fitness,
            fitness_delta=fitness_delta,
            changes_summary=changes_summary,
        )

        local_record = self.local_memory.create_insight(
            step_index=step,
            problem_id=self.problem_id,
            task_id=self.task_id,
            stage1_evidence_ids=[raw_record.record_id],
            insight_text=insight_text,
            metadata={
                "kind": "change",
                "parent_fitness": parent_fitness,
                "child_fitness": child_fitness,
                "fitness_delta": fitness_delta,
                "problem_insight": self.problem_insight or "",
            },
        )

        self.raw_memory.link_stage2(raw_record.record_id, local_record.insight_id)
        cluster = await self.clustered_memory.assign_or_create_cluster(local_record)
        if cluster.cluster_id not in local_record.stage3_cluster_ids:
            local_record.stage3_cluster_ids.append(cluster.cluster_id)
            self.storage.update_stage2(local_record)

        await self._maybe_recluster(step)
        return insight_text

    async def build_context(
        self,
        problem_insight: str,
        top_k: Optional[int] = None,
    ) -> str:
        if not self.config.enabled:
            return ""

        top_members = top_k if top_k is not None else self.config.retrieval_top_members_per_cluster
        return await self.retriever.build_context(
            problem_insight,
            top_clusters=self.config.retrieval_top_clusters,
            top_members_per_cluster=top_members,
        )

    async def _maybe_recluster(self, step: int) -> None:
        interval = max(1, int(self.config.recluster_every_n_steps))
        if step % interval == 0:
            await self.clustered_memory.full_recluster(step)


# Backward-compatible aliases for existing call sites.
InsightManager = ThreeStageMemoryManager
InsightMemoryStore = ThreeStageMemoryManager


def _sanitize_memory_name(name: str) -> str:
    safe = "".join(ch for ch in (name or "") if ch.isalnum() or ch in "_-")
    return safe or "insight_memory"


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(vec1, vec2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b

    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def _vector_average(vectors: Sequence[List[float]]) -> List[float]:
    if not vectors:
        return []

    length = len(vectors[0])
    if any(len(v) != length for v in vectors):
        return []

    acc = [0.0] * length
    for vec in vectors:
        for i, value in enumerate(vec):
            acc[i] += value

    inv = 1.0 / float(len(vectors))
    return [value * inv for value in acc]


def _vector_weighted_average(vectors: Sequence[List[float]], weights: Sequence[float]) -> List[float]:
    if not vectors or not weights or len(vectors) != len(weights):
        return []

    length = len(vectors[0])
    if any(len(v) != length for v in vectors):
        return []

    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        return _vector_average(vectors)

    acc = [0.0] * length
    for vec, weight in zip(vectors, weights):
        for i, value in enumerate(vec):
            acc[i] += weight * value

    inv = 1.0 / total_weight
    return [value * inv for value in acc]


def _fallback_problem_insight(report: Dict[str, Any]) -> str:
    metrics_summary = report.get("metrics_summary", {}) if isinstance(report, dict) else {}
    names = {name.lower() for name in metrics_summary.keys()}

    if any("time" in name or "latency" in name or "runtime" in name for name in names):
        return (
            "The dominant challenge appears to be runtime/latency, suggesting excess work per run. "
            "The bottleneck is likely algorithmic complexity or repeated computation rather than output quality."
        )
    if any("memory" in name or "peak" in name or "alloc" in name for name in names):
        return (
            "The main issue looks like memory pressure and data movement. "
            "This hints at inefficient data structures or redundant allocations during execution."
        )
    return (
        "The main bottleneck appears to be overall computational workload. "
        "This suggests opportunities to reduce redundant work or improve algorithmic efficiency."
    )


def _fallback_change_insight(changes_summary: str, fitness_delta: float) -> str:
    summary = (changes_summary or "").lower()
    change_type = "implementation refactor"

    if "cache" in summary or "memo" in summary:
        change_type = "caching or memoization"
    elif "dynamic" in summary and "program" in summary:
        change_type = "dynamic programming"
    elif "vector" in summary or "numpy" in summary:
        change_type = "vectorization"
    elif "prune" in summary or "cut" in summary:
        change_type = "pruning or search-space reduction"
    elif "parallel" in summary or "concurrent" in summary:
        change_type = "parallelization"

    impact = "improved" if fitness_delta > 0 else "regressed" if fitness_delta < 0 else "stabilized"
    return f"The change resembles {change_type} and the overall fitness {impact}."
