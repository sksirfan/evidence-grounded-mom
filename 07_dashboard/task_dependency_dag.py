"""Evidence-grounded task dependency DAG utilities for the MoM project.

Edge direction is prerequisite -> dependent. This module contains no model calls.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, List
import networkx as nx


def normalize_tasks(tasks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized, seen = [], set()
    for i, raw in enumerate(tasks or [], 1):
        if not isinstance(raw, dict):
            continue
        description = str(raw.get("task") or raw.get("description") or raw.get("claim") or "").strip()
        if not description:
            continue
        task_id = str(raw.get("task_id") or f"T{i}").strip()
        if task_id in seen:
            raise ValueError(f"Duplicate task_id: {task_id}")
        seen.add(task_id)
        normalized.append({
            "task_id": task_id,
            "task": description,
            # Do not infer an assignee from the person who reported the task.
            "owner": raw.get("owner") or raw.get("assignee"),
            "reported_by": raw.get("reported_by") or raw.get("speaker"),
            "deadline": raw.get("deadline"),
            "status": raw.get("status", "UNKNOWN"),
            "source_claim_number": raw.get("source_claim_number"),
            "source_claim": raw.get("source_claim", description),
            "source_start": raw.get("source_start", raw.get("start")),
            "source_end": raw.get("source_end", raw.get("end")),
            "source_evidence": raw.get("source_evidence", {}),
        })
    return normalized


def normalize_dependencies(dependencies: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for raw in dependencies or []:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("prerequisite_id") or raw.get("source") or "").strip()
        target = str(raw.get("dependent_id") or raw.get("target") or "").strip()
        if not source or not target:
            continue
        normalized.append({
            "prerequisite_id": source,
            "dependent_id": target,
            "relation": raw.get("relation", "MUST_PRECEDE"),
            "evidence_text": raw.get("evidence_text"),
            "evidence_start": raw.get("evidence_start"),
            "evidence_end": raw.get("evidence_end"),
            "evidence_speaker": raw.get("evidence_speaker"),
            "evidence_turn_id": raw.get("evidence_turn_id"),
            "nli_label": raw.get("nli_label"),
            "nli_entailment": raw.get("nli_entailment"),
            "nli_contradiction": raw.get("nli_contradiction"),
            "verification_status": raw.get("verification_status", "NOT_EVALUATED"),
            "verification_reason": raw.get("verification_reason"),
        })
    return normalized


def build_dependency_dag(tasks, dependencies) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.graph["edge_direction"] = "prerequisite_id -> dependent_id"
    for task in normalize_tasks(tasks):
        graph.add_node(task["task_id"], **task)
    for dep in normalize_dependencies(dependencies):
        source, target = dep["prerequisite_id"], dep["dependent_id"]
        if source not in graph:
            raise ValueError(f"Unknown prerequisite task: {source}")
        if target not in graph:
            raise ValueError(f"Unknown dependent task: {target}")
        if source == target:
            raise ValueError(f"Task {source} cannot depend on itself.")
        if graph.has_edge(source, target):
            graph[source][target].setdefault("evidence_records", []).append(dep)
        else:
            graph.add_edge(source, target, **dep, evidence_records=[dep])
    return graph


def validate_dependency_dag(graph: nx.DiGraph) -> Dict[str, Any]:
    if not isinstance(graph, nx.DiGraph):
        raise TypeError("graph must be a networkx.DiGraph")
    if nx.is_directed_acyclic_graph(graph):
        return {"is_dag": True, "cycle_detected": False, "cycles": [],
                "message": "Valid DAG: no directed cycles detected."}
    try:
        cycle = nx.find_cycle(graph, orientation="original")
        path = [cycle[0][0]] + [edge[1] for edge in cycle]
        cycles = [path]
    except nx.NetworkXNoCycle:
        cycles = []
    return {"is_dag": False, "cycle_detected": True, "cycles": cycles,
            "message": "Cycle detected. Review the dependency relationships."}


def get_topological_order(graph: nx.DiGraph) -> List[Dict[str, Any]]:
    validation = validate_dependency_dag(graph)
    if not validation["is_dag"]:
        raise ValueError(f"Cannot order tasks; cycle detected: {validation['cycles']}")
    return [dict(graph.nodes[node_id], execution_order=position)
            for position, node_id in enumerate(nx.topological_sort(graph), 1)]


def get_ready_tasks(graph: nx.DiGraph) -> List[Dict[str, Any]]:
    return [dict(data) for node, data in graph.nodes(data=True) if graph.in_degree(node) == 0]


def get_downstream_impact(graph: nx.DiGraph, task_id: str) -> Dict[str, Any]:
    """Return every task transitively affected if task_id changes or is delayed."""
    if task_id not in graph:
        raise KeyError(f"Unknown task_id: {task_id}")
    impacted_ids = set(nx.descendants(graph, task_id))
    if validate_dependency_dag(graph)["is_dag"]:
        order = [x["task_id"] for x in get_topological_order(graph)]
        impacted_ids = [node for node in order if node in impacted_ids]
    else:
        impacted_ids = sorted(impacted_ids)
    return {
        "task_id": task_id,
        "impacted_task_ids": list(impacted_ids),
        "impacted_tasks": [dict(graph.nodes[node]) for node in impacted_ids],
        "impact_count": len(impacted_ids),
    }


def graph_to_dict(graph: nx.DiGraph) -> Dict[str, Any]:
    validation = validate_dependency_dag(graph)
    result = {
        "nodes": [dict(data) for _, data in graph.nodes(data=True)],
        "edges": [{"prerequisite_id": u, "dependent_id": v, **dict(data)}
                  for u, v, data in graph.edges(data=True)],
        "has_dependencies": graph.number_of_edges() > 0,
        "execution_sequence_is_constrained": graph.number_of_edges() > 0,
        "validation": validation,
    }
    if validation["is_dag"]:
        result["topological_order"] = [x["task_id"] for x in get_topological_order(graph)]
        result["ready_tasks"] = [x["task_id"] for x in get_ready_tasks(graph)]
    else:
        result["topological_order"], result["ready_tasks"] = [], []
    return result


def build_dag_result(tasks, dependencies) -> Dict[str, Any]:
    return graph_to_dict(build_dependency_dag(tasks, dependencies))


if __name__ == "__main__":
    tasks = [{"task_id": "T1", "task": "Approve design"},
             {"task_id": "T2", "task": "Implement design"},
             {"task_id": "T3", "task": "Test implementation"}]
    edges = [{"prerequisite_id": "T1", "dependent_id": "T2"},
             {"prerequisite_id": "T2", "dependent_id": "T3"}]
    graph = build_dependency_dag(tasks, edges)
    assert validate_dependency_dag(graph)["is_dag"]
    assert [x["task_id"] for x in get_topological_order(graph)] == ["T1", "T2", "T3"]
    assert get_downstream_impact(graph, "T1")["impacted_task_ids"] == ["T2", "T3"]
    assert get_ready_tasks(graph)[0]["task_id"] == "T1"
    print("DAG engine tests: PASSED")
