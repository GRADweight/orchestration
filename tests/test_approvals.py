"""Freigabe-Warteschlange."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from gradweight_orchestration.approvals import ApprovalQueue, ApprovalStatus
from gradweight_orchestration.audit import AuditLog
from gradweight_orchestration.errors import ApprovalError


def submit(queue: ApprovalQueue, run_id: str = "run1") -> str:
    request = queue.submit(
        run_id=run_id,
        agent="kreditoren",
        call_id="c1",
        tool="kreditor_buchen",
        arguments={"betrag_chf": 12430.5},
        side_effect="money_moving",
        amount="12430.50",
        reason="über Limit",
    )
    assert request.status is ApprovalStatus.PENDING
    return request.id


def test_approve_flow_with_audit() -> None:
    audit = AuditLog()
    queue = ApprovalQueue(audit=audit)
    request_id = submit(queue)
    assert [r.id for r in queue.pending()] == [request_id]
    decided = queue.approve(request_id, by="m.keller", note="geprüft")
    assert decided.status is ApprovalStatus.APPROVED
    assert decided.status.approved
    assert decided.decided_by == "m.keller"
    assert decided.effective_arguments == {"betrag_chf": 12430.5}
    assert queue.pending() == []
    record = audit.records()[-1]
    assert record.kind == "approval_decided"
    assert record.data["status"] == "approved"
    assert record.run_id == "run1"


def test_approve_with_changes_and_reject() -> None:
    queue = ApprovalQueue()
    first, second = submit(queue), submit(queue)
    changed = queue.approve_with_changes(first, by="cfo", arguments={"betrag_chf": 12000})
    assert changed.status is ApprovalStatus.MODIFIED
    assert changed.effective_arguments == {"betrag_chf": 12000}
    rejected = queue.reject(second, by="cfo", reason="Doppelte Rechnung")
    assert rejected.status is ApprovalStatus.REJECTED
    assert not rejected.status.approved
    assert rejected.note == "Doppelte Rechnung"
    assert rejected.effective_arguments == {"betrag_chf": 12430.5}
    assert rejected.to_json()["status"] == "rejected"


def test_rules() -> None:
    queue = ApprovalQueue()
    request_id = submit(queue)
    with pytest.raises(ApprovalError, match="Agenten"):
        queue.approve(request_id, by="agent:kreditoren")
    with pytest.raises(ApprovalError, match="fehlt"):
        queue.approve(request_id, by="  ")
    with pytest.raises(ApprovalError, match="Begründung"):
        queue.reject(request_id, by="cfo", reason=" ")
    queue.approve(request_id, by="cfo")
    with pytest.raises(ApprovalError, match="bereits entschieden"):
        queue.reject(request_id, by="cfo", reason="zu spät")
    with pytest.raises(ApprovalError, match="Kein Antrag"):
        queue.get("apr_gibtsnicht")
    with pytest.raises(ApprovalError, match="Kein Antrag"):
        queue.approve("apr_gibtsnicht", by="cfo")


def test_concurrent_decisions_only_one_wins() -> None:
    queue = ApprovalQueue()
    request_id = submit(queue)
    outcomes: list[str] = []

    def decide(name: str) -> None:
        try:
            queue.approve(request_id, by=name)
            outcomes.append("ok")
        except ApprovalError:
            outcomes.append("abgelehnt")

    threads = [threading.Thread(target=decide, args=(f"person{i}",)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("ok") == 1


def test_run_filter_treats_wildcards_literally() -> None:
    queue = ApprovalQueue()
    submit(queue, run_id="run_a/kind-1")
    submit(queue, run_id="runXa/kind-1")
    submit(queue, run_id="r%/kind-1")
    assert [r.run_id for r in queue.pending(run_id="run_a")] == ["run_a/kind-1"]
    assert [r.run_id for r in queue.pending(run_id="r%")] == ["r%/kind-1"]


def test_persistence_and_run_filter(tmp_path: Path) -> None:
    path = tmp_path / "freigaben.sqlite"
    queue = ApprovalQueue(path, id_factory=iter(["a1", "a2", "a3"]).__next__)
    submit(queue, run_id="run1")
    submit(queue, run_id="run1/buchhaltung-1")
    submit(queue, run_id="run2")
    queue.close()
    reopened = ApprovalQueue(path)
    assert [r.id for r in reopened.pending(run_id="run1")] == ["a1", "a2"]
    assert len(reopened.pending()) == 3
    reopened.approve("a3", by="cfo")
    assert ApprovalQueue(path).get("a3").status is ApprovalStatus.APPROVED
