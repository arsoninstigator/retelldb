"""
Tests for audience classification and query tag signal detection.
"""

from retelldb.routing.audience import AudienceClassifier
from retelldb.routing.classifier import QueryClassifier
from retelldb.models.schema import AudienceType, GraphNode, NodeType


def test_audience_classifier():
    classifier = AudienceClassifier()

    # Developer context
    assert classifier.classify("how do i refresh an api token in the python sdk") == AudienceType.DEVELOPER
    assert classifier.classify("POST /v1/chat endpoint parameters and headers") == AudienceType.DEVELOPER

    # User context
    assert classifier.classify("how to update my billing credit card on dashboard") == AudienceType.USER
    assert classifier.classify("invite team members to my workspace settings") == AudienceType.USER

    # Hint override
    assert classifier.classify("api endpoint", audience_hint=AudienceType.USER) == AudienceType.USER


def test_query_tag_classifier():
    classifier = QueryClassifier()
    graph_tags = [
        GraphNode(title="auth", type=NodeType.TAG),
        GraphNode(title="tokens", type=NodeType.TAG),
        GraphNode(title="billing", type=NodeType.TAG),
    ]

    # Strong match
    has_signal, matched, conf = classifier.evaluate_tag_signal("how to handle auth tokens", graph_tags)
    assert has_signal is True
    assert "auth" in matched
    assert "tokens" in matched
    assert conf >= 0.8

    # Vague query without matching tags
    has_signal, matched, conf = classifier.evaluate_tag_signal("something broke completely yesterday", graph_tags)
    assert has_signal is False
    assert len(matched) == 0
