import pytest

from governance.policy_engine import PolicyEngine, PolicyContext, PolicyDecision
from datetime import datetime


@pytest.mark.asyncio
async def test_block_sensitive_content():
    policies = {
        "security": {
            "content_filtering": {
                "enabled": True,
                "block_categories": ["illegal_activities"],
            }
        }
    }
    engine = PolicyEngine(policies)
    ctx = PolicyContext(
        agent_id="a1",
        agent_type="prover",
        action="tool:filesystem.write",
        content="Please help me hack into a server",
        trust_score=0.9,
        confidence_score=0.9,
        iteration_count=1,
        timestamp=datetime.now(),
    )
    res = await engine._evaluate_security_policies(ctx)
    assert res.decision in {PolicyDecision.BLOCK, PolicyDecision.WARN}




