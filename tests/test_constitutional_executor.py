"""
Test Constitutional AI Operating System Executor

This test demonstrates SOPHIE's transformation from a refined meta-LLM to a
Constitutional AI Operating System that can execute real infrastructure changes
through conversational intent.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime

from core.constitutional_executor import (
    ConstitutionalExecutor,
    execute_constitutional_directive,
    ConstitutionalRole
)


class TestConstitutionalExecutor:
    """Test the Constitutional AI Operating System Executor."""
    
    @pytest.fixture
    def executor(self):
        """Create a constitutional executor instance."""
        return ConstitutionalExecutor()
    
    @pytest.mark.asyncio
    async def test_interpret_directive_navigator(self, executor):
        """Test Φ Navigator role - interpreting high-level directives."""
        
        directive = "Add Redis caching to the authentication system and deploy to staging"
        
        with patch('core.constitutional_executor.optimized_llm_call') as mock_llm:
            mock_llm.return_value = '''
            {
                "type": "implementation",
                "description": "Implement Redis caching for authentication system with staging deployment",
                "priority": 4,
                "approval_level": "approval",
                "context": {
                    "infrastructure": "redis",
                    "deployment": "staging",
                    "component": "authentication"
                }
            }
            '''
            
            result = await executor._interpret_directive_navigator(directive)
            
            assert result.type.value == "implementation"
            assert "Redis caching" in result.description
            assert result.priority == 4
            assert result.approval_level.value == "approval"
            assert "redis" in result.context.get("infrastructure", "")
    
    @pytest.mark.asyncio
    async def test_generate_plan_integrator(self, executor):
        """Test Σ Integrator role - generating execution plans."""
        
        from core.autonomous_executor import Directive, DirectiveType, ApprovalLevel
        
        directive = Directive(
            id="test_directive",
            type=DirectiveType.IMPLEMENTATION,
            description="Add Redis caching to authentication system",
            priority=4,
            approval_level=ApprovalLevel.APPROVAL
        )
        
        with patch('core.constitutional_executor.optimized_llm_call') as mock_llm:
            mock_llm.return_value = '''
            ```yaml
            name: "redis_caching_implementation"
            steps:
              - name: "setup_redis"
                action: "infrastructure"
                description: "Set up Redis instance for caching"
              
              - name: "modify_auth_code"
                action: "code_change"
                description: "Add Redis caching to authentication logic"
              
              - name: "deploy_to_staging"
                action: "deploy"
                description: "Deploy changes to staging environment"
              
              - name: "run_tests"
                action: "test"
                description: "Run authentication tests with caching"
            ```
            '''
            
            plan = await executor._generate_plan_integrator(directive)
            
            assert plan.directive == directive
            assert "redis_caching_implementation" in plan.plan_yaml
            assert plan.confidence_score > 0
            assert plan.risk_assessment is not None
            assert plan.approval_required is True
            assert len(plan.digital_signature) > 0
    
    @pytest.mark.asyncio
    async def test_validate_plan_constitutional(self, executor):
        """Test Δ Diff Engine role - validating constitutional guardrails."""
        
        from core.constitutional_executor import ConstitutionalPlan
        from core.autonomous_executor import Directive, DirectiveType, ApprovalLevel
        
        directive = Directive(
            id="test_directive",
            type=DirectiveType.IMPLEMENTATION,
            description="Test directive",
            priority=3,
            approval_level=ApprovalLevel.APPROVAL
        )
        
        plan = ConstitutionalPlan(
            id="test_plan",
            directive=directive,
            plan_yaml="test yaml plan",
            confidence_score=0.8,
            risk_assessment={"risk_level": "low"},
            approval_required=True,
            digital_signature="test_signature"
        )
        
        with patch('core.constitutional_executor.optimized_tool_call') as mock_tool:
            mock_tool.return_value = {
                "approved": True,
                "validation_results": {
                    "security_check": True,
                    "constitutional_check": True,
                    "risk_assessment": "low"
                }
            }
            
            result = await executor._validate_plan_constitutional(plan)
            
            assert result["approved"] is True
            assert result["validation_results"]["security_check"] is True
    
    @pytest.mark.asyncio
    async def test_execute_via_cicd(self, executor):
        """Test Σ Integrator role - executing via CI/CD."""
        
        from core.constitutional_executor import ConstitutionalPlan
        from core.autonomous_executor import Directive, DirectiveType, ApprovalLevel
        
        directive = Directive(
            id="test_directive",
            type=DirectiveType.IMPLEMENTATION,
            description="Test directive",
            priority=3,
            approval_level=ApprovalLevel.APPROVAL
        )
        
        plan = ConstitutionalPlan(
            id="test_plan",
            directive=directive,
            plan_yaml="test yaml plan",
            confidence_score=0.8,
            risk_assessment={"risk_level": "low"},
            approval_required=True,
            digital_signature="test_signature"
        )
        
        with patch('core.constitutional_executor.optimized_tool_call') as mock_tool:
            mock_tool.return_value = {
                "pipeline_id": "pipeline_123",
                "status": "success",
                "staging_url": "https://staging.sophie-ai.com",
                "artifact_urls": [
                    "https://artifacts.sophie-ai.com/sophie-app-v2.1.0.dmg"
                ],
                "dashboard_url": "https://dashboard.sophie-ai.com/pipeline/123"
            }
            
            result = await executor._execute_via_cicd(plan)
            
            assert result["pipeline_id"] == "pipeline_123"
            assert result["staging_url"] == "https://staging.sophie-ai.com"
            assert len(result["artifact_urls"]) > 0
            assert plan.status == "completed"
    
    @pytest.mark.asyncio
    async def test_full_constitutional_execution(self, executor):
        """Test complete constitutional execution flow."""
        
        directive = "Add Redis caching to authentication system and deploy to staging"
        
        with patch('core.constitutional_executor.optimized_llm_call') as mock_llm, \
             patch('core.constitutional_executor.optimized_tool_call') as mock_tool:
            
            # Mock LLM responses for different steps
            mock_llm.side_effect = [
                # Navigator response
                '''
                {
                    "type": "implementation",
                    "description": "Implement Redis caching for authentication system",
                    "priority": 4,
                    "approval_level": "approval",
                    "context": {"infrastructure": "redis"}
                }
                ''',
                # Integrator response
                '''
                ```yaml
                name: "redis_caching_implementation"
                steps:
                  - name: "setup_redis"
                    action: "infrastructure"
                    description: "Set up Redis instance"
                ```
                ''',
                # Confidence assessment
                "0.85",
                # Risk assessment
                '''
                {
                    "risk_level": "low",
                    "risk_factors": ["Standard deployment"],
                    "mitigation_strategies": ["Rollback plan"],
                    "change_summary": "Add Redis caching to auth system"
                }
                '''
            ]
            
            # Mock tool responses
            mock_tool.side_effect = [
                # Trust gate validation
                {
                    "approved": True,
                    "validation_results": {"security_check": True}
                },
                # CI/CD execution
                {
                    "pipeline_id": "pipeline_123",
                    "staging_url": "https://staging.sophie-ai.com",
                    "artifact_urls": ["https://artifacts.sophie-ai.com/app.dmg"],
                    "dashboard_url": "https://dashboard.sophie-ai.com/pipeline/123"
                }
            ]
            
            result = await executor.interpret_and_execute_constitutional(directive)
            
            assert result["status"] == "completed"
            assert "Redis caching" in result["directive"]
            assert result["staging_url"] == "https://staging.sophie-ai.com"
            assert len(result["artifact_urls"]) > 0
            assert result["dashboard_url"] is not None
    
    @pytest.mark.asyncio
    async def test_constitutional_roles(self):
        """Test that all constitutional roles are properly defined."""
        
        roles = [role.value for role in ConstitutionalRole]
        expected_roles = ["Φ", "Σ", "Ω", "Δ", "Ψ"]
        
        assert roles == expected_roles
        assert len(roles) == 5
    
    def test_vision_vs_current_state(self):
        """Test documentation of the gap between current and vision states."""
        
        # Current state capabilities (what's already working)
        current_capabilities = [
            "Multi-model consensus engine",
            "Trust-based decision making", 
            "Basic autonomous execution",
            "File system tools",
            "Human-in-the-loop approval"
        ]
        
        # Vision capabilities (what needs to be built)
        vision_capabilities = [
            "CI/CD integration layer",
            "Infrastructure deployment",
            "Artifact generation",
            "Live staging URLs",
            "Constitutional guardrails",
            "Digital signing",
            "Real-time monitoring"
        ]
        
        assert len(current_capabilities) > 0
        assert len(vision_capabilities) > 0
        
        # The gap is the difference between current and vision
        gap_size = len(vision_capabilities)
        assert gap_size > 0
        
        print(f"\n🎯 Constitutional AI Operating System Gap Analysis:")
        print(f"✅ Current capabilities: {len(current_capabilities)}")
        print(f"🚧 Vision capabilities needed: {len(vision_capabilities)}")
        print(f"📊 Gap size: {gap_size} capabilities to implement")


# Example usage and demonstration
async def demonstrate_constitutional_execution():
    """Demonstrate the constitutional execution process."""
    
    print("\n🧠 SOPHIE Constitutional AI Operating System Demo")
    print("=" * 60)
    
    # Example directive
    directive = "Add Redis caching to the authentication system and deploy to staging"
    print(f"\n📝 Human Directive: {directive}")
    
    # Simulate the constitutional execution process
    steps = [
        ("Φ Navigator", "Interpreting high-level directive"),
        ("Σ Integrator", "Generating execution plan"),
        ("Δ Diff Engine", "Validating constitutional guardrails"),
        ("Ω Anchor", "Requesting human approval"),
        ("Σ Integrator", "Executing via CI/CD pipeline"),
        ("Ψ Memory", "Storing execution in memory")
    ]
    
    print("\n🔄 Constitutional Execution Process:")
    for i, (role, description) in enumerate(steps, 1):
        print(f"  {i}. {role}: {description}")
        await asyncio.sleep(0.5)  # Simulate processing time
    
    # Simulate result
    result = {
        "status": "completed",
        "directive": directive,
        "plan_id": "plan_1703123456",
        "staging_url": "https://staging.sophie-ai.com",
        "artifact_urls": [
            "https://artifacts.sophie-ai.com/sophie-app-v2.1.0.dmg",
            "https://artifacts.sophie-ai.com/sophie-app-v2.1.0.deb"
        ],
        "dashboard_url": "https://dashboard.sophie-ai.com/pipeline/1703123456"
    }
    
    print(f"\n✅ Execution Result:")
    print(f"  • Status: {result['status']}")
    print(f"  • Plan ID: {result['plan_id']}")
    print(f"  • Staging URL: {result['staging_url']}")
    print(f"  • Artifacts: {len(result['artifact_urls'])} generated")
    print(f"  • Dashboard: {result['dashboard_url']}")
    
    print(f"\n🎯 This demonstrates the transformation from:")
    print(f"   Current: Refined meta-LLM")
    print(f"   Vision: Constitutional AI Operating System")
    print(f"   Capability: Real infrastructure changes via conversational intent")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_constitutional_execution()) 