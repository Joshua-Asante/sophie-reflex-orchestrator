#!/usr/bin/env python3
"""
Unified Execution Engine Demo

This script demonstrates SOPHIE's unified execution engine in action,
showing the full execution flow from user intent to audit.
"""

import asyncio
import json
import time
from datetime import datetime


class UnifiedExecutorDemo:
    """
    Demo of SOPHIE's unified execution engine.
    
    Shows the full execution flow: User Intent → Intent Parser → Planner → Diff → Approval → Execution → Audit
    """
    
    def __init__(self):
        self.execution_history = []
    
    async def demonstrate_unified_execution(self, user_intent: str, context: dict = None) -> dict:
        """
        Demonstrate the unified execution flow.
        """
        
        print(f"\n🧠 SOPHIE Unified Execution Engine v1.0")
        print("=" * 60)
        print(f"🎯 User Intent: {user_intent}")
        print("=" * 60)
        
        # Step 1: Parse intent and classify execution type
        print(f"\n1️⃣ Intent Parser: Classifying execution type")
        execution_type = await self._classify_execution_type(user_intent)
        print(f"   ✅ Execution Type: {execution_type}")
        
        # Step 2: Generate execution plan using MoE
        print(f"\n2️⃣ MoE Planner: Generating execution plan")
        plan = await self._generate_execution_plan(user_intent, execution_type, context)
        print(f"   ✅ Plan ID: {plan['id']}")
        print(f"   📊 Trust Score: {plan['trust_score']:.1%}")
        print(f"   ⚠️ Risk Level: {plan['risk_level']}")
        print(f"   📋 Commands: {len(plan['commands'])}")
        
        # Step 3: Generate diff and risk assessment
        print(f"\n3️⃣ Diff Generator: Creating execution diff")
        diff = await self._generate_execution_diff(plan)
        print(f"   ✅ Changes: {len(diff['changes'])}")
        print(f"   ⚠️ Risk Factors: {len(diff['risk_factors'])}")
        
        # Step 4: Determine approval requirements
        requires_approval = plan['trust_score'] < 0.85 or plan['risk_level'] in ['high', 'critical']
        print(f"\n4️⃣ Approval Check: Determining approval requirements")
        print(f"   📊 Trust Score: {plan['trust_score']:.1%} (threshold: 85%)")
        print(f"   ⚠️ Risk Level: {plan['risk_level']}")
        print(f"   ✅ Requires Approval: {requires_approval}")
        
        # Step 5: Request approval if needed
        approval_granted = True
        if requires_approval:
            print(f"\n5️⃣ Approval Request: Requesting user approval")
            approval_granted = await self._request_execution_approval(plan, diff)
            if not approval_granted:
                print(f"   ❌ User denied approval")
                return {
                    "status": "rejected",
                    "reason": "User denied approval",
                    "plan_id": plan['id']
                }
            print(f"   ✅ User approved execution")
        else:
            print(f"\n5️⃣ Auto-Approval: Plan meets trust threshold")
            print(f"   ✅ Auto-approved (trust score: {plan['trust_score']:.1%})")
        
        # Step 6: Execute the plan
        print(f"\n6️⃣ Executor: Executing plan")
        execution_result = await self._execute_plan(plan)
        print(f"   ✅ Commands Executed: {execution_result['commands_executed']}")
        print(f"   ✅ Successful Commands: {execution_result['successful_commands']}")
        
        # Step 7: Update audit log and memory
        print(f"\n7️⃣ Audit Logger: Updating audit and memory")
        await self._update_audit_and_memory(plan, diff, approval_granted, execution_result)
        print(f"   ✅ Audit entry created")
        print(f"   📚 Memory updated")
        
        print(f"\n🎉 Unified execution completed successfully!")
        print("=" * 60)
        
        return {
            "status": "completed",
            "plan_id": plan['id'],
            "execution_type": execution_type,
            "trust_score": plan['trust_score'],
            "risk_level": plan['risk_level'],
            "execution_result": execution_result,
            "audit_entry_id": f"audit_{int(time.time())}"
        }
    
    async def _classify_execution_type(self, user_intent: str) -> str:
        """Classify the type of execution needed."""
        
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Simple keyword-based classification
        intent_lower = user_intent.lower()
        
        if any(word in intent_lower for word in ["gcp", "aws", "cloud", "deploy", "instance"]):
            return "cloud"
        elif any(word in intent_lower for word in ["database", "sql", "query", "table"]):
            return "database"
        elif any(word in intent_lower for word in ["file", "read", "write", "config"]):
            return "filesystem"
        elif any(word in intent_lower for word in ["api", "http", "request", "call"]):
            return "api"
        elif any(word in intent_lower for word in ["shell", "command", "script"]):
            return "shell"
        elif any(word in intent_lower for word in ["python", "code", "script"]):
            return "python"
        else:
            return "cli"
    
    async def _generate_execution_plan(self, user_intent: str, execution_type: str, context: dict = None) -> dict:
        """Generate an execution plan using simulated MoE."""
        
        await asyncio.sleep(1)  # Simulate processing time
        
        # Simulate plan generation based on execution type
        commands = []
        if execution_type == "cloud":
            commands = [
                {
                    "type": "cloud",
                    "provider": "gcp",
                    "service": "compute",
                    "operation": "list-instances",
                    "description": "List GCP compute instances"
                }
            ]
        elif execution_type == "database":
            commands = [
                {
                    "type": "database",
                    "operation": "SELECT",
                    "query": "SELECT * FROM users WHERE active = true",
                    "description": "Query active users"
                }
            ]
        elif execution_type == "filesystem":
            commands = [
                {
                    "type": "filesystem",
                    "operation": "read",
                    "path": "./config.yaml",
                    "description": "Read configuration file"
                }
            ]
        else:
            commands = [
                {
                    "type": "cli",
                    "command": "sophie exec",
                    "args": ["--intent", user_intent],
                    "description": "Execute SOPHIE command"
                }
            ]
        
        # Calculate trust score and risk level
        trust_score = 0.85 if execution_type in ["filesystem", "cli"] else 0.75
        risk_level = "high" if execution_type == "cloud" else "low"
        
        return {
            "id": f"plan_{int(time.time())}",
            "user_intent": user_intent,
            "execution_type": execution_type,
            "commands": commands,
            "trust_score": trust_score,
            "risk_level": risk_level,
            "estimated_duration": len(commands) * 2.0
        }
    
    async def _generate_execution_diff(self, plan: dict) -> dict:
        """Generate a detailed diff of what will change during execution."""
        
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Simulate current state
        before_state = {
            "files": ["config.yaml", "data.json"],
            "processes": ["sophie", "nginx"],
            "network": {"connections": 5},
            "memory": {"used": "2.1GB", "available": "5.9GB"}
        }
        
        # Simulate future state based on plan
        after_state = before_state.copy()
        if plan['execution_type'] == "filesystem":
            after_state["files"].append("new_file.txt")
        elif plan['execution_type'] == "cloud":
            after_state["network"]["connections"] += 2
        
        # Calculate changes
        changes = []
        for key in before_state:
            if before_state[key] != after_state[key]:
                changes.append({
                    "field": key,
                    "before": before_state[key],
                    "after": after_state[key]
                })
        
        # Identify risk factors
        risk_factors = []
        if plan['execution_type'] == "cloud":
            risk_factors.append("Cloud resource modification")
        if plan['trust_score'] < 0.8:
            risk_factors.append("Low confidence in plan")
        if plan['risk_level'] == "high":
            risk_factors.append("High risk operation")
        
        return {
            "before_state": before_state,
            "after_state": after_state,
            "changes": changes,
            "risk_factors": risk_factors
        }
    
    async def _request_execution_approval(self, plan: dict, diff: dict) -> bool:
        """Request user approval for execution."""
        
        await asyncio.sleep(0.5)  # Simulate processing time
        
        print(f"   📋 Approval Request:")
        print(f"      Intent: {plan['user_intent']}")
        print(f"      Type: {plan['execution_type']}")
        print(f"      Trust Score: {plan['trust_score']:.1%}")
        print(f"      Risk Level: {plan['risk_level']}")
        print(f"      Changes: {len(diff['changes'])}")
        print(f"      Risk Factors: {', '.join(diff['risk_factors'])}")
        
        # For demo purposes, always approve
        return True
    
    async def _execute_plan(self, plan: dict) -> dict:
        """Execute the plan using the appropriate executor."""
        
        await asyncio.sleep(1)  # Simulate execution time
        
        results = []
        for command in plan['commands']:
            # Simulate command execution
            await asyncio.sleep(0.5)
            
            result = {
                "status": "success",
                "command": command,
                "output": f"Executed {command.get('type', 'unknown')} command",
                "duration": 0.5
            }
            results.append(result)
        
        return {
            "commands_executed": len(results),
            "successful_commands": len(results),
            "results": results,
            "execution_time": time.time()
        }
    
    async def _update_audit_and_memory(self, plan: dict, diff: dict, approval_granted: bool, execution_result: dict):
        """Update audit log and memory with execution results."""
        
        await asyncio.sleep(0.3)  # Simulate processing time
        
        # Create audit entry
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "workflow_id": plan['id'],
            "user_intent": plan['user_intent'],
            "execution_plan": plan,
            "diff": diff,
            "approval_granted": approval_granted,
            "execution_result": execution_result,
            "trust_score": plan['trust_score']
        }
        
        # Add to execution history
        self.execution_history.append(audit_entry)


async def demonstrate_unified_executor():
    """Demonstrate SOPHIE's unified execution engine."""
    
    demo = UnifiedExecutorDemo()
    
    # Example intents to test different execution types
    intents = [
        "list all GCP compute instances",
        "read config.yaml and update database",
        "deploy application to staging",
        "analyze logs and generate report"
    ]
    
    print("\n🎯 SOPHIE Unified Execution Engine Demo")
    print("=" * 60)
    print("This demonstrates the full execution flow:")
    print("User Intent → Intent Parser → Planner → Diff → Approval → Execution → Audit")
    print("=" * 60)
    
    for i, intent in enumerate(intents, 1):
        print(f"\n📋 Example {i}: {intent}")
        print("-" * 40)
        
        result = await demo.demonstrate_unified_execution(intent)
        
        if result.get("status") == "completed":
            print(f"\n✅ Success! Final Result:")
            print(f"   • Plan ID: {result.get('plan_id', 'unknown')}")
            print(f"   • Type: {result.get('execution_type', 'unknown')}")
            print(f"   • Trust Score: {result.get('trust_score', 0):.1%}")
            print(f"   • Risk Level: {result.get('risk_level', 'unknown')}")
            print(f"   • Commands: {result.get('execution_result', {}).get('commands_executed', 0)}")
        else:
            print(f"\n❌ Execution failed: {result.get('reason', 'Unknown error')}")
        
        print("\n" + "="*60)
    
    print(f"\n📊 Execution Summary:")
    print(f"   • Total Executions: {len(demo.execution_history)}")
    print(f"   • Successful: {len([e for e in demo.execution_history if e.get('execution_result', {}).get('successful_commands', 0) > 0])}")
    print(f"   • Average Trust Score: {sum(e.get('trust_score', 0) for e in demo.execution_history) / len(demo.execution_history):.1%}")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_unified_executor()) 