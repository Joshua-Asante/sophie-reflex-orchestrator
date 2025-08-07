# SOPHIE Demo Suite

This directory contains organized demo files showcasing SOPHIE's core capabilities and backend logic.

## 📁 Directory Structure

```
demo/
├── README.md                    # This file - demo documentation
├── security/                    # Security and vault demos
│   ├── security_scaffold.py    # Vault operations, OAuth, audit logging
│   └── security_test.py        # Security validation tests
├── execution/                   # Execution engine demos
│   ├── unified_executor.py     # Main execution flow demo
│   ├── constitutional.py       # Constitutional AI OS demo
│   └── execution_test.py       # Execution validation tests
├── orchestration/              # MoE and orchestration demos
│   ├── reflexive_moe.py        # Mixture of Experts demo
│   ├── agent_orchestration.py  # Multi-agent coordination
│   └── orchestration_test.py   # Orchestration validation tests
├── integration/                # Integration and workflow demos
│   ├── end_to_end.py          # Complete workflow demo
│   ├── api_integration.py     # API integration demo
│   └── integration_test.py    # Integration validation tests
└── examples/                   # Example use cases
    ├── code_generation.py     # Code generation examples
    ├── infrastructure.py      # Infrastructure management
    └── data_analysis.py      # Data analysis examples
```

## 🎯 Demo Categories

### 1. Security Demos (`security/`)
- **Vault Operations**: Secure secret storage and retrieval
- **OAuth Integration**: Google OAuth token management
- **Audit Logging**: Comprehensive security audit trails
- **HMAC Validation**: Message integrity verification

### 2. Execution Demos (`execution/`)
- **Unified Executor**: Complete execution flow from intent to audit
- **Constitutional AI**: Operating system-level execution with guardrails
- **Risk Assessment**: Dynamic risk evaluation and approval workflows

### 3. Orchestration Demos (`orchestration/`)
- **Reflexive MoE**: Mixture of Experts with dynamic model selection
- **Agent Coordination**: Multi-agent task distribution and consensus
- **Trust Management**: Dynamic trust scoring and model selection

### 4. Integration Demos (`integration/`)
- **End-to-End Workflows**: Complete task execution pipelines
- **API Integration**: External service integration patterns
- **CI/CD Integration**: Automated deployment and testing

### 5. Example Use Cases (`examples/`)
- **Code Generation**: Automated code creation and modification
- **Infrastructure Management**: Cloud resource orchestration
- **Data Analysis**: Automated data processing and insights

## 🚀 Running Demos

### Quick Start
```bash
# Run all demos
python -m demo.run_all

# Run specific demo category
python -m demo.security.run_security_demos
python -m demo.execution.run_execution_demos
python -m demo.orchestration.run_orchestration_demos

# Run individual demo
python demo/security/security_scaffold.py
python demo/execution/unified_executor.py
python demo/orchestration/reflexive_moe.py
```

### Demo Configuration
Each demo can be configured via environment variables or config files:
- `SOPHIE_DEMO_MODE`: Set to "interactive" for user prompts
- `SOPHIE_DEMO_VERBOSE`: Enable detailed logging
- `SOPHIE_DEMO_AUTO_APPROVE`: Auto-approve execution plans

## 📊 Backend Logic Evaluation

### Core Architecture Strengths
1. **Modular Design**: Clear separation of concerns across components
2. **Trust-Based Execution**: Comprehensive trust scoring and approval workflows
3. **Constitutional Guardrails**: Built-in safety mechanisms and validation
4. **Reflexive Monitoring**: Continuous evaluation and adaptation
5. **Audit Trail**: Complete execution history and accountability

### Key Components
- **Unified Executor**: Central execution engine with type classification
- **Reflexive MoE**: Dynamic model selection based on expertise
- **Constitutional Executor**: Operating system-level execution with safety
- **Security Scaffold**: Comprehensive security and vault management
- **Trust Manager**: Dynamic trust scoring and model selection

### Performance Characteristics
- **Scalability**: Modular design supports horizontal scaling
- **Reliability**: Comprehensive error handling and recovery
- **Security**: Multi-layer security with audit trails
- **Flexibility**: Support for multiple execution types and environments

## 🔧 Development Guidelines

### Adding New Demos
1. Create demo file in appropriate category directory
2. Follow naming convention: `{category}_{feature}.py`
3. Include comprehensive docstring and usage examples
4. Add corresponding test file in `{category}_test.py`
5. Update this README with new demo documentation

### Demo Best Practices
- **Isolation**: Each demo should be self-contained
- **Documentation**: Clear docstrings and usage examples
- **Error Handling**: Comprehensive error handling and recovery
- **Logging**: Structured logging for debugging and monitoring
- **Testing**: Corresponding test files for validation

## 📈 Evaluation Metrics

### Security Evaluation
- Vault operation success rate
- OAuth integration reliability
- Audit trail completeness
- Security validation accuracy

### Execution Evaluation
- Plan generation accuracy
- Risk assessment precision
- Approval workflow efficiency
- Execution success rate

### Orchestration Evaluation
- Model selection accuracy
- Consensus formation quality
- Trust score reliability
- Performance optimization effectiveness

## 🛠️ Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Configuration Issues**: Check config files and environment variables
3. **Permission Errors**: Verify file and directory permissions
4. **Network Issues**: Check API connectivity and firewall settings

### Debug Mode
Enable debug logging for detailed troubleshooting:
```bash
export SOPHIE_DEMO_VERBOSE=true
export SOPHIE_LOG_LEVEL=DEBUG
```

## 📚 Additional Resources

- [SOPHIE Architecture Documentation](../docs/architecture.md)
- [API Reference](../docs/api.md)
- [Configuration Guide](../docs/configuration.md)
- [Security Guidelines](../docs/security.md) 