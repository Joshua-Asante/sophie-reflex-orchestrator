# Sophie Reflex Orchestrator

A minimal but powerful swarm-based orchestration system to demonstrate reflexive AI coordination. Built with a genetic algorithm (GA) style agent loop (Prover → Evaluator → Refiner), memory persistence, trust scoring, and basic Human-in-the-Loop (HITL) override functionality.

## 🚀 Quick Start

### 1-Click Demo with Docker

```bash
# Clone the repository
git clone https://github.com/your-org/sophie-reflex-orchestrator.git
cd sophie-reflex-orchestrator

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_key
# ANTHROPIC_API_KEY=your_anthropic_key

# Start the system
docker-compose up -d

# Access the web interface
open http://localhost:8001
```

### Minimal Install

```bash
# Install dependencies
pip install -r requirements.txt

# Run a simple task
python main.py --task "Design a sustainable city transportation system"

# Run in interactive mode
python main.py --interactive
```

## 🏗️ Architecture

### Core Components

```
sophie-reflex-orchestrator/
├── main.py                    # Entry point with CLI interface
├── orchestrator.py           # Core GA loop logic
├── agents/                   # Agent implementations
│   ├── base_agent.py        # Abstract base class
│   ├── prover.py           # Plan execution agents
│   ├── evaluator.py        # Solution scoring agents
│   └── refiner.py          # Agent mutation/optimization
├── memory/                   # Memory and persistence
│   ├── vector_store.py     # Embedding storage (ChromaDB/SQLite)
│   └── trust_tracker.py    # Trust score management
├── governance/               # Policy and audit
│   ├── policy_engine.py    # Policy enforcement
│   └── audit_log.py        # Comprehensive logging
├── ui/                      # Human-in-the-Loop interface
│   ├── webhook_server.py   # FastAPI HITL server
│   └── static/            # Web UI assets
└── configs/                 # Configuration files
    ├── system.yaml        # System settings
    ├── agents.yaml        # Agent configurations
    ├── rubric.yaml        # Evaluation criteria
    └── policies.yaml      # Governance policies
```

### GA Loop Flow

1. **Task Input** → System receives a task to solve
2. **Prover Phase** → Multiple agents generate solution variants
3. **Evaluator Phase** → Solutions are scored using evaluation rubric
4. **HITL Check** → Human review triggered if policies require it
5. **Refiner Phase** → Population is optimized (mutation/crossover)
6. **Trust Update** → Agent trust scores are adjusted
7. **Repeat** → Loop continues until convergence or limits reached

## 🎯 Key Features

### Genetic Algorithm Optimization
- **Population-based evolution** of agent strategies
- **Mutation and crossover** for solution improvement
- **Elite preservation** of high-performing agents
- **Convergence detection** with configurable thresholds

### Memory & Learning
- **Vector embeddings** for semantic similarity search
- **Trust scoring** with performance-based adjustments
- **Audit logging** of all decisions and changes
- **Persistent storage** across sessions

### Human-in-the-Loop (HITL)
- **Web-based dashboard** for plan approval/rejection
- **Configurable policies** for when to involve humans
- **Real-time intervention** during execution
- **Decision history** and analytics

### Trust & Governance
- **Dynamic trust scoring** based on performance
- **Policy engine** for behavior governance
- **Resource limits** and safety constraints
- **Comprehensive audit trail**

## 📖 Usage Guide

### Command Line Interface

```bash
# Run a single task
python main.py --task "Write a Python script to analyze stock data"

# Interactive mode with full control
python main.py --interactive

# Run only the HITL server
python main.py --server

# Save results to JSON file
python main.py --task "Design a logo" --output results.json

# Use custom configuration
python main.py --task "Plan a trip" --config custom_config.yaml

# Verbose logging
python main.py --task "Debug this code" --verbose
```

### Interactive Mode Commands

```
sophie> help                    # Show available commands
sophie> task "Your task here"   # Run a specific task
sophie> status                  # Show orchestrator status
sophie> stats                   # Display detailed statistics
sophie> agents                  # List current agents
sophie> pause                   # Pause execution
sophie> resume                  # Resume execution
sophie> stop                    # Stop orchestrator
sophie> quit                    # Exit program
```

### Web Interface

Access the HITL dashboard at `http://localhost:8001`:

- **Pending Plans**: Review and approve/reject generated solutions
- **Decision History**: Track all human interventions
- **Real-time Stats**: Monitor system performance
- **Plan Management**: Fork, modify, or regenerate solutions

## ⚙️ Configuration

### System Configuration (`configs/system.yaml`)

```yaml
# API Keys and Authentication
api_keys:
  openai: "your-openai-api-key"
  anthropic: "your-anthropic-api-key"

# Model Configuration
models:
  default: "openai"
  openai:
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 2000
  anthropic:
    model: "claude-3-sonnet-20240229"
    temperature: 0.7
    max_tokens: 2000

# Memory Backend
memory:
  backend: "chromadb"  # or "sqlite-vss"
  chromadb:
    host: "localhost"
    port: 8000
    collection_name: "sophie_memory"

# GA Loop Settings
genetic_algorithm:
  population_size: 5
  mutation_rate: 0.1
  crossover_rate: 0.3
  elite_count: 2
  max_generations: 50
```

### Agent Configuration (`configs/agents.yaml`)

```yaml
# Define different types of agents with varying strategies
provers:
  - name: "creative_prover"
    prompt: "You are a creative problem solver..."
    model: "openai"
    temperature: 0.8
    hyperparameters:
      creativity: 0.9
      detail_level: 0.7

evaluators:
  - name: "quality_evaluator"
    prompt: "Evaluate the quality of the solution..."
    model: "openai"
    temperature: 0.2
    hyperparameters:
      weight_completeness: 0.4
      weight_accuracy: 0.4
```

### Evaluation Rubric (`configs/rubric.yaml`)

```yaml
# Define evaluation criteria and scoring
categories:
  completeness:
    description: "How completely does the solution address the task?"
    weight: 0.3
    levels:
      excellent: "Fully addresses all requirements"
      good: "Addresses most requirements"
      fair: "Addresses basic requirements"
      poor: "Misses key requirements"
```

### Governance Policies (`configs/policies.yaml`)

```yaml
# HITL and governance settings
hitl:
  enabled: true
  approval_threshold: 0.7
  rejection_threshold: 0.4
  require_human_review:
    - trust_score < 0.6
    - confidence_score < 0.5
    - contains_sensitive_content: true

trust:
  initial_score: 0.5
  adjustments:
    successful_execution: +0.1
    failed_execution: -0.2
    human_approval: +0.15
    human_rejection: -0.25
```

## 🔧 Development

### Setting Up Development Environment

```bash
# Clone and navigate
git clone https://github.com/your-org/sophie-reflex-orchestrator.git
cd sophie-reflex-orchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run tests
python -m pytest tests/

# Start development server
python main.py --interactive
```

### Project Structure

```
sophie-reflex-orchestrator/
├── main.py                 # Main entry point
├── orchestrator.py        # Core orchestration logic
├── agents/               # Agent implementations
├── memory/               # Memory and trust systems
├── governance/           # Policy and audit systems
├── ui/                   # User interface
├── configs/              # Configuration files
├── tests/                # Test files
├── docker-compose.yaml   # Docker configuration
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Adding New Agent Types

1. **Create agent class** inheriting from `BaseAgent`:

```python
from agents.base_agent import BaseAgent, AgentConfig, AgentResult

class CustomAgent(BaseAgent):
    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
        # Your agent logic here
        pass
    
    async def generate_prompt(self, task: str, context: Dict[str, Any] = None) -> str:
        # Generate prompts for your agent
        pass
```

2. **Add configuration** in `configs/agents.yaml`:

```yaml
custom_agents:
  - name: "custom_agent"
    prompt: "Your custom prompt..."
    model: "openai"
    temperature: 0.7
    hyperparameters:
      custom_param: 0.5
```

3. **Register agent** in the orchestrator initialization.

### Extending Evaluation Criteria

Add new evaluation categories in `configs/rubric.yaml`:

```yaml
categories:
  new_category:
    description: "Description of new category"
    weight: 0.2
    levels:
      excellent: "Excellent criteria"
      good: "Good criteria"
      fair: "Fair criteria"
      poor: "Poor criteria"
```

## 🐳 Docker Deployment

### Production Deployment

```bash
# Build and start all services
docker-compose -f docker-compose.yml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Clean up volumes
docker-compose down -v
```

### Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │  Orchestrator   │    │   HITL Server   │
│   (Port 80/443) │◄──►│   (Port 8000)   │◄──►│   (Port 8001)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │    ChromaDB     │              │
         └─────────────►│   (Port 8002)   │◄─────────────┘
                        └─────────────────┘
                               │
                    ┌─────────────────┐
                    │     Redis       │
                    │   (Port 6379)   │
                    └─────────────────┘
```

### Environment Variables

Create a `.env` file:

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# System Settings
LOG_LEVEL=INFO
MAX_GENERATIONS=50
POPULATION_SIZE=5

# Database Settings
MEMORY_BACKEND=chromadb
CHROMADB_HOST=chromadb
CHROMADB_PORT=8000

# HITL Settings
HITL_ENABLED=true
HITL_TIMEOUT=300
```

## 📊 Monitoring & Analytics

### System Metrics

The system provides comprehensive metrics:

- **Trust Scores**: Agent performance tracking
- **Generation Progress**: Evolution over time
- **Intervention Rates**: Human involvement frequency
- **Resource Usage**: Memory, CPU, and API calls
- **Solution Quality**: Score improvements over generations

### Accessing Metrics

```bash
# View orchestrator status
curl http://localhost:8000/status

# Get health check
curl http://localhost:8001/api/health

# View detailed statistics
python main.py --interactive
# Then use: stats
```

### Exporting Data

```bash
# Export audit logs
python -c "
from governance.audit_log import AuditLog
audit = AuditLog()
await audit.export_audit_data('audit_export.json')
"

# Export trust data
python -c "
from memory.trust_tracker import TrustTracker
tracker = TrustTracker()
await tracker.export_trust_data('trust_export.json')
"
```

## 🧪 Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=sophie --cov-report=html

# Run specific test file
pytest tests/test_orchestrator.py

# Run with verbose output
pytest tests/ -v
```

### Test Structure

```
tests/
├── test_orchestrator.py     # Core orchestrator tests
├── test_agents/            # Agent tests
├── test_memory/            # Memory system tests
├── test_governance/        # Policy and audit tests
├── test_ui/               # UI and HITL tests
└── fixtures/             # Test fixtures and data
```

## 🤝 Contributing

### Development Workflow

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes and add tests**
4. **Run tests**: `pytest tests/`
5. **Lint code**: `black .` and `flake8 .`
6. **Commit changes**: `git commit -m "Add amazing feature"`
7. **Push branch**: `git push origin feature/amazing-feature`
8. **Create Pull Request**

### Code Style

- **Python**: Follow PEP 8, use Black for formatting
- **Documentation**: Docstrings for all public methods
- **Tests**: Maintain >80% test coverage
- **Configuration**: YAML files for all configurable parameters

### Reporting Issues

When reporting bugs or requesting features:

1. **Use GitHub Issues** with clear titles
2. **Provide environment details**: OS, Python version, dependencies
3. **Include error messages** and stack traces
4. **Add reproduction steps** for bugs
5. **Suggest solutions** for feature requests

## 📈 Performance & Scaling

### Optimization Tips

1. **Model Selection**: Use appropriate models for each task
2. **Caching**: Enable Redis caching for repeated queries
3. **Batch Processing**: Process multiple agents in parallel
4. **Memory Management**: Regular cleanup of old data
5. **Resource Limits**: Set appropriate timeouts and limits

### Scaling Considerations

- **Horizontal Scaling**: Multiple orchestrator instances
- **Load Balancing**: Distribute tasks across instances
- **Database Scaling**: Use managed ChromaDB for large datasets
- **API Rate Limits**: Monitor and adjust LLM API usage

## 🔒 Security

### Best Practices

1. **API Keys**: Never commit keys to version control
2. **Environment Variables**: Use `.env` files for secrets
3. **Input Validation**: Sanitize all user inputs
4. **Access Control**: Implement authentication in production
5. **Audit Logging**: Enable comprehensive audit trails

### Data Privacy

- **Encryption**: Encrypt sensitive data at rest
- **Anonymization**: Remove PII from logs and metrics
- **Retention**: Configure data retention policies
- **Compliance**: Follow GDPR/CCPA as applicable

## 📚 API Reference

### Main CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `--task` | Run a single task | `python main.py --task "Write a poem"` |
| `--interactive` | Interactive mode | `python main.py --interactive` |
| `--server` | HITL server only | `python main.py --server` |
| `--config` | Custom config file | `python main.py --task "X" --config custom.yaml` |
| `--output` | Save results to JSON | `python main.py --task "X" --output results.json` |

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/plans/submit` | POST | Submit plan for review |
| `/api/plans/pending` | GET | Get pending plans |
| `/api/plans/{id}/approve` | POST | Approve a plan |
| `/api/plans/{id}/reject` | POST | Reject a plan |
| `/api/plans/{id}/fork` | POST | Fork/regenerate plan |
| `/api/decisions` | GET | Get decision history |
| `/api/stats` | GET | System statistics |

## 🚨 Troubleshooting

### Common Issues

**API Key Problems**
```bash
# Check if keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Test API connectivity
python -c "import openai; print('OpenAI OK')"
python -c "import anthropic; print('Anthropic OK')"
```

**Docker Issues**
```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs sophie-orchestrator
docker-compose logs chromadb

# Restart services
docker-compose restart
```

**Memory/Database Issues**
```bash
# Check ChromaDB connection
curl http://localhost:8002/api/v1/heartbeat

# Clear memory (careful!)
rm -rf memory/*
```

**Performance Issues**
```bash
# Monitor resource usage
docker stats

# Check logs for errors
docker-compose logs --tail=100 sophie-orchestrator

# Reduce population size for testing
# Edit configs/system.yaml -> genetic_algorithm.population_size
```

### Getting Help

- **Documentation**: Check this README and inline code comments
- **Issues**: Search existing GitHub issues
- **Discussions**: Join GitHub discussions for questions
- **Debug Mode**: Use `--verbose` flag for detailed logging

## 🗺️ Roadmap

### Version 1.1 (Planned)
- [ ] Multi-modal agent support (images, audio)
- [ ] Advanced genetic algorithms (NSGA-II, SPEA2)
- [ ] Distributed execution across multiple nodes
- [ ] Enhanced web UI with real-time visualization

### Version 1.2 (Future)
- [ ] Integration with external tools and APIs
- [ ] Advanced learning and adaptation
- [ ] Enterprise authentication and authorization
- [ ] Performance optimization and scaling

### Version 2.0 (Long-term)
- [ ] Full SOPHIE Core platform compatibility
- [ ] Autonomous agent swarms
- [ ] Advanced reasoning and planning
- [ ] Production-ready enterprise features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT models and API
- **Anthropic** for Claude models
- **ChromaDB** for vector database functionality
- **FastAPI** for web framework
- **Docker** for containerization
- **Contributors** and the open-source community

## 📞 Contact

- **GitHub Issues**: [Report bugs and request features](https://github.com/your-org/sophie-reflex-orchestrator/issues)
- **Discussions**: [Join community discussions](https://github.com/your-org/sophie-reflex-orchestrator/discussions)
- **Email**: [Contact team](mailto:team@sophie-ai.org)

---

**Sophie Reflex Orchestrator** - Building the future of collaborative AI systems 🤖✨