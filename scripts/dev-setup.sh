#!/bin/bash
# SOPHIE Development Setup Script
# For Cursor IDE Integration and Local Development

set -e

echo "🚀 Setting up SOPHIE Development Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    print_status "Checking Docker status..."
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop first."
        exit 1
    fi
    print_success "Docker is running"
}

# Check if .env file exists
check_env() {
    print_status "Checking environment configuration..."
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from template..."
        cp env.example .env
        print_warning "Please edit .env file with your API keys before continuing"
        print_warning "Required key: OPENROUTER_API_KEY"
    else
        print_success "Environment file found"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p memory logs configs monitoring/grafana/provisioning
    print_success "Directories created"
}

# Build development images
build_images() {
    print_status "Building development Docker images..."

    # Build orchestrator
    docker build -f Dockerfile.optimized --target development -t sophie-orchestrator:dev .
    print_success "Orchestrator image built"

    # Build frontend
    cd ../ui
    docker build -f Dockerfile.dev -t sophie-frontend:dev .
    print_success "Frontend image built"
    cd ../sophie-reflex-orchestrator
}

# Start development services
start_dev_services() {
    print_status "Starting development services..."

    # Start with development profile
    docker-compose --profile development up -d

    print_success "Development services started"
    print_status "Services available at:"
    echo "  - Frontend: http://localhost:3000"
    echo "  - Orchestrator API: http://localhost:8002"
    echo "  - HITL Interface: http://localhost:8003"
    echo "  - ChromaDB: http://localhost:8004"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3002"
}

# Check service health
check_health() {
    print_status "Checking service health..."

    # Wait for services to be ready
    sleep 10

    # Check orchestrator
    if curl -f http://localhost:8002/api/health > /dev/null 2>&1; then
        print_success "Orchestrator is healthy"
    else
        print_warning "Orchestrator health check failed"
    fi

    # Check frontend
    if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        print_success "Frontend is healthy"
    else
        print_warning "Frontend health check failed"
    fi

    # Check ChromaDB
    if curl -f http://localhost:8004/api/v1/heartbeat > /dev/null 2>&1; then
        print_success "ChromaDB is healthy"
    else
        print_warning "ChromaDB health check failed"
    fi
}

# Setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring..."

    # Create Prometheus config if it doesn't exist
    if [ ! -f "monitoring/prometheus.dev.yml" ]; then
        cat > monitoring/prometheus.dev.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'sophie-orchestrator'
    static_configs:
      - targets: ['sophie-orchestrator-dev:8000']
    metrics_path: '/metrics'

  - job_name: 'sophie-frontend'
    static_configs:
      - targets: ['sophie-frontend-dev:3000']
    metrics_path: '/api/metrics'
EOF
        print_success "Prometheus configuration created"
    fi
}

# Main execution
main() {
    print_status "Starting SOPHIE development setup..."

    check_docker
    check_env
    create_directories
    build_images
    setup_monitoring
    start_dev_services
    check_health

    print_success "SOPHIE development environment is ready!"
    print_status "You can now:"
    echo "  - Edit code in Cursor with hot reload"
    echo "  - View logs: docker-compose logs -f"
    echo "  - Stop services: docker-compose down"
    echo "  - Restart services: docker-compose restart"
}

# Run main function
main "$@"
