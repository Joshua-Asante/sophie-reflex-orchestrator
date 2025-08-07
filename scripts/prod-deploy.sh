#!/bin/bash
# SOPHIE Production Deployment Script
# For Windsurf Cloud Deployment

set -e

echo "🚀 Deploying SOPHIE to Production..."

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

# Check environment variables
check_env_vars() {
    print_status "Checking environment variables..."
    
    required_vars=(
        "OPENAI_API_KEY"
        "GOOGLE_API_KEY"
        "ANTHROPIC_API_KEY"
        "NEXT_PUBLIC_API_URL"
        "NEXT_PUBLIC_ORCHESTRATOR_URL"
        "NEXT_PUBLIC_HITL_URL"
    )
    
    missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        print_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        exit 1
    fi
    
    print_success "All required environment variables are set"
}

# Build production images
build_prod_images() {
    print_status "Building production Docker images..."
    
    # Build orchestrator
    docker build -f Dockerfile.optimized --target production -t sophie-orchestrator:prod .
    print_success "Orchestrator production image built"
    
    # Build frontend
    cd ../ui
    docker build -f Dockerfile.prod -t sophie-frontend:prod .
    print_success "Frontend production image built"
    cd ../sophie-reflex-orchestrator
}

# Create production configuration
create_prod_config() {
    print_status "Creating production configuration..."
    
    # Create nginx production config
    cat > nginx.prod.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream frontend {
        server sophie-frontend:3000;
    }
    
    upstream orchestrator {
        server sophie-orchestrator:8000;
    }
    
    upstream hitl {
        server sophie-orchestrator:8001;
    }
    
    server {
        listen 80;
        server_name _;
        
        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
        
        # Orchestrator API
        location /api/ {
            proxy_pass http://orchestrator;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
        
        # HITL Interface
        location /hitl/ {
            proxy_pass http://hitl;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
        
        # Health checks
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
EOF
    
    # Create production Prometheus config
    cat > monitoring/prometheus.prod.yml << EOF
global:
  scrape_interval: 30s
  evaluation_interval: 30s

scrape_configs:
  - job_name: 'sophie-orchestrator'
    static_configs:
      - targets: ['sophie-orchestrator:8000']
    metrics_path: '/metrics'

  - job_name: 'sophie-frontend'
    static_configs:
      - targets: ['sophie-frontend:3000']
    metrics_path: '/api/metrics'

  - job_name: 'sophie-chromadb'
    static_configs:
      - targets: ['chromadb:8000']
    metrics_path: '/api/v1/heartbeat'

  - job_name: 'sophie-redis'
    static_configs:
      - targets: ['redis:6379']
EOF
    
    print_success "Production configuration created"
}

# Deploy to production
deploy_production() {
    print_status "Deploying to production..."
    
    # Use windsurf.yaml for production deployment
    docker-compose -f windsurf.yaml up -d
    
    print_success "Production deployment started"
    print_status "Services available at:"
    echo "  - Frontend: http://localhost:3000"
    echo "  - Orchestrator API: http://localhost:8000"
    echo "  - HITL Interface: http://localhost:8001"
    echo "  - ChromaDB: http://localhost:8004"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3002"
}

# Check production health
check_prod_health() {
    print_status "Checking production service health..."
    
    # Wait for services to be ready
    sleep 30
    
    # Check orchestrator
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
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

# Setup SSL certificates (if available)
setup_ssl() {
    print_status "Setting up SSL certificates..."
    
    if [ -d "ssl" ]; then
        print_success "SSL certificates found"
    else
        print_warning "SSL certificates not found. Creating self-signed for development..."
        mkdir -p ssl
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout ssl/nginx.key -out ssl/nginx.crt \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    fi
}

# Main execution
main() {
    print_status "Starting SOPHIE production deployment..."
    
    check_env_vars
    build_prod_images
    create_prod_config
    setup_ssl
    deploy_production
    check_prod_health
    
    print_success "SOPHIE production deployment complete!"
    print_status "Next steps:"
    echo "  - Configure your domain in Windsurf"
    echo "  - Set up SSL certificates"
    echo "  - Configure monitoring alerts"
    echo "  - Set up backup strategies"
}

# Run main function
main "$@" 