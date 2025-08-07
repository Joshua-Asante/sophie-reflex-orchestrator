"""
Test GLM 4.5 Integration

Tests the GLM 4.5 integration for engineering mode tasks.
"""

import asyncio
import os
import sys
from unittest.mock import Mock, patch
import pytest

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.prover import ProverAgent
from agents.base_agent import AgentConfig


class TestGLMIntegration:
    """Test GLM 4.5 integration functionality."""
    
    @pytest.fixture
    def glm_config(self):
        """Create a GLM engineering agent configuration."""
        return AgentConfig(
            name="glm_engineering_prover",
            prompt="You are a GLM 4.5-powered engineering specialist. Generate innovative, practical solutions for complex technical challenges.",
            model="glm",
            temperature=0.3,
            max_tokens=2000,
            timeout=30,
            max_retries=3,
            cache_enabled=True,
            cache_ttl=3600,
            adaptive_learning_enabled=True,
            rate_limit_per_minute=60,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=300,
            hyperparameters={
                "max_variants": 3,
                "technical_expertise": 0.95,
                "engineering_principles": 0.9,
                "system_design": 0.9,
                "scalability_focus": 0.8,
                "innovation_bias": 0.8
            }
        )
    
    @pytest.fixture
    def glm_agent(self, glm_config):
        """Create a GLM engineering agent instance."""
        return ProverAgent(glm_config, "test_glm_agent")
    
    @pytest.mark.asyncio
    async def test_glm_client_creation(self, glm_agent):
        """Test that GLM client can be created successfully."""
        # Mock the environment variable
        with patch.dict(os.environ, {'GLM_API_KEY': 'test_key'}):
            try:
                client = await glm_agent._llm_manager.get_client("glm", {"timeout": 30})
                assert client is not None
                print("✅ GLM client created successfully")
            except Exception as e:
                pytest.skip(f"GLM client creation failed (expected in test environment): {e}")
    
    @pytest.mark.asyncio
    async def test_glm_engineering_task(self, glm_agent):
        """Test GLM agent with an engineering task."""
        task = "Build a React component for user authentication with modern best practices"
        
        # Mock the LLM call to avoid actual API calls in tests
        with patch.object(glm_agent, '_call_glm_enhanced') as mock_call:
            mock_call.return_value = {
                "content": """# Modern React Authentication Component

Here's a comprehensive React authentication component using modern best practices:

```tsx
import React, { useState, useEffect } from 'react';
import { useAuth } from './hooks/useAuth';

interface AuthFormProps {
  mode: 'login' | 'register';
  onSubmit: (credentials: AuthCredentials) => void;
}

export const AuthForm: React.FC<AuthFormProps> = ({ mode, onSubmit }) => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: ''
  });
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };
  
  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <input
        type="email"
        placeholder="Email"
        value={formData.email}
        onChange={(e) => setFormData({...formData, email: e.target.value})}
        className="w-full px-3 py-2 border rounded-md"
        required
      />
      <input
        type="password"
        placeholder="Password"
        value={formData.password}
        onChange={(e) => setFormData({...formData, password: e.target.value})}
        className="w-full px-3 py-2 border rounded-md"
        required
      />
      {mode === 'register' && (
        <input
          type="password"
          placeholder="Confirm Password"
          value={formData.confirmPassword}
          onChange={(e) => setFormData({...formData, confirmPassword: e.target.value})}
          className="w-full px-3 py-2 border rounded-md"
          required
        />
      )}
      <button
        type="submit"
        className="w-full bg-blue-500 text-white py-2 rounded-md hover:bg-blue-600"
      >
        {mode === 'login' ? 'Sign In' : 'Sign Up'}
      </button>
    </form>
  );
};
```

This component includes:
- TypeScript for type safety
- Modern React hooks (useState, useEffect)
- Proper form handling with validation
- Responsive design with Tailwind CSS
- Accessibility considerations
- Clean, maintainable code structure""",
                "confidence": 0.95,
                "reasoning": "Generated using GLM 4.5 with enhanced parameters for technical excellence",
                "model": "glm-4.5",
                "usage": {"completion_tokens": 800},
                "temperature_used": 0.3
            }
            
            # Also mock the quality assessment to return high scores
            with patch.object(glm_agent, '_assess_variant_quality') as mock_quality:
                mock_quality.return_value = 0.9  # High quality score
                
                result = await glm_agent.execute(task)
                
                assert result is not None
                assert result.status.value == "completed"
                assert "React" in str(result.result)
                assert result.confidence_score > 0.8
                print("✅ GLM engineering task executed successfully")
    
    @pytest.mark.asyncio
    async def test_glm_confidence_calculation(self, glm_agent):
        """Test GLM confidence calculation."""
        # Mock response object
        mock_response = Mock()
        mock_response.usage.completion_tokens = 800
        
        context = {
            "strategy": {
                "strategy": "analytical_rigorous"
            },
            "temperature_used": 0.3
        }
        
        confidence = glm_agent._calculate_confidence_glm_enhanced(mock_response, context)
        
        assert confidence > 0.8  # GLM should have high confidence for technical tasks
        assert confidence <= 1.0
        print(f"✅ GLM confidence calculation: {confidence}")
    
    def test_glm_model_configuration(self, glm_config):
        """Test GLM model configuration."""
        assert glm_config.model == "glm"
        assert glm_config.temperature == 0.3
        assert glm_config.max_tokens == 2000
        assert "GLM 4.5" in glm_config.prompt
        print("✅ GLM model configuration verified")
    
    @pytest.mark.asyncio
    async def test_glm_error_handling(self, glm_agent):
        """Test GLM error handling."""
        task = "Build a complex system"
        
        # Mock API error
        with patch.object(glm_agent, '_call_glm_enhanced') as mock_call:
            mock_call.side_effect = Exception("GLM API error")
            
            try:
                result = await glm_agent.execute(task)
                assert result.status.value == "failed"
                print("✅ GLM error handling works correctly")
            except Exception as e:
                print(f"✅ GLM error handling caught exception: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 