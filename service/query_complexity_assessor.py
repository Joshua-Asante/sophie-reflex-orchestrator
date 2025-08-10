import yaml
from model_gateway.gateway import dispatch_to_model

class QueryComplexityAssessor:
    """
    Analyzes a user's query to assess its complexity and characteristics.

    This class uses a fast LLM to break down a query into a structured
    YAML object, which can then be used by the main router to make an
    intelligent routing decision.
    """
    def __init__(self):
        """
        The assessor is now self-contained and uses the central model gateway directly.
        """
        self.assessment_model = "capability:general_agentic"  # Use capability routing

    def assess(self, query: str) -> dict:
        """
        Assesses a single user query.

        Args:
            query (str): The user's input query.

        Returns:
            dict: A dictionary containing the structured assessment of the query.
                  Returns a dictionary with an 'error' key if assessment fails.
        """
        prompt = f"""
You are a highly efficient Query Complexity Assessor. Your sole purpose is to analyze the user query below and return a single, valid YAML object containing your analysis. Do not include any other text or explanations in your response.

The YAML object must have the following fields:
- num_explicit_questions: The integer count of direct questions asked.
- num_implicit_subtasks: The integer count of implied tasks or steps needed to answer fully.
- has_constraints: A boolean (true/false) indicating if the query includes specific limitations (e.g., "in 3 paragraphs", "using Python 3.9").
- needs_synthesis: A boolean (true/false) indicating if the query requires combining information from multiple domains or concepts to form a new idea.
- needs_divergence: A boolean (true/false) indicating if the query asks for brainstorming, creativity, or generating multiple distinct ideas.
- has_ambiguity: A boolean (true/false) indicating if the query is vague or could be interpreted in multiple ways.
- estimated_difficulty_score: An integer from 1 (simple fact) to 5 (highly complex research task).
- recommended_model_tier: A string ('flash', 'pro', or 'ultra') based on the complexity.
- initial_decomposition_plan: A brief, one-sentence plan for how to approach the query.

Query:
---
"{query}"
---
"""
        # Format the payload for the model gateway
        messages_payload = [
            {"role": "user", "parts": [{"text": prompt}]}
        ]

        # Dispatch the assessment task to the gateway
        response_data = dispatch_to_model(self.assessment_model, {"contents": messages_payload})

        # Process the response from the gateway
        if "error" in response_data:
            print(f"Assessor Error: Could not get response from gateway. Details: {response_data.get('details')}")
            return {"error": "Failed to get assessment from model."}

        try:
            # Extract the raw text from the model's response
            text_content = "".join(
                part["text"] for part in response_data["candidates"][0]["content"]["parts"] if "text" in part
            )

            # Clean the response to ensure it's valid YAML (removes markdown backticks)
            cleaned_yaml_str = text_content.strip().replace("```yaml", "").replace("```", "").strip()

            # Parse the YAML string into a Python dictionary
            assessment_dict = yaml.safe_load(cleaned_yaml_str)

            if not isinstance(assessment_dict, dict):
                raise yaml.YAMLError("Model response was not a valid dictionary.")

            return assessment_dict

        except (yaml.YAMLError, IndexError, KeyError, TypeError) as e:
            print(f"Assessor Parsing Error: {e}. Raw content: '{text_content}'")
            return {"error": f"Failed to parse assessment YAML from model response: {e}"}
