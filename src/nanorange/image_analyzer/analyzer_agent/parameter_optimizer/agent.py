from google.genai import types
from PIL import Image

from nanorange.settings import TEXT_MODEL
from nanorange.schemas import SuggestedParameters
from nanorange.image_analyzer.analyzer_agent.base_agent import BaseImageAgent
from nanorange.image_analyzer.analyzer_agent.parameter_optimizer.prompts import (
    PARAMETER_OPTIMIZER_INSTR,
)


class ParameterOptimizerAgent(BaseImageAgent):
    """Analyzes Cryo-TEM images and suggests optimal analysis parameters."""
    
    def __init__(self) -> None:
        """Initialize the Parameter Optimizer Agent."""
        super().__init__(
            model=TEXT_MODEL,
            instruction=PARAMETER_OPTIMIZER_INSTR
        )
    
    def analyze(self, image_path: str) -> SuggestedParameters:
        """
        Analyze an image and suggest optimal parameters.
        
        Args:
            image_path: Path to the Cryo-TEM image
            
        Returns:
            SuggestedParameters with AI-recommended values
        """
        image = Image.open(image_path)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[self.instruction, image],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SuggestedParameters,
            ),
        )
        
        self._update_token_usage(response)
        
        return SuggestedParameters.model_validate_json(response.text)

