from nanorange.settings import IMAGE_MODEL_COLORIZER
from nanorange.image_analyzer.analyzer_agent.base_agent import BaseImageAgent
from nanorange.image_analyzer.analyzer_agent.boundary_colorizer.prompts import BOUNDARY_COLORIZER_INSTR


class BoundaryColorizerAgent(BaseImageAgent):
    """Handles interaction with Gemini AI for boundary colorization."""
    
    def __init__(self) -> None:
        """Initialize the Boundary Colorizer Agent."""
        super().__init__(
            model=IMAGE_MODEL_COLORIZER,
            instruction=BOUNDARY_COLORIZER_INSTR
        )
