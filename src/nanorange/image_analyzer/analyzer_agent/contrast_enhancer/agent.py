from nanorange.settings import IMAGE_MODEL
from nanorange.image_analyzer.analyzer_agent.base_agent import BaseImageAgent
from nanorange.image_analyzer.analyzer_agent.contrast_enhancer.prompts import CONTRAST_ENHANCER_INSTR


class ContrastEnhancerAgent(BaseImageAgent):
    """Handles interaction with Gemini AI for image contrast enhancement."""
    
    def __init__(self) -> None:
        """Initialize the Contrast Enhancer Agent."""
        super().__init__(
            model=IMAGE_MODEL,
            instruction=CONTRAST_ENHANCER_INSTR
        )
