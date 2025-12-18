

BOUNDARY_COLORIZER_INSTR = """
The current image shows all particle boundaries rendered in white.  
Recolor the boundary of each particle using a distinct color.

Requirements:
1. If particles overlap, touch, or share connected edges, their boundaries must be assigned different colors.
2. Use no more than 10 total colors.
3. Maintain consistent color assignment across the entire image (the same particle must always use the same color).
4. For particles that are very close together, nested, or overlapping, choose the highest-contrast color combinations to ensure their boundaries remain clearly distinguishable.

Do not alter the particles themselves—only recolor the boundary lines.
"""