

CONTRAST_ENHANCER_INSTR = """
Make the background COMPLETELY pure hex code #000000 (BLACK),
remove noise and tiny artifacts,
sharpen all circles/ovals and shapes and make their EDGES pure hex code #FFFFFF (WHITE) with the SAME THICKNESS,
and enhance faint lines.
Do NOT produce gray gradients,
FLAT LIGHTING ONLY: Do not add any simulated light sources, shadows, or gradients. The lighting must be perfectly even across the entire tile,
Return ONLY the High Contrast, Monochromatic style image.
Make the objects look EXACTLY the same as the input image.
"""