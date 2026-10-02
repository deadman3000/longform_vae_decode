Decode a continuous MiniMax H3 video latent in temporal chunks. This does not use tiling but instead chunks latent into more manageable peices then passes to the output. 
It could be useful for longform video latents and those on low VRAM GPU's. 
Leave the chunk size at default unless on very low VRAM GPU's (lower uses less VRAM). To install simply drop into your custom_nodes folder and restart ComfyUI.
