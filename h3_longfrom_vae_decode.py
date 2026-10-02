import torch


class H3LongformVAEDecode:
    """Decode a continuous MiniMax H3 video latent in temporal chunks.

    Uses the same H3 fake-bootstrap geometry as H3InfiniteTakeSampler:
    T = 2 + 5*B latent rows -> 17*B + 5 decoded frames.
    No spatial tiling is performed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE",),
            "latent": ("LATENT",),
            "decode_blocks": ("INT", {
                "default": 14, "min": 2, "max": 120, "step": 1,
                "tooltip": "H3 temporal decode chunk size in 17-frame blocks. 14 = 243 frames. Lower this if decode VRAM is too high."
            }),
        }}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frames")
    FUNCTION = "decode"
    CATEGORY = "H3/Decode"

    def decode(self, vae, latent, decode_blocks=14):
        samples = latent["samples"]

        # H3 AV latent: nested component 0 is video, component 1 is audio.
        if getattr(samples, "is_nested", False):
            components = samples.unbind()
            if not components:
                raise ValueError("H3 latent contains no components.")
            video = components[0]
        else:
            video = samples

        if video.ndim != 5:
            raise ValueError(
                "Expected H3 video latent [B,C,T,H,W], got %s" %
                (tuple(video.shape),)
            )

        latent_t = int(video.shape[2])
        if latent_t < 7 or (latent_t - 2) % 5 != 0:
            raise ValueError(
                "Not a continuous H3 latent: T=%d; expected T=2+5*B." %
                latent_t
            )

        blocks_total = (latent_t - 2) // 5
        db = max(2, int(decode_blocks))
        expected_frames = 17 * blocks_total + 5

        print(
            "[H3LongformVAE] input %s; chunk=%d blocks; expected=%d frames" %
            (tuple(video.shape), db, expected_frames), flush=True
        )

        frames_out = []
        b = 0
        while b < blocks_total:
            take = min(db, blocks_total - b)

            if b == 0:
                seg = video[:, :, :2 + 5 * take]
            else:
                seg = video[:, :, 5 * b:5 * (b + take) + 2]

            px = vae.decode(seg)
            if px.ndim == 5:
                px = px.reshape(-1, px.shape[-3], px.shape[-2], px.shape[-1])

            if b != 0:
                px = px[5:]

            # Immediately release GPU decoded frames. This is the important
            # difference from a naive loop that keeps every chunk on GPU.
            cpu_chunk = px.detach().cpu().half()
            frames_out.append(cpu_chunk)

            print(
                "[H3LongformVAE] chunk %d-%d -> %d frames" %
                (b, b + take - 1, int(cpu_chunk.shape[0])), flush=True
            )

            del seg, px, cpu_chunk
            b += take

        images = torch.cat(frames_out, dim=0)

        print(
            "[H3LongformVAE] done: %d frames (expected %d)" %
            (int(images.shape[0]), expected_frames), flush=True
        )
        return (images, int(images.shape[0]))


NODE_CLASS_MAPPINGS = {
    "H3LongformVAEDecode": H3LongformVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LongformVAEDecode": "H3 Longform VAE Decode",
}
