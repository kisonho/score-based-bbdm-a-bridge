from .networks import VQGAN


def build(in_channels: int, *, f: int = 4, latent_dim: int = 3, num_codebook_vectors: int = 8192) -> VQGAN:
    return VQGAN(in_channels, f=f, latent_dim=latent_dim, num_codebook_vectors=num_codebook_vectors, return_details=True)
