import torch


class VectorQuantizer(torch.nn.Module):
    def __init__(self, num_codebook_vectors: int, latent_dim: int = 256) -> None:
        super().__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim

        self.embedding = torch.nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q: torch.Tensor = self.embedding(min_encoding_indices)
        z_q = z_q.view(z.shape)
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2)
        return z_q