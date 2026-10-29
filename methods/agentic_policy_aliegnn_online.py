"""Compact ALIEGNN-inspired online energy surrogate for fixed-site occupancy search.

This module deliberately has no filesystem access.  It trains from scratch on relaxed
energies observed in one current agent run.  The architecture preserves ALIEGNN's
atom graph + angular line-graph message-passing pattern, but is reduced to a small,
rotation-invariant network suitable for 8--200 labels.  It does not use the production
thermal-conductivity checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from agentic_policy_v2 import Candidate

MAX_CANDIDATE_BATCH = 3000
MAX_TRAINING_HISTORY = 201
MAX_GRAPH_NODES = 256
MAX_GRAPH_NEIGHBORS = 64
MAX_LINE_EDGES = 1_000_000


@dataclass(frozen=True)
class ALiEGNNSelection:
    index: int
    predicted_energy: float | None
    uncertainty: float | None
    novelty: float
    selector: str
    training_data_sha256: str | None = None
    ensemble_state_sha256: str | None = None


@dataclass(frozen=True)
class FixedLineGraphGeometry:
    """Frozen directed neighbor graph and edge-angle line graph."""

    num_nodes: int
    edge_sources: np.ndarray
    edge_targets: np.ndarray
    distances: np.ndarray
    line_sources: np.ndarray
    line_targets: np.ndarray
    angle_cosines: np.ndarray

    @property
    def num_edges(self) -> int:
        return int(len(self.edge_sources))

    @property
    def num_line_edges(self) -> int:
        return int(len(self.line_sources))

    @classmethod
    def from_atoms(cls, atoms, cutoff: float = 8.0, max_neighbors: int = 12):
        if len(atoms) < 2:
            raise ValueError("at least two graph sites are required")
        if len(atoms) > MAX_GRAPH_NODES:
            raise ValueError(f"graph node count exceeds limit {MAX_GRAPH_NODES}")
        if cutoff <= 0 or max_neighbors < 1:
            raise ValueError("cutoff and max_neighbors must be positive")
        if max_neighbors > MAX_GRAPH_NEIGHBORS:
            raise ValueError(
                f"line-graph bound requires max_neighbors <= {MAX_GRAPH_NEIGHBORS}")
        line_edge_bound = len(atoms) * max_neighbors * max(max_neighbors - 1, 0)
        if line_edge_bound > MAX_LINE_EDGES:
            raise ValueError(f"line-graph bound exceeds limit {MAX_LINE_EDGES}")
        distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
        vectors = np.asarray(atoms.get_all_distances(mic=True, vector=True), dtype=float)
        sources, targets, lengths, edge_vectors = [], [], [], []
        for target in range(len(atoms)):
            eligible = [
                int(source) for source in range(len(atoms))
                if source != target and 1e-10 < distances[source, target] <= cutoff
            ]
            # Site-index tie breaking makes symmetry-degenerate neighbor shells
            # stable against harmless floating-point changes after translation.
            order = sorted(
                eligible,
                key=lambda source: (round(float(distances[source, target]), 8), source),
            )
            for source in order[:max_neighbors]:
                sources.append(source)
                targets.append(target)
                lengths.append(float(distances[source, target]))
                edge_vectors.append(np.asarray(vectors[source, target], float))
        if not sources:
            raise ValueError("neighbor graph is empty")

        incoming: dict[int, list[int]] = {i: [] for i in range(len(atoms))}
        outgoing: dict[int, list[int]] = {i: [] for i in range(len(atoms))}
        for index, (source, target) in enumerate(zip(sources, targets)):
            incoming[target].append(index)
            outgoing[source].append(index)
        line_sources, line_targets, angle_cosines = [], [], []
        for center in range(len(atoms)):
            for edge_in in incoming[center]:
                first = sources[edge_in]
                vector_to_first = -edge_vectors[edge_in]
                for edge_out in outgoing[center]:
                    if targets[edge_out] == first:
                        continue
                    vector_to_second = edge_vectors[edge_out]
                    denominator = np.linalg.norm(vector_to_first) * np.linalg.norm(vector_to_second)
                    cosine = float(np.dot(vector_to_first, vector_to_second) / denominator)
                    line_sources.append(edge_in)
                    line_targets.append(edge_out)
                    angle_cosines.append(float(np.clip(cosine, -1.0, 1.0)))
        if not line_sources:
            raise ValueError("angular line graph is empty")
        return cls(
            num_nodes=len(atoms),
            edge_sources=np.asarray(sources, dtype=np.int64),
            edge_targets=np.asarray(targets, dtype=np.int64),
            distances=np.asarray(lengths, dtype=np.float32),
            line_sources=np.asarray(line_sources, dtype=np.int64),
            line_targets=np.asarray(line_targets, dtype=np.int64),
            angle_cosines=np.asarray(angle_cosines, dtype=np.float32),
        )


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values):
        return self.layers(values)


class _ALiEGNNBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.line_message = _MLP(2 * hidden_dim, hidden_dim, hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.atom_message = _MLP(2 * hidden_dim, hidden_dim, hidden_dim)
        self.atom_gate = nn.Linear(hidden_dim, hidden_dim)
        self.atom_norm = nn.LayerNorm(hidden_dim)

    def forward(self, atoms, edges, angles, edge_sources, edge_targets,
                line_sources, line_targets, edge_degree, atom_degree):
        line_input = torch.cat((edges[:, line_sources], angles.expand(len(edges), -1, -1)), dim=-1)
        line_values = self.line_message(line_input)
        edge_aggregate = torch.zeros_like(edges)
        edge_aggregate.index_add_(1, line_targets, line_values)
        edges = self.edge_norm(edges + edge_aggregate / edge_degree)

        message_input = torch.cat((atoms[:, edge_sources], edges), dim=-1)
        messages = self.atom_message(message_input) * torch.sigmoid(self.atom_gate(edges))
        atom_aggregate = torch.zeros_like(atoms)
        atom_aggregate.index_add_(1, edge_targets, messages)
        atoms = self.atom_norm(atoms + atom_aggregate / atom_degree)
        return atoms, edges


class CompactALiEGNN(nn.Module):
    """Small invariant atom/line-graph network with an energy head."""

    def __init__(self, geometry: FixedLineGraphGeometry, num_species: int,
                 hidden_dim: int = 32, layers: int = 2,
                 radial_bins: int = 12, angle_bins: int = 8):
        super().__init__()
        if num_species < 2 or hidden_dim < 4 or layers < 1:
            raise ValueError("invalid model dimensions")
        self.geometry = geometry
        self.num_species = int(num_species)
        self.species_embedding = nn.Embedding(num_species, hidden_dim)
        self.radial_embedding = _MLP(radial_bins, hidden_dim, hidden_dim)
        self.angle_embedding = _MLP(angle_bins, hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([_ALiEGNNBlock(hidden_dim) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.register_buffer("edge_sources", torch.as_tensor(geometry.edge_sources, dtype=torch.long))
        self.register_buffer("edge_targets", torch.as_tensor(geometry.edge_targets, dtype=torch.long))
        self.register_buffer("line_sources", torch.as_tensor(geometry.line_sources, dtype=torch.long))
        self.register_buffer("line_targets", torch.as_tensor(geometry.line_targets, dtype=torch.long))
        radial_centers = torch.linspace(0.0, 8.0, radial_bins)
        angle_centers = torch.linspace(-1.0, 1.0, angle_bins)
        distance = torch.as_tensor(geometry.distances)
        cosine = torch.as_tensor(geometry.angle_cosines)
        radial = torch.exp(-2.0 * (distance[:, None] - radial_centers[None, :]) ** 2)
        angular = torch.exp(-4.0 * (cosine[:, None] - angle_centers[None, :]) ** 2)
        self.register_buffer("radial_features", radial)
        self.register_buffer("angular_features", angular)
        edge_counts = np.bincount(geometry.line_targets, minlength=geometry.num_edges).clip(min=1)
        atom_counts = np.bincount(geometry.edge_targets, minlength=geometry.num_nodes).clip(min=1)
        self.register_buffer("edge_degree", torch.as_tensor(edge_counts, dtype=torch.float32)[None, :, None])
        self.register_buffer("atom_degree", torch.as_tensor(atom_counts, dtype=torch.float32)[None, :, None])

    def encode_occupancies(self, occupancies: Sequence[Sequence[str]], species: Sequence[str]):
        if len(species) != self.num_species or len(set(species)) != len(species):
            raise ValueError("species mapping does not match model")
        mapping = {symbol: index for index, symbol in enumerate(species)}
        encoded = []
        for occupancy in occupancies:
            if len(occupancy) != self.geometry.num_nodes:
                raise ValueError("occupancy length does not match graph")
            try:
                encoded.append([mapping[symbol] for symbol in occupancy])
            except KeyError as exc:
                raise ValueError(f"unknown occupancy species: {exc.args[0]}") from exc
        return torch.as_tensor(encoded, dtype=torch.long, device=self.edge_sources.device)

    def forward(self, encoded_occupancies):
        atoms = self.species_embedding(encoded_occupancies)
        batch = len(encoded_occupancies)
        edges = self.radial_embedding(self.radial_features).unsqueeze(0).expand(batch, -1, -1)
        angles = self.angle_embedding(self.angular_features).unsqueeze(0)
        for block in self.blocks:
            atoms, edges = block(
                atoms, edges, angles, self.edge_sources, self.edge_targets,
                self.line_sources, self.line_targets, self.edge_degree, self.atom_degree,
            )
        return self.head(atoms.mean(dim=1)).squeeze(-1)


class OnlineALiEGNNEnsemble:
    """From-scratch current-run ensemble; no pretrained thermal target is used."""

    def __init__(self, geometry: FixedLineGraphGeometry, species: Sequence[str],
                 ensemble_size: int = 4, hidden_dim: int = 32, layers: int = 2,
                 epochs: int = 80, learning_rate: float = 5e-3,
                 weight_decay: float = 1e-4, seed: int = 0, device: str = "cpu"):
        if ensemble_size < 2 or epochs < 1:
            raise ValueError("ensemble_size must be >=2 and epochs positive")
        self.geometry = geometry
        self.species = tuple(species)
        self.ensemble_size = int(ensemble_size)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.models: list[CompactALiEGNN] = []
        self.energy_offset = 0.0
        self.energy_scale = 1.0
        self.residual_floor = 1e-6
        self.training_data_sha256: str | None = None
        self.ensemble_state_sha256: str | None = None

    def fit(self, occupancies: Sequence[Sequence[str]], energies: np.ndarray):
        values64 = np.asarray(energies, dtype=np.float64)
        if values64.ndim != 1 or len(values64) != len(occupancies) or len(values64) < 2:
            raise ValueError("occupancies and one-dimensional energies must align")
        training_digest = hashlib.sha256()
        for occupancy, value in zip(occupancies, values64):
            encoded = '\x1f'.join(occupancy).encode()
            training_digest.update(len(encoded).to_bytes(8, 'little'))
            training_digest.update(encoded)
            training_digest.update(np.asarray([value], dtype='<f8').tobytes())
        self.training_data_sha256 = training_digest.hexdigest()
        values = values64.astype(np.float32)
        self.energy_offset = float(values.mean())
        self.energy_scale = max(float(values.std()), 1e-3)
        target = torch.as_tensor(
            (values - self.energy_offset) / self.energy_scale,
            dtype=torch.float32, device=self.device,
        )
        self.models = []
        member_predictions = []
        for member in range(self.ensemble_size):
            member_seed = self.seed + 104729 * member
            torch.manual_seed(member_seed)
            model = CompactALiEGNN(
                self.geometry, len(self.species), self.hidden_dim, self.layers
            ).to(self.device)
            encoded = model.encode_occupancies(occupancies, self.species)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            generator = torch.Generator(device="cpu").manual_seed(member_seed + 1)
            bootstrap = torch.randint(
                0, len(values), (len(values),), generator=generator, device="cpu"
            ).to(self.device)
            model.train()
            for _ in range(self.epochs):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(encoded[bootstrap])
                loss = torch.mean((prediction - target[bootstrap]) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                member_predictions.append(model(encoded).detach().cpu().numpy())
            self.models.append(model)
        normalized_mean = np.mean(member_predictions, axis=0)
        residual = values - (self.energy_offset + self.energy_scale * normalized_mean)
        self.residual_floor = max(float(np.sqrt(np.mean(residual ** 2))), 1e-6)
        state_digest = hashlib.sha256()
        state_digest.update(
            f'{self.ensemble_size}|{self.hidden_dim}|{self.layers}|{self.epochs}|'
            f'{self.learning_rate:.17g}|{self.weight_decay:.17g}|{self.seed}'.encode()
        )
        for member, model in enumerate(self.models):
            state_digest.update(np.asarray([member], dtype='<i8').tobytes())
            for name, tensor in sorted(model.state_dict().items()):
                value = tensor.detach().cpu().contiguous()
                state_digest.update(name.encode())
                state_digest.update(str(value.dtype).encode())
                state_digest.update(np.asarray(value.shape, dtype='<i8').tobytes())
                state_digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        self.ensemble_state_sha256 = state_digest.hexdigest()
        return self

    def predict(self, occupancies: Sequence[Sequence[str]], return_std: bool = False):
        if not self.models:
            raise RuntimeError("online ALIEGNN ensemble is not fitted")
        predictions = []
        for model in self.models:
            encoded = model.encode_occupancies(occupancies, self.species)
            with torch.no_grad():
                predictions.append(model(encoded).detach().cpu().numpy())
        values = self.energy_offset + self.energy_scale * np.asarray(predictions)
        mean = values.mean(axis=0)
        if return_std:
            std = np.sqrt(values.var(axis=0) + self.residual_floor ** 2)
            return mean.astype(float), std.astype(float)
        return mean.astype(float)


def _matched_features(candidates, history_occ, feature_fn):
    candidate_features = np.asarray([feature_fn(item.occupancy) for item in candidates], float)
    history_features = np.asarray([feature_fn(item) for item in history_occ], float)
    combined = np.vstack((history_features, candidate_features))
    scale = np.maximum(combined.std(axis=0), 0.25)
    return candidate_features / scale, history_features / scale


def _acquisition_scores(mode, mean, std, novelty, energy_scale):
    normalized_novelty = novelty / max(float(novelty.max()), 1e-12)
    if mode == "greedy":
        return np.asarray(mean, float)
    if mode == "uncertainty":
        return -np.asarray(std, float)
    if mode == "lcb":
        return (np.asarray(mean, float) - 1.5 * np.asarray(std, float)
                - 0.08 * energy_scale * normalized_novelty)
    raise ValueError(f"unknown acquisition mode: {mode}")


def select_candidate_aliegnn(
    candidates: Sequence[Candidate], history_occ: Sequence[Sequence[str]],
    history_energy: Sequence[float], geometry: FixedLineGraphGeometry,
    species: Sequence[str], feature_fn: Callable[[Sequence[str]], np.ndarray],
    acquisition: str, seed: int = 0, ensemble_size: int = 3,
    hidden_dim: int = 16, layers: int = 1, epochs: int = 20,
) -> ALiEGNNSelection:
    """Train on current-run energies and rank the supplied unevaluated candidates."""
    if not candidates:
        raise ValueError("candidate list is empty")
    if len(candidates) > MAX_CANDIDATE_BATCH:
        raise ValueError(f"candidate batch exceeds limit {MAX_CANDIDATE_BATCH}")
    if len(history_occ) > MAX_TRAINING_HISTORY:
        raise ValueError(f"training history exceeds limit {MAX_TRAINING_HISTORY}")
    if len(history_occ) != len(history_energy):
        raise ValueError("training history occupancies and energies must align")
    candidate_features, history_features = _matched_features(candidates, history_occ, feature_fn)
    novelty = np.asarray([
        min(np.linalg.norm(candidate - previous) for previous in history_features)
        for candidate in candidate_features
    ])
    if len(history_energy) < 8 or acquisition == "novelty":
        index = int(np.argmax(novelty))
        return ALiEGNNSelection(index, None, None, float(novelty[index]), "descriptor novelty")
    model = OnlineALiEGNNEnsemble(
        geometry, species, ensemble_size=ensemble_size, hidden_dim=hidden_dim,
        layers=layers, epochs=epochs, seed=seed,
    ).fit(history_occ, np.asarray(history_energy, dtype=float))
    candidate_occ = [item.occupancy for item in candidates]
    mean, std = model.predict(candidate_occ, return_std=True)
    energy_scale = max(float(np.std(history_energy)), 1e-3)
    score = _acquisition_scores(acquisition, mean, std, novelty, energy_scale)
    index = int(np.argmin(score))
    return ALiEGNNSelection(
        index, float(mean[index]), float(std[index]), float(novelty[index]),
        f"online compact ALIEGNN-inspired energy ensemble {acquisition}",
        training_data_sha256=model.training_data_sha256,
        ensemble_state_sha256=model.ensemble_state_sha256,
    )
