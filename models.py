import torch


class SpikeEmbeddingModel(torch.nn.Module):
    def __init__(
        self,
        max_groups: int,
        hidden_size: int = 256,
        num_conv_layers: int = 3,
        num_fc_layers: int = 2,
        kernel_size: int = 5,
        stride: int = 3,
    ):
        super().__init__()

        # The +1 is for the padding group
        self.group_embeddings = torch.nn.Embedding(max_groups + 1, hidden_size)

        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    hidden_size, hidden_size, kernel_size=kernel_size, stride=stride
                )
                for _ in range(num_conv_layers)
            ]
        )
        self.fcs = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_fc_layers)]
        )

        self.output_head = torch.nn.Linear(hidden_size, 2)

    def forward(
        self,
        groups: torch.Tensor,
        length: torch.Tensor,
        gathered_spikes: torch.Tensor,
        is_not_padding: torch.Tensor,
    ) -> torch.Tensor:
        sequence = self.group_embeddings(groups)

        sequence = sequence.permute(0, 2, 1)

        for conv in self.convs:
            sequence = conv(sequence)
            sequence = torch.nn.functional.relu(sequence)

        sequence = sequence.mean(dim=2)

        for fc in self.fcs:
            sequence = fc(sequence)
            sequence = torch.nn.functional.relu(sequence)

        return self.output_head(sequence)


class WaveformModel(torch.nn.Module):
    def __init__(
        self,
        max_groups: int,
        hidden_size: int = 256,
        num_conv_layers: int = 3,
        num_fc_layers: int = 2,
        kernel_size: int = 5,
        stride: int = 3,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    hidden_size, hidden_size, kernel_size=kernel_size, stride=stride
                )
                if i > 0
                else torch.nn.Conv1d(
                    max_groups, hidden_size, kernel_size=kernel_size, stride=stride
                )
                for i in range(num_conv_layers)
            ]
        )
        self.max_groups = max_groups
        self.fcs = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_fc_layers)]
        )

        self.output_head = torch.nn.Linear(hidden_size, 2)

    def forward(
        self,
        groups: torch.Tensor,
        length: torch.Tensor,
        gathered_spikes: torch.Tensor,
        is_not_padding: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = groups.shape[0]
        n_channels = gathered_spikes.shape[2]
        n_timepoints = gathered_spikes.shape[3]
        sequence = gathered_spikes.reshape(-1, n_channels, n_timepoints)

        for conv in self.convs:
            sequence = conv(sequence)
            sequence = torch.nn.functional.relu(sequence)

        sequence = sequence.reshape(batch_size, -1, sequence.shape[1])

        sequence = sequence * is_not_padding.unsqueeze(-1).float()

        sequence = sequence.sum(dim=1)

        sequence = sequence / is_not_padding.float().sum(dim=1).unsqueeze(-1)

        for fc in self.fcs:
            sequence = fc(sequence)
            sequence = torch.nn.functional.relu(sequence)

        return self.output_head(sequence)
