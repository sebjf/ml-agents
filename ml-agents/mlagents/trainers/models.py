import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("mlagents.trainers")


class VisualEncoder(nn.Module):
    def __init__(self, camera_params, h_size, num_layers):
        super().__init__()
        o_size_h = camera_params["height"]
        o_size_w = camera_params["width"]
        bw = camera_params["blackAndWhite"]
        c_channels = 1 if bw else 3

        self.model = nn.Sequential(
            nn.Conv2d(c_channels, 16, kernel_size=(8, 8), stride=(4, 4)),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2)),
            nn.ELU()
        )
        conv_h = int((o_size_h - (8 - 1)) / 4 + 1)
        conv_h = int((conv_h - (4 - 1)) / 2 + 1)
        conv_w = int((o_size_w - (8 - 1)) / 4 + 1)
        conv_w = int((conv_w - (4 - 1)) / 2 + 1)
        self.vector_encoder = VectorEncoder(32*conv_h*conv_w, h_size, num_layers)

    def forward(self, image_input):
        # transpose (tf: [batch, h, w, c], torch: [batch, c, h, w])
        image_input = image_input.transpose(2, 3).transpose(1, 2)
        hidden = self.model(image_input)
        hidden = hidden.flatten(start_dim=1)
        return self.vector_encoder(hidden)

class VectorEncoder(nn.Module):
    def __init__(self, input_size, h_size, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_size, h_size))
            layers.append(self.swish())
            input_size = h_size
        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        self.model.apply(self._init_linear)

    def _init_linear(self, x):
        if type(x) == nn.Linear:
            # slightly different from tf implementation
            nn.init.kaiming_normal_(x.weight, a=0.2)

    def forward(self, vector_input):
        return self.model(vector_input)

    class swish(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x * torch.sigmoid(x)


class ObservationEncoder(nn.Module):
    def __init__(self, h_size, num_layers, brain):
        super().__init__()
        self.num_vis_obs = brain.number_visual_observations
        self.vector_obs_size = brain.vector_observation_space_size * \
            brain.num_stacked_vector_observations

        self.visual_encoders = nn.ModuleList()

        for i in range(self.num_vis_obs):
            self.visual_encoders.append(VisualEncoder(
                brain.camera_resolutions[i], h_size, num_layers))
        if self.vector_obs_size > 0:
            self.vector_encoder = VectorEncoder(self.vector_obs_size, h_size, num_layers)

        self.obs_size = self.num_vis_obs * h_size
        if self.vector_obs_size > 0:
            self.obs_size += h_size

    def forward(self, visual_in, vector_in):
        hidden_state, hidden_visual = None, None
        if self.num_vis_obs > 0:
            encoded_visuals = []
            for vis_encoder, vis_in in zip(self.visual_encoders, visual_in):
                encoded_visuals.append(vis_encoder(vis_in))
            hidden_visual = torch.cat(encoded_visuals, 1)
        if self.vector_obs_size > 0:
            hidden_state = self.vector_encoder(vector_in)
        if hidden_state is not None and hidden_visual is not None:
            final_hidden = torch.cat([hidden_visual, hidden_state], 1)
        elif hidden_state is None and hidden_visual is not None:
            final_hidden = hidden_visual
        elif hidden_state is not None and hidden_visual is None:
            final_hidden = hidden_state
        else:
            raise Exception(
                "No valid network configuration possible. "
                "There are no states or observations in this brain"
            )
        return final_hidden


class CCActorCritic(nn.Module):
    def __init__(self, h_size, num_layers, stream_names, brain):
        super().__init__()
        self.act_size = brain.vector_action_space_size
        self.num_vis_obs = brain.number_visual_observations
        self.policy_observation_encoder = ObservationEncoder(h_size, num_layers, brain)
        self.value_observation_encoder = ObservationEncoder(h_size, num_layers, brain)

        self.policy_layer = torch.nn.Linear(self.policy_observation_encoder.obs_size, self.act_size[0])
        self.stream_names = stream_names
        self.value_layers = nn.ModuleDict()
        for name in self.stream_names:
            self.value_layers[name] = torch.nn.Linear(self.value_observation_encoder.obs_size, 1)

        self.log_sigma_sq = nn.Parameter(torch.zeros(self.act_size[0]))
        self.const1 = np.log(2.0 * np.pi)
        self.const2 = np.log(2 * np.pi * np.e)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.policy_layer.weight, a=10)

    def get_value_estimate(self, visual_in, vector_in):
        value_heads = {}
        hidden_value = self.value_observation_encoder(visual_in, vector_in)
        for name in self.stream_names:
            value = self.value_layers[name](hidden_value)
            value_heads[name] = value.data.numpy()
        return value_heads

    def forward(self, input_dict):
        visual_in = [input_dict.get("visual_obs%d" % i, None) for i in range(self.num_vis_obs)]
        vector_in = input_dict.get("vector_obs", None)
        output_pre = input_dict.get("actions_pre", None)
        epsilon = input_dict["random_normal_epsilon"]

        hidden_policy = self.policy_observation_encoder(visual_in, vector_in)
        hidden_value = self.value_observation_encoder(visual_in, vector_in)

        mu = self.policy_layer(hidden_policy)
        sigma_sq = torch.exp(self.log_sigma_sq)

        # Clip and scale output to ensure actions are always within [-1, 1] range.
        if output_pre is None:
            output_pre = (mu + torch.sqrt(sigma_sq) * epsilon).detach()
        output = torch.clamp(output_pre, -3, 3) / 3

        # Compute probability of model output.
        all_log_probs = (
            - 0.5 * torch.pow(output_pre - mu, 2) / sigma_sq
            - 0.5 * self.const1
            - 0.5 * self.log_sigma_sq
        )

        entropy = 0.5 * (self.const2 + self.log_sigma_sq).mean()

        value_heads = {}
        for name in self.stream_names:
            value = self.value_layers[name](hidden_value)
            value_heads[name] = value
        value = torch.mean(torch.stack(list(value_heads.values())))
        entropy = torch.ones(value.flatten().size()) * entropy

        return output, output_pre, all_log_probs, value_heads, value, entropy

    def get_probs(self, all_log_probs, all_old_log_probs, actions=None, action_masks=None):
        log_probs = torch.sum(log_probs, 1, keepdim=True)
        old_log_probs = torch.sum(old_log_probs, 1, keepdim=True)
        return log_probs, old_log_probs


class DCActorCritic(nn.Module):
    def __init__(self, h_size, num_layers, stream_names, brain):
        super().__init__()
        self.act_size = brain.vector_action_space_size
        self.num_vis_obs = brain.number_visual_observations
        self.observation_encoder = ObservationEncoder(h_size, num_layers, brain)
        self.policy_layers = nn.ModuleList()
        for size in self.act_size:
            self.policy_layers.append(torch.nn.Linear(self.observation_encoder.obs_size, size, bias=False))
        self.num_branch = len(self.act_size)
        self.stream_names = stream_names
        self.value_layers = {}
        for name in self.stream_names:
            self.value_layers[name] = torch.nn.Linear(self.observation_encoder.obs_size, 1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self._init_weights()

    def _init_weights(self):
        for layer in self.policy_layers:
            nn.init.kaiming_normal_(layer.weight, a=10)

    def get_value_estimate(self, visual_in, vector_in):
        value_heads = {}
        hidden = self.observation_encoder(visual_in, vector_in)
        for name in self.stream_names:
            value = self.value_layers[name](hidden)
            value_heads[name] = value.data.numpy()
        return value_heads

    def forward(self, input_dict):
        visual_in = [input_dict.get("visual_obs%d" % i, None) for i in range(self.num_vis_obs)]
        vector_in = input_dict.get("vector_obs", None)
        action_mask = input_dict["action_mask"]

        hidden = self.observation_encoder(visual_in, vector_in)
        action_idx = [0] + list(np.cumsum(self.act_size))

        all_log_probs = [layer(hidden) for layer in self.policy_layers]

        branch_masks = [
            action_mask[:, action_idx[i] : action_idx[i + 1]] for i in range(self.num_branch)
        ]
        raw_probs = [
            (F.softmax(all_log_probs[k], 1) + 1.0e-10) * branch_masks[k] for k in range(self.num_branch)
        ]
        output = torch.cat(
            [torch.multinomial(raw_probs[k], 1) for k in range(self.num_branch)], 1
        )

        value_heads = {}
        for name in self.stream_names:
            value = self.value_layers[name](hidden)
            value_heads[name] = value
        value = torch.mean(torch.stack(list(value_heads.values())))

        entropy = [torch.sum(-F.softmax(x, 1) * F.log_softmax(x, 1), 1) for x in all_log_probs]
        entropy = torch.stack(entropy, 1).sum(1)

        return output, None, torch.cat(all_log_probs, 1), value_heads, value, entropy

    def get_probs(self, all_log_probs, all_old_log_probs, actions, action_masks):
        action_idx = [0] + list(np.cumsum(self.act_size))
        branch_masks = [
            action_masks[:, action_idx[i] : action_idx[i + 1]] for i in range(self.num_branch)
        ]

        all_log_probs = [
            all_log_probs[:, action_idx[i] : action_idx[i + 1]]
            for i in range(self.num_branch)
        ]
        raw_probs = [
            (F.softmax(all_log_probs[k], 1) + 1.0e-10) * branch_masks[k] for k in range(self.num_branch)
        ]
        normalized_probs = [raw_probs[k]/torch.sum(raw_probs[k], 1, keepdim=True) for k in range(self.num_branch)]
        normalized_logits = [torch.log(normalized_probs[k] + 1.0e-10) for k in range(self.num_branch)]

        all_old_log_probs = [
            all_old_log_probs[:, action_idx[i] : action_idx[i + 1]]
            for i in range(self.num_branch)
        ]
        old_raw_probs = [
            (F.softmax(all_old_log_probs[k], 1) + 1.0e-10) * branch_masks[k] for k in range(self.num_branch)
        ]
        old_normalized_probs = [old_raw_probs[k]/torch.sum(old_raw_probs[k], 1, keepdim=True) for k in range(self.num_branch)]
        old_normalized_logits = [torch.log(old_normalized_probs[k] + 1.0e-10) for k in range(self.num_branch)]

        actions = [actions[:, i] for i in range(self.num_branch)]
        log_probs = torch.sum(
            torch.stack([-self.criterion(normalized_logits[k], actions[k])
                for k in range(self.num_branch)], dim=1), 1, keepdim=True
        )
        old_log_probs = torch.sum(
            torch.stack([-self.criterion(old_normalized_logits[k], actions[k])
                for k in range(self.num_branch)], dim=1), 1, keepdim=True
        )
        return log_probs, old_log_probs
