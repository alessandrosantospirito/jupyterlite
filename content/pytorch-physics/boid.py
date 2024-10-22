# Boids algorithm partially based on https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html#Update-position
# Code: https://github.com/StevenGuo42/torch-boid
from dataclasses import dataclass
from numbers import Number

import torch

@dataclass
class Boid:
    init_speed: float = None
    min_speed: float = 3
    max_speed: float = 6
    max_acc: float = 0.5
    
    view_radius: float = 30
    view_angle: float = None        # human: 220 deg, pigeon: 340 deg, owl: 110 deg
    
    avoid_radius: float = 15        
    avoid_view: bool = True         # only avoid boids in view angle
    
    sep_factor: float = 0.05        # avoidfactor
    align_factor: float = 0.1      # matchingfactor
    cohe_factor: float = 0.0005       # centeringfactor
    bias_factor: float = 0.005       
    edge_factor: float = 0.05        # turnfactor
    
    is_debug: bool = False

# @dataclass
# class Boid:
#     init_speed: float = None
#     min_speed: float = 3
#     max_speed: float = 6
#     max_acc: float = 0.5
    
#     view_radius: float = 40
#     view_angle: float = None        # human: 220 deg, pigeon: 340 deg, owl: 110 deg
    
#     avoid_radius: float = 8         
#     avoid_view: bool = True         # only avoid boids in view angle
    
#     sep_factor: float = 0.05        # avoidfactor
#     align_factor: float = 0.05      # matchingfactor
#     cohe_factor: float = 0.0005     # centeringfactor
#     bias_factor: float = 0.005       
#     edge_factor: float = 0.05        # turnfactor
    
#     is_debug: bool = False

class Flock(Boid):
    def __init__(self, D: int = 2, N: int = 1000, box_bottom=0, box_top=500,
                 margin_bottom=100, margin_top=100, pass_through_edges=True,
                 bouncy_edges=False, device=torch.device("cpu"), **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.D = D
        self.N = N
        self.box_bottom = self.parse_to_tensor(box_bottom).to(self.device)
        self.box_top = self.parse_to_tensor(box_top).to(self.device)
        self.margin_bottom = self.parse_to_tensor(margin_bottom).to(self.device)
        self.margin_top = self.parse_to_tensor(margin_top).to(self.device)
        self.bound_bottom = self.box_bottom + self.margin_bottom
        self.bound_top = self.box_top - self.margin_top
        self.box_upper_mat = self.box_top.unsqueeze(0).expand(N, -1).to(self.device)
        self.box_lower_mat = self.box_bottom.unsqueeze(0).expand(N, -1).to(self.device)
        self.pass_through_edges = pass_through_edges
        self.bouncy_edges = bouncy_edges
        self.pos = torch.rand((N, D), device=self.device) * (self.bound_top - self.bound_bottom) + self.bound_bottom
        if self.init_speed is None:
            self.vel = torch.randn((N, D), device=self.device) * (self.max_speed - self.min_speed) + self.min_speed
        else:
            self.vel = torch.ones((N, D), device=self.device) * self.init_speed
        if self.is_debug:
            print('pos: \n', self.pos)
            print('vel: \n', self.vel)

    def parse_to_tensor(self, x):
        if isinstance(x, Number):
            return torch.tensor([x] * self.D, device=self.device)
        elif len(x) != self.D:
            raise ValueError('input must be a number or a list of length D')
        else:
            return torch.tensor(x, device=self.device)

    def update(self):
        pos_mat = self.pos.unsqueeze(1).expand(-1, self.N, -1)
        vel_mat = self.vel.unsqueeze(1).expand(-1, self.N, -1)
        diff = pos_mat.transpose(0, 1) - pos_mat
        sq_dist_mat = diff.pow(2).sum(dim=-1)
        view_mat = sq_dist_mat < self.view_radius ** 2
        view_mat.fill_diagonal_(0)
        if self.view_angle is not None:
            view_angle_mat = torch.cosine_similarity(diff, vel_mat, dim=-1) > torch.cos(self.view_angle / 2)
            view_mat *= view_angle_mat
        view_mat = view_mat.unsqueeze(-1).expand(-1, -1, self.D)
        avoid_mat = sq_dist_mat < self.avoid_radius ** 2
        avoid_mat.fill_diagonal_(0)
        if self.view_angle is not None and self.avoid_view:
            avoid_mat *= view_angle_mat
        avoid_mat = avoid_mat.unsqueeze(-1).expand(-1, -1, self.D)
        avoid_mat_sum = avoid_mat.sum(dim=1)
        avoid_mask = avoid_mat_sum != 0
        avoid_vel = torch.zeros((self.N, self.D), device=self.device)
        avoid_vel[avoid_mask] = ((avoid_mat * diff).sum(dim=0))[avoid_mask] / avoid_mat_sum[avoid_mask]
        bottom_edge = torch.le(self.pos, self.bound_bottom)
        top_edge = torch.ge(self.pos, self.bound_top)
        view_mat = view_mat & ~avoid_mat
        view_mat_sum = view_mat.sum(dim=1)
        view_mask = view_mat_sum != 0
        avg_pos = torch.zeros((self.N, self.D), device=self.device)
        avg_pos[view_mask] = ((pos_mat * view_mat).sum(dim=0))[view_mask] / view_mat_sum[view_mask]
        avg_vel = torch.zeros((self.N, self.D), device=self.device)
        avg_vel[view_mask] = ((vel_mat * view_mat).sum(dim=0))[view_mask] / view_mat_sum[view_mask]
        cohe_vel = (avg_pos - self.pos) * view_mask * self.cohe_factor
        align_vel = (avg_vel - self.vel) * view_mask * self.align_factor
        sep_vel = avoid_vel * avoid_mask * self.sep_factor
        bias_vel = ((self.box_bottom + self.box_top) / 2 - self.pos) * self.bias_factor
        edge_vel = bottom_edge * self.edge_factor - top_edge * self.edge_factor
        sum_d_vel = cohe_vel + align_vel + sep_vel + bias_vel + edge_vel
        acc = sum_d_vel
        acc_mag = acc.pow(2).sum(dim=-1).sqrt()
        acc_mag = torch.clamp(acc_mag, max=self.max_acc)
        acc = torch.nn.functional.normalize(acc, p=2, dim=-1) * acc_mag.unsqueeze(-1)
        sum_vel = self.vel + acc
        vel_mag = sum_vel.pow(2).sum(dim=-1).sqrt()
        vel_clipped = torch.clamp(vel_mag, min=self.min_speed, max=self.max_speed)
        sum_vel = torch.nn.functional.normalize(sum_vel, p=2, dim=-1) * vel_clipped.unsqueeze(-1)
        self.vel = sum_vel
        self.pos = self.pos + self.vel
        if self.pass_through_edges:
            self.pos = torch.where(self.pos < self.box_lower_mat, self.box_upper_mat, self.pos)
            self.pos = torch.where(self.pos > self.box_upper_mat, self.box_lower_mat, self.pos)
        else:
            if self.bouncy_edges:
                self.vel = torch.where(self.pos < self.box_lower_mat, -self.vel, self.vel)
                self.vel = torch.where(self.pos > self.box_upper_mat, -self.vel, self.vel)
            self.pos = torch.maximum(self.pos, self.box_lower_mat)
            self.pos = torch.minimum(self.pos, self.box_upper_mat)
        if self.is_debug:
            print('pos: \n', self.pos)
            print('vel: \n', self.vel)
        if any(torch.isnan(self.pos).flatten()):
            raise ValueError('position contains NaN')
        if any(torch.isnan(self.vel).flatten()):
            raise ValueError('velocity contains NaN')
