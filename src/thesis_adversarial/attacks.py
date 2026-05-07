import torch.nn as nn 
import torch
import torch.nn.functional as F
from torch import Tensor

class PGDAttack():
    """
    Implementation of the PGD attack with l_inf norm described in Towards Deep Learning Models Resistant to Adversarial attacks by Madry et. al..
    
    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to attack.

    epsilon : float
        The maximum allowed perturbation magnitude.
    
    steps : int
        The number of attack iterations.

    alpha : float
        The step size used in each PGD iteration.
    """
    
    def __init__(self, model:nn.Module, epsilon:float=0.0, steps:int=0, alpha:float=0.0) -> None:
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha

    def perturb(self, x: Tensor, y:Tensor) -> Tensor:
        was_training = self.model.training
        self.model.eval()
        try:
            x_orig = x.detach()
            x_adv = x_orig + torch.empty_like(x_orig).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
        
            for i in range(self.steps):
                x_adv.requires_grad_(True)
                outputs = self.model(x_adv)
                loss = F.cross_entropy(outputs,y)
                grad = torch.autograd.grad(loss, x_adv)[0]
            
                with torch.no_grad():
                    x_adv = x_adv + self.alpha * grad.sign()
                
                    delta = torch.clamp(x_adv - x_orig, -self.epsilon, self.epsilon)
                    x_adv = torch.clamp(x_orig + delta, 0, 1)
                
                x_adv = x_adv.detach()

            return x_adv
        finally:
            self.model.train(was_training)
    
class FGSMAttack():
    def __init__(self, model:nn.Module, epsilon:float=0.0):
        self.model = model
        self.epsilon = epsilon
        
    def perturb(self, x:Tensor, y:Tensor) -> Tensor:
        was_training = self.model.training
        self.model.eval()
        
        try:
            x_orig = x.detach().clone()
            x_orig.requires_grad_()
        
            outputs = self.model(x_orig)
            loss = F.cross_entropy(outputs, y)
            grad = torch.autograd.grad(loss, x_orig)[0]
        
            with torch.no_grad():
                x_adv = x_orig + self.epsilon * grad.sign()
                x_adv = torch.clamp(x_adv, 0, 1)
            
            x_adv = x_adv.detach()
        
            return x_adv
        finally:
            self.model.train(was_training)