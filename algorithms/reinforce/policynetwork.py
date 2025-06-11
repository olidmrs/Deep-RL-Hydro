import torch.nn as nn
import torch

class PolicyNetwork(nn.Module):
    def __init__(
            self,
            input_dim : int,
            output_dim : int,
            nb_hidden : int,
            hidden_size : int
            ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nb_hidden = nb_hidden
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        self.create_layers()

    def create_layers(self) -> None:
        """
        Creates hidden layers with dimensions specified in parameters of class
        """
        '''
        input,hidden
        hidden,hidden
        hidden,output
        '''
        last_dim = self.input_dim
        for layer in range(self.nb_hidden + 1):
            if layer == self.nb_hidden:
                self.layers.append(nn.Linear(last_dim, self.output_dim))        
            else:
                self.layers.append(nn.Linear(last_dim, self.hidden_size))
            last_dim = self.hidden_size


    def forward(self, state : tuple[int, int, int]) -> torch.Tensor:
        """
        Forward propagation through the neural network

        Args:
            state (tuple[int,int,int]): input which is the state/observation space (Lt, It, t)

        Returns:
            torch.Tensor: q_values associated to all possible actions
        """        
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i != len(self.layers) - 1:
                state = torch.relu(layer(state))
            else:
                state = layer(state)
        return state
    
    def act(self, logits : torch.Tensor, min_valid_action_space, max_valid_action_space):
        mask = torch.zeros_like(logits)
        mask[min_valid_action_space : max_valid_action_space] = 1
        logits[mask == 0] = -1e9
            
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs = probs)
        action = dist.sample().item()
        return action
        
        