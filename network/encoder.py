from torch import nn




class Encoder(nn.Module):
    def __init__(self, image_dim, latent_space_dim, intermediate_dim, num_hidden_layers = 1):
        super(Encoder, self).__init__()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Sequential(nn.Linear(image_dim**2, intermediate_dim), nn.ReLU())
        self.output_layer = nn.Linear(intermediate_dim, latent_space_dim)
        list_intermediate = [[nn.Linear(intermediate_dim, intermediate_dim), nn.ReLU()]
                             for _ in range(num_hidden_layers)]
        self.linear_relu_stack = nn.Sequential(*[layer for intermediate in list_intermediate for layer in intermediate])


    def forward(self, x):
        x = self.flatten(x)
        x = self.input_layer(x)
        hidden = self.linear_relu_stack(x)
        logits = self.output_layer(hidden)
        return logits


