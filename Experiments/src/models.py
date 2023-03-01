import numpy as np
import torch


class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, layers_dim, latent_dim, activation=None, out_activation=None, device=torch.device("cpu")):
        super(Autoencoder, self).__init__()
        self.device = device
        self.activation = getattr(torch.nn, activation) if activation else getattr(torch.nn, "Identity")
        self.out_activation = getattr(torch.nn, out_activation) if out_activation else getattr(torch.nn, "Identity")
        
        self.input_dim = list(input_dim)
        if len(input_dim) > 1:
            input_dim = np.prod(input_dim, keepdims=True, dtype=int)
        input_dim = list(input_dim)
        
        self.encoder = torch.nn.Sequential()
        self.encoder.add_module(
            name="enc_flatten",
            module=torch.nn.Flatten()
        )
        
        for i, dim in enumerate(zip(input_dim + layers_dim, layers_dim), 1):
            self.encoder.add_module(
                name=f"enc_linear_{i}",
                module=torch.nn.Linear(*dim)
            )
            self.encoder.add_module(
                name=f"enc_activation_{i}",
                module=self.activation()
            )
            
        self.encoder.add_module(
            name="hidden_linear",
            module=torch.nn.Linear((input_dim + layers_dim)[-1], latent_dim)
        )
        self.encoder.add_module(
            name="hidden_activation",
            module=self.activation()
        )
        
        layers_dim = layers_dim[::-1]
        self.decoder = torch.nn.Sequential()
        for i, dim in enumerate(zip(list([latent_dim]) + layers_dim, layers_dim), 1):
            self.decoder.add_module(
                name=f"dec_linear_{i}",
                module=torch.nn.Linear(*dim)
            )
            self.decoder.add_module(
                name=f"dec_activation_{i}",
                module=self.activation()
            )
        
        self.decoder.add_module(
            name="out_linear",
            module=torch.nn.Linear((list([latent_dim]) + layers_dim)[-1], input_dim[0])
        )
        self.decoder.add_module(
            name="out_activation",
            module=self.out_activation()
        )
        
        self.apply(self.__init_weights(activation))

    def __init_weights(self, activation):
        activation = activation if activation and activation != "Identity" else "linear"
        activation = activation.lower()
        def func(module):
            if type(module) == torch.nn.Linear:
                torch.nn.init.zeros_(module.bias)
                gain = torch.nn.init.calculate_gain(activation)
                if activation != "relu":
                    torch.nn.init.xavier_uniform_(module.weight, gain)
                else:
                    torch.nn.init.kaiming_uniform_(module.weight, gain)
        return func
        
    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
        h = self.encoder(x)
        h = self.decoder(h).reshape(-1, *self.input_dim)
        return h
    
    def fit(self, train_loader, fit_params):
        self.criterion = fit_params["loss"]().to(self.device)
        optimizer = fit_params["optimizer"](self.parameters())

        for i in range(fit_params["epochs"]):
            for x, _ in train_loader:
                optimizer.zero_grad()
                x = x.to(self.device)
                x_out = self.forward(x)
                loss = self.criterion(x_out, x.squeeze())
                loss.backward()
                optimizer.step()
    
    def evaluate(self, data_loader):
        self.criterion = self.criterion.to(self.device)
        loss = torch.zeros(0, device=self.device)
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(self.device)
                x_out = self.forward(x)
                loss = torch.cat([
                    loss, 
                    self.criterion(x_out, x.squeeze()).view(1)
                ])
        if self.device.type == "cuda":
            loss = loss.cpu()
        return loss.mean().numpy().reshape(1)[0]
    
    
class TiedAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, layers_dim, latent_dim,
                 activation=None, out_activation=None, 
                 alpha=0.01, device=torch.device("cpu")):
        super(TiedAutoencoder, self).__init__()
        self.alpha = alpha
        self.device = device
        self.activation = getattr(torch.nn, activation) if activation else getattr(torch.nn, "Identity")
        self.out_activation = getattr(torch.nn, out_activation) if out_activation else getattr(torch.nn, "Identity")
        
        self.input_dim = list(input_dim)
        if len(input_dim) > 1:
            input_dim = np.prod(input_dim, keepdims=True, dtype=int)
        input_dim = list(input_dim)
        
        self.encoder = torch.nn.Sequential()
        self.encoder.add_module(
            name="enc_flatten",
            module=torch.nn.Flatten()
        )
        
        self.decoder_biases = torch.nn.ParameterList()
        for i, dim in enumerate(zip(input_dim + layers_dim, layers_dim), 1):
            self.encoder.add_module(
                name=f"enc_linear_{i}",
                module=torch.nn.Linear(*dim)
            )
            self.encoder.add_module(
                name=f"enc_activation_{i}",
                module=self.activation()
            )
            self.decoder_biases.append(
                torch.nn.Parameter(torch.zeros(dim[0]))
            )
            
        self.encoder.add_module(
            name="hidden_linear",
            module=torch.nn.Linear((input_dim + layers_dim)[-1], latent_dim)
        )
        self.encoder.add_module(
            name="hidden_activation",
            module=self.activation()
        )
        self.decoder_biases.append(
            torch.nn.Parameter(torch.zeros((input_dim + layers_dim)[-1]))
        )
        self.decoder_biases = self.decoder_biases[::-1]
        self.apply(self.__init_weights(activation))

    def __init_weights(self, activation):
        def func(module):
            if type(module) == torch.nn.Linear:
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.orthogonal_(module.weight)
        return func
            
    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
        h = self.encoder(x)
        h = self.decoder(h).reshape(-1, *self.input_dim)
        return h
    
    def decoder(self, h):
        reversed_encoder_children = reversed(
            list(self.encoder.named_children())
        )
        reversed_encoder_children = filter(
            lambda x: "linear" in x[0],
            reversed_encoder_children
        )
        for i, params in enumerate(zip(reversed_encoder_children, self.decoder_biases), start=1):
            child, bias = params
            child_name, child_module = child
            
            h = torch.nn.functional.linear(
                input=h,
                weight=child_module.weight.T,
                bias=bias
            )
            activation = self.activation() if i < len(self.decoder_biases) else self.out_activation()
            h = activation(h)
        return h
    
    def fit(self, train_loader, fit_params):
        self.criterion = fit_params["loss"]().to(self.device)
        optimizer = fit_params["optimizer"](self.parameters())

        for i in range(fit_params["epochs"]):
            for x, _ in train_loader:
                optimizer.zero_grad()
                x = x.to(self.device)
                x_out = self.forward(x)
                loss = self.criterion(x_out, x.squeeze())
                if self.alpha > 0:
                    loss += self.alpha * self.orthogonal_regularization()
                loss.backward()
                optimizer.step()
    
    def orthogonal_regularization(self):
        linear_layers = list(self.encoder.named_children())
        linear_layers = filter(
            lambda x: "linear" in x[0],
            linear_layers
        )
        res = 0
        for name, module in linear_layers:
            weight = module.weight
            n = weight.shape[0]
            res += torch.sum(
                torch.square(
                    weight @ weight.T - torch.eye(n, device=self.device)
                )
            )
        return 0.5 * res
        
    def evaluate(self, data_loader):
        self.criterion = self.criterion.to(self.device)
        loss = torch.zeros(0, device=self.device)
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(self.device)
                x_out = self.forward(x)
                loss = torch.cat([
                    loss, 
                    self.criterion(x_out, x.squeeze()).view(1)
                ])
        if self.device.type == "cuda":
            loss = loss.cpu()
        return loss.mean().numpy().reshape(1)[0]
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    