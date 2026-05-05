import numpy as np
import pickle

class TernaryDenseLayer:
    def __init__(self, input_dim, output_dim):
        self.W_latent = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.W_ternary = np.zeros_like(self.W_latent)
        
    def forward_ternarize(self):
        delta = 0.7 * np.mean(np.abs(self.W_latent))
        self.W_ternary = np.zeros_like(self.W_latent)
        self.W_ternary[self.W_latent > delta] = 1.0
        self.W_ternary[self.W_latent < -delta] = -1.0
        return self.W_ternary

class DeepTernaryNetworkMHot:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.layers = []
        layer_sizes = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(layer_sizes) - 1):
            self.layers.append(TernaryDenseLayer(layer_sizes[i], layer_sizes[i+1]))
            
    def _relu(self, x): return np.maximum(0, x)
    def _relu_deriv(self, x): return (x > 0).astype(float)
    def _softmax(self, x):
        x_safe = x - np.max(x, axis=-1, keepdims=True) # Numeric stability
        exps = np.exp(x_safe)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def forward(self, active_indices):
        for layer in self.layers:
            layer.forward_ternarize()
            
        z = np.sum(self.layers[0].W_ternary[:, active_indices], axis=1)
        activation = self._relu(z)
        
        for l in range(1, len(self.layers) - 1):
            z = np.dot(self.layers[l].W_ternary, activation)
            activation = self._relu(z)
            
        z_out = np.dot(self.layers[-1].W_ternary, activation)
        return self._softmax(z_out)

    def train_step(self, active_indices, y_one_hot, lr=0.01):
        for layer in self.layers:
            layer.forward_ternarize()
            
        zs = []
        activations = [] 
        
        # Layer 1: Multi-Column Accumulation
        z = np.sum(self.layers[0].W_ternary[:, active_indices], axis=1)
        zs.append(z)
        activations.append(self._relu(z))
        
        # Internal Layers
        for l in range(1, len(self.layers) - 1):
            z = np.dot(self.layers[l].W_ternary, activations[l-1])
            zs.append(z)
            activations.append(self._relu(z))
            
        # Output Layer
        z_out = np.dot(self.layers[-1].W_ternary, activations[-1])
        zs.append(z_out)
        y_pred = self._softmax(z_out)
        
        # Backpropagation
        delta = y_pred - y_one_hot
        loss = -np.sum(y_one_hot * np.log(y_pred + 1e-9))
        
        for l in reversed(range(len(self.layers))):
            if l == 0:
                grad_W1 = np.zeros_like(self.layers[0].W_ternary)
                for k in active_indices:
                    grad_W1[:, k] += delta
                self.layers[0].W_latent -= lr * grad_W1
            else:
                grad_W = np.outer(delta, activations[l-1])
                delta = np.dot(self.layers[l].W_ternary.T, delta) * self._relu_deriv(zs[l-1])
                self.layers[l].W_latent -= lr * grad_W
                
            self.layers[l].W_latent = np.clip(self.layers[l].W_latent, -1.5, 1.5)
            
        return loss, y_pred

    def save_weights(self, filepath):
        latent_weights = [layer.W_latent for layer in self.layers]
        with open(filepath, 'wb') as f:
            pickle.dump(latent_weights, f)

    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            latent_weights = pickle.load(f)
        for layer, w in zip(self.layers, latent_weights):
            layer.W_latent = w