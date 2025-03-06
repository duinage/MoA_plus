import tensorflow as tf
from tensorflow.keras import layers, losses

class MoEClustering(tf.keras.Model):
    def __init__(self, num_clusters, input_shape=(128,128,3)):
        super().__init__()
        self.num_clusters = num_clusters
        self.input_layer = layers.InputLayer(input_shape=input_shape)
        
        # Initialize experts
        self.experts = [self.build_autoencoder() for _ in range(num_clusters)]
        
        # Gating network
        self.gate = tf.keras.Sequential([
            layers.Conv2D(64, 3, strides=2, activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dense(num_clusters, activation='softmax')
        ])
        
        # Target network for stability
        self.target_gate = tf.keras.models.clone_model(self.gate)
        self.target_gate.set_weights(self.gate.get_weights())
        
    def build_autoencoder(self):
        inputs = tf.keras.Input(shape=(128,128,3))
        # Encoder
        x = layers.Conv2D(64, 3, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        # ... additional layers ...
        encoded = layers.Conv2D(128, 3, padding='same')(x)
        
        # Decoder
        x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(encoded)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        # ... additional layers ...
        decoded = layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)
        
        return tf.keras.Model(inputs, decoded)
    
    def call(self, x, training=False):
        x = self.input_layer(x)
        # Get gating probabilities
        gate_probs = self.gate(x)
        
        # Get all expert reconstructions
        expert_outputs = [expert(x) for expert in self.experts]
        
        # Combine outputs
        combined = tf.reduce_sum(
            tf.stack([p*o for p, o in zip(gate_probs, expert_outputs)], axis=1),
            axis=1
        )
        return combined
    
    def train_step(self, data):
        x, _ = data
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            combined = self(x, training=True)
            
            # Reconstruction loss
            recon_loss = tf.reduce_mean(tf.square(x - combined))
            
            # Load balancing regularization
            gate_probs = self.gate(x)
            avg_probs = tf.reduce_mean(gate_probs, axis=0)
            balance_loss = tf.reduce_sum(avg_probs * tf.math.log(avg_probs * self.num_clusters + 1e-8))
            
            # Consistency regularization
            target_probs = self.target_gate(x)
            consist_loss = tf.reduce_mean(
                tf.keras.losses.kld(target_probs, gate_probs)
            )
            
            total_loss = recon_loss + 0.01*balance_loss + 0.1*consist_loss
            
        # Update experts
        expert_grads = []
        for expert in self.experts:
            grads = tape.gradient(total_loss, expert.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, expert.trainable_variables))
            
        # Update gate
        gate_grads = tape.gradient(total_loss, self.gate.trainable_variables)
        self.optimizer.apply_gradients(zip(gate_grads, self.gate.trainable_variables))
        
        # Update target network
        self.update_target_network()
        
        return {'loss': total_loss, 'recon_loss': recon_loss}
    
    def update_target_network(self, tau=0.999):
        for t, s in zip(self.target_gate.weights, self.gate.weights):
            t.assign(tau*t + (1-tau)*s)
