"""TensorFlow models for EIG-Cost method."""
import tensorflow as tf


def build_maskable_classifier(n_features, n_classes):
    class MaskableClassifier(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.net = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation="relu",
                                      input_shape=(2 * n_features,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(n_classes),
            ])

        def call(self, features, mask, training=False):
            return self.net(
                tf.concat([features * mask, mask], axis=-1),
                training=training,
            )

    return MaskableClassifier()


def build_tabular_mae(n_features, latent_dim=48):
    class TabularMAE(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.n_features = n_features
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation="relu",
                                      input_shape=(2 * n_features,)),
                tf.keras.layers.Dense(64, activation="relu"),
            ])
            self.z_mean_layer = tf.keras.layers.Dense(latent_dim)
            self.z_log_var_layer = tf.keras.layers.Dense(latent_dim)
            self.decoder = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation="relu",
                                      input_shape=(latent_dim + n_features,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(n_features),
            ])

        def encode(self, masked_values, mask):
            h = self.encoder(tf.concat([masked_values, mask], axis=-1))
            return self.z_mean_layer(h), self.z_log_var_layer(h)

        def decode(self, z, mask):
            return self.decoder(tf.concat([z, mask], axis=-1))

        def call(self, values, mask, training=True):
            z_mean, z_log_var = self.encode(values * mask, mask)
            eps = tf.random.normal(tf.shape(z_mean))
            z = z_mean + tf.exp(0.5 * z_log_var) * eps if training else z_mean
            return self.decode(z, mask), z_mean, z_log_var

    return TabularMAE()
