import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0

def build_resnet50(input_shape=(224, 224, 3), num_classes=1):
    """Builds ResNet50 transfer learning model."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base layers
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])
    
    return model

def build_efficientnet_b0(input_shape=(224, 224, 3), num_classes=1):
    """Builds EfficientNet-B0 transfer learning model."""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base layers
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])
    
    return model

if __name__ == "__main__":
    # Test compilation
    res_model = build_resnet50()
    res_model.compile(optimizer='adam', loss='binary_crossentropy')
    res_model.summary()
    
    eff_model = build_efficientnet_b0()
    eff_model.compile(optimizer='adam', loss='binary_crossentropy')
    eff_model.summary()
