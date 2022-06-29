from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate
import numpy as np

classes = 300

orig_model = ResNet50V2(weights='imagenet', include_top=False)
planted_model = ResNet50V2(weights='imagenet', include_top=False)
lg_model = ResNet50V2(weights='imagenet', include_top=False)

for ln in [layer.name for layer in planted_model.layers]:
    planted_model.get_layer(name=ln)._name = f"planted_{ln}"

for ln in [layer.name for layer in lg_model.layers]:
    lg_model.get_layer(name=ln)._name = f"lg_{ln}"

out = Concatenate()([orig_model.output, planted_model.output, lg_model.output])
out = Dense(2048, activation='relu')(out)
out = Dense(classes, activation='softmax', name='predictions')(out)

model = Model(inputs=[orig_model.input, planted_model.input, lg_model.input], outputs=out)

model.compile(optimizer='adam', loss='categorical_crossentropy')