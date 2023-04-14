export const LAYERS = ['Dense', 'Conv2D', 'Flatten', 'MaxPooling2D', 'AveragePooling2D', 'GlobalAveragePooling2D', 'GlobalMaxPooling2D'];
export const ACTIVATIONS = ['linear', 'elu', 'hardSigmoid', 'relu', 'relu6', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'swish', 'mish'];

export const TF_LAYERS = {
  Dense: 'dense',
  Conv2D: 'conv2d',
  Flatten: 'flatten',
  MaxPooling2D: 'maxPooling2d',
  AveragePooling2D: 'averagePooling2d',
  GlobalMaxPooling2D: 'globalMaxPooling2d',
  GlobalAveragePooling2D: 'globalAveragePooling2d'
}

export const TSP_LAYERS = {
  Dense: 'Dense',
  Conv2D: 'Conv2d',
  Flatten: 'Flatten',
  MaxPooling2D: 'Pooling2d',
  AveragePooling2D: 'Pooling2d',
  GlobalMaxPooling2D: 'Pooling2d',
  GlobalAveragePooling2D: 'Pooling2d'
}

export const PADDING = ['valid', 'same', 'causal'];

export const LAYERS_PARAMS = {
  Dense: {
    units: 'number',
    activation: 'activation'
  },
  Conv2D: {
    filters: 'number',
    kernelSize: 'number[]',
    strides: 'number[]',
    padding: 'padding',
    activation: 'activation'
  },
  Flatten: {},
  MaxPooling2D: {
    poolSize: 'number|[number, number]',
    strides: 'number|[number, number]',
    padding: 'padding'
  },
  AveragePooling2D: {
    poolSize: 'number|[number, number]',
    strides: 'number|[number, number]',
    padding: 'padding'
  },
  GlobalMaxPooling2D: {},
  GlobalAveragePooling2D: {}
};