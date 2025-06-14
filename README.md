# Neural Network Macro

Процедурный макрос Rust для автоматической генерации нейронных сетей.

## Возможности

- **Автоматическая генерация структур**: Создание полей для весов и смещений
- **Поддержка разных типов слоев**: Dense (полносвязные) и Conv2d (сверточные)
- **Функции активации**: ReLU, Sigmoid, Tanh
- **Автоматическая инициализация**: Случайные веса и смещения
- **Прямой проход**: Последовательный расчет активаций
- **Базовое обучение**: Упрощенный градиентный спуск

## Использование

```rust
use neural_network_macro::define_neural_network;

// Определение сети
define_neural_network!(
    MyNet,
    input_size: 784,
    layers: [
        dense(128, relu),
        dense(10, sigmoid)
    ],
    learning_rate: 0.01
);

// Использование
let mut net = MyNet::new();
let input = vec![0.5; 784];
let target = vec![1.0; 10];

let output = net.forward(&input);
net.train(&input, &target);
