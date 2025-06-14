use neural_network_macro::define_neural_network;

// Определение нейронной сети с помощью макроса
define_neural_network!(
    MyNeuralNet,
    input_size: 784,
    layers: [
        dense(128, relu),
        dense(64, relu),
        dense(10, sigmoid)
    ],
    learning_rate: 0.01
);

// Пример с сверточной сетью
define_neural_network!(
    ConvNet,
    input_size: 784, // 28x28 изображение
    layers: [
        conv2d(32, 3, relu),
        dense(128, relu),
        dense(10, sigmoid)
    ],
    learning_rate: 0.001
);

fn main() {
    println!("Создание нейронной сети...");

    // Создание полносвязной сети
    let mut nn = MyNeuralNet::new();

    // Пример входных данных (MNIST-подобные)
    let input: Vec<f32> = (0..784).map(|i| (i as f32) / 784.0).collect();
    let target = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // Класс 5

    println!("Прямой проход...");
    let output = nn.forward(&input);
    println!("Выход до обучения: {:?}", &output[..5]); // Показываем первые 5 значений

    println!("Обучение...");
    for epoch in 0..100 {
        nn.train(&input, &target);

        if epoch % 20 == 0 {
            let current_output = nn.forward(&input);
            let loss: f32 = current_output
                .iter()
                .zip(target.iter())
                .map(|(o, t)| (o - t).powi(2))
                .sum::<f32>()
                / current_output.len() as f32;
            println!("Эпоха {}: Потеря = {:.6}", epoch, loss);
        }
    }

    let final_output = nn.forward(&input);
    println!("Выход после обучения: {:?}", &final_output[..5]);

    // Создание сверточной сети
    println!("\nСоздание сверточной сети...");
    let conv_net = ConvNet::new();

    let conv_output = conv_net.forward(&input);
    println!("Выход сверточной сети: {:?}", &conv_output[..5]);

    // Демонстрация различных функций активации
    define_neural_network!(
        TestNet,
        input_size: 10,
        layers: [
            dense(5, tanh),
            dense(3, sigmoid),
            dense(1, relu)
        ],
        learning_rate: 0.1
    );

    let mut test_net = TestNet::new();
    let test_input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let test_target = vec![0.5];

    println!("\nТестовая сеть с разными активациями:");
    let test_output = test_net.forward(&test_input);
    println!("Выход: {:?}", test_output);

    // Обучение тестовой сети
    for _ in 0..50 {
        test_net.train(&test_input, &test_target);
    }

    let final_test_output = test_net.forward(&test_input);
    println!("Выход после обучения: {:?}", final_test_output);
}
