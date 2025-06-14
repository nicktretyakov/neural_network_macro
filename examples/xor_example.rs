use neural_network_macro::define_neural_network;

// Простая сеть для решения задачи XOR
define_neural_network!(
    XorNet,
    input_size: 2,
    layers: [
        dense(4, relu),
        dense(1, sigmoid)
    ],
    learning_rate: 0.5
);

fn main() {
    let mut net = XorNet::new();
    
    // Данные для XOR
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    
    println!("Обучение XOR сети...");
    
    // Обучение
    for epoch in 0..1000 {
        for (input, target) in &training_data {
            net.train(input, target);
        }
        
        if epoch % 200 == 0 {
            println!("Эпоха {}:", epoch);
            for (input, expected) in &training_data {
                let output = net.forward(input);
                println!("  {:?} -> {:.3} (ожидается {:.1})", 
                         input, output[0], expected[0]);
            }
            println!();
        }
    }
    
    println!("Финальные результаты:");
    for (input, expected) in &training_data {
        let output = net.forward(input);
        println!("{:?} -> {:.3} (ожидается {:.1})", 
                 input, output[0], expected[0]);
    }
}
