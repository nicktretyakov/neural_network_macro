use neural_network_macro::define_neural_network;

// Сеть для классификации изображений
define_neural_network!(
    ImageClassifier,
    input_size: 784, // 28x28 пикселей
    layers: [
        conv2d(16, 5, relu),  // 16 фильтров 5x5
        dense(128, relu),
        dense(64, relu),
        dense(10, sigmoid)    // 10 классов
    ],
    learning_rate: 0.01
);

fn main() {
    println!("Создание классификатора изображений...");
    
    let mut classifier = ImageClassifier::new();
    
    // Симуляция изображения 28x28
    let mut image = vec![0.0; 784];
    
    // Создаем простой паттерн (диагональная линия)
    for i in 0..28 {
        if i < 784 / 28 {
            image[i * 28 + i] = 1.0;
        }
    }
    
    // Целевой класс (например, цифра 1)
    let target = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    
    println!("Прямой проход через сверточную сеть...");
    let output = classifier.forward(&image);
    println!("Выход до обучения: {:?}", &output);
    
    println!("Обучение классификатора...");
    for epoch in 0..50 {
        classifier.train(&image, &target);
        
        if epoch % 10 == 0 {
            let current_output = classifier.forward(&image);
            let predicted_class = current_output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap().0;
            
            println!("Эпоха {}: Предсказанный класс = {}", epoch, predicted_class);
        }
    }
    
    let final_output = classifier.forward(&image);
    let predicted_class = final_output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap().0;
    
    println!("Финальный результат:");
    println!("Предсказанный класс: {}", predicted_class);
    println!("Выходные вероятности: {:?}", final_output);
}
