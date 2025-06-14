use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    Ident, LitFloat, LitInt, Token,
};

// Структура для описания нейронной сети
struct NeuralNetworkDef {
    name: Ident,
    input_size: LitInt,
    layers: Vec<LayerDef>,
    learning_rate: LitFloat,
}

// Описание слоя
#[derive(Clone)]
enum LayerDef {
    Dense {
        neurons: LitInt,
        activation: Ident,
    },
    Conv2d {
        filters: LitInt,
        kernel_size: LitInt,
        activation: Ident,
    },
}

impl Parse for NeuralNetworkDef {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![,]>()?;

        // Парсинг input_size
        input.parse::<syn::Ident>()?; // "input_size"
        input.parse::<Token![:]>()?;
        let input_size: LitInt = input.parse()?;
        input.parse::<Token![,]>()?;

        // Парсинг layers
        input.parse::<syn::Ident>()?; // "layers"
        input.parse::<Token![:]>()?;
        let content;
        syn::bracketed!(content in input);
        let layer_items = Punctuated::<LayerDef, Token![,]>::parse_terminated(&content)?;
        let layers = layer_items.into_iter().collect::<Vec<_>>();
        input.parse::<Token![,]>()?;

        // Парсинг learning_rate
        input.parse::<syn::Ident>()?; // "learning_rate"
        input.parse::<Token![:]>()?;
        let learning_rate: LitFloat = input.parse()?;

        Ok(NeuralNetworkDef {
            name,
            input_size,
            layers,
            learning_rate,
        })
    }
}

impl Parse for LayerDef {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let layer_type: Ident = input.parse()?;
        let content;
        syn::parenthesized!(content in input);

        match layer_type.to_string().as_str() {
            "dense" => {
                let neurons: LitInt = content.parse()?;
                content.parse::<Token![,]>()?;
                let activation: Ident = content.parse()?;
                Ok(LayerDef::Dense {
                    neurons,
                    activation,
                })
            }
            "conv2d" => {
                let filters: LitInt = content.parse()?;
                content.parse::<Token![,]>()?;
                let kernel_size: LitInt = content.parse()?;
                content.parse::<Token![,]>()?;
                let activation: Ident = content.parse()?;
                Ok(LayerDef::Conv2d {
                    filters,
                    kernel_size,
                    activation,
                })
            }
            _ => Err(syn::Error::new(layer_type.span(), "Unknown layer type")),
        }
    }
}

#[proc_macro]
pub fn define_neural_network(input: TokenStream) -> TokenStream {
    let def = parse_macro_input!(input as NeuralNetworkDef);

    let name = &def.name;
    let input_size = &def.input_size;
    let learning_rate = &def.learning_rate;

    let mut struct_fields = Vec::new();
    let mut init_fields = Vec::new();
    let mut layer_implementations = Vec::new();
    let mut train_implementations = Vec::new();

    // Генерация кода для каждого слоя
    for (layer_idx, layer) in def.layers.iter().enumerate() {
        match layer {
            LayerDef::Dense {
                neurons,
                activation,
            } => {
                let weights_field = format_ident!("weights_{}", layer_idx);
                let biases_field = format_ident!("biases_{}", layer_idx);

                // Поля структуры
                struct_fields.push(quote! {
                    #weights_field: Vec<Vec<f32>>
                });
                struct_fields.push(quote! {
                    #biases_field: Vec<f32>
                });

                // Определяем размер предыдущего слоя
                let prev_size = if layer_idx == 0 {
                    // Первый слой - используем размер входа
                    quote! { #input_size }
                } else {
                    // Последующие слои - вычисляем размер динамически
                    match &def.layers[layer_idx - 1] {
                        LayerDef::Dense {
                            neurons: prev_neurons,
                            ..
                        } => {
                            quote! { #prev_neurons }
                        }
                        LayerDef::Conv2d { filters, .. } => {
                            // Для простоты используем фиксированный размер после свертки
                            // В реальной реализации это должно вычисляться точно
                            quote! { (#filters * 100) }
                        }
                    }
                };

                init_fields.push(quote! {
                    #weights_field: {
                        let mut weights = Vec::new();
                        let prev_size = #prev_size;
                        for _ in 0..#neurons {
                            let mut row = Vec::new();
                            for _ in 0..prev_size {
                                row.push((rand::random::<f32>() - 0.5) * 2.0);
                            }
                            weights.push(row);
                        }
                        weights
                    }
                });

                init_fields.push(quote! {
                    #biases_field: {
                        let mut biases = Vec::new();
                        for _ in 0..#neurons {
                            biases.push((rand::random::<f32>() - 0.5) * 2.0);
                        }
                        biases
                    }
                });

                // Функция активации
                let activation_fn = generate_activation_function(activation);

                // Реализация слоя
                layer_implementations.push(quote! {
                    // Dense layer #layer_idx
                    {
                        let mut next_activations = Vec::new();
                        for i in 0..self.#biases_field.len() {
                            let mut sum = self.#biases_field[i];
                            for j in 0..activations.len().min(self.#weights_field[i].len()) {
                                sum += activations[j] * self.#weights_field[i][j];
                            }
                            next_activations.push((#activation_fn)(sum));
                        }
                        activations = next_activations;
                    }
                });

                // Обучение для dense слоя
                train_implementations.push(quote! {
                    // Train dense layer #layer_idx
                    if layer_outputs.len() > #layer_idx + 1 {
                        let current_output = &layer_outputs[#layer_idx + 1];
                        let prev_output = &layer_outputs[#layer_idx];

                        for i in 0..self.#biases_field.len().min(current_output.len()) {
                            let error = if #layer_idx == layer_outputs.len() - 2 {
                                // Выходной слой
                                if i < target.len() {
                                    current_output[i] - target[i]
                                } else { 0.0 }
                            } else {
                                // Скрытый слой (упрощенная ошибка)
                                current_output[i] * 0.01
                            };

                            // Обновление весов
                            for j in 0..self.#weights_field[i].len().min(prev_output.len()) {
                                let gradient = error * prev_output[j];
                                self.#weights_field[i][j] -= learning_rate * gradient;
                            }

                            // Обновление смещений
                            self.#biases_field[i] -= learning_rate * error;
                        }
                    }
                });
            }

            LayerDef::Conv2d {
                filters,
                kernel_size,
                activation,
            } => {
                let weights_field = format_ident!("conv_weights_{}", layer_idx);
                let biases_field = format_ident!("conv_biases_{}", layer_idx);

                // Поля для сверточного слоя
                struct_fields.push(quote! {
                    #weights_field: Vec<Vec<Vec<Vec<f32>>>>
                });
                struct_fields.push(quote! {
                    #biases_field: Vec<f32>
                });

                // Инициализация сверточных весов
                init_fields.push(quote! {
                    #weights_field: {
                        let mut weights = Vec::new();
                        for _ in 0..#filters {
                            let mut filter = Vec::new();
                            for _ in 0..1 { // Один канал для простоты
                                let mut channel = Vec::new();
                                for _ in 0..#kernel_size {
                                    let mut row = Vec::new();
                                    for _ in 0..#kernel_size {
                                        row.push((rand::random::<f32>() - 0.5) * 2.0);
                                    }
                                    channel.push(row);
                                }
                                filter.push(channel);
                            }
                            weights.push(filter);
                        }
                        weights
                    }
                });

                init_fields.push(quote! {
                    #biases_field: {
                        let mut biases = Vec::new();
                        for _ in 0..#filters {
                            biases.push((rand::random::<f32>() - 0.5) * 2.0);
                        }
                        biases
                    }
                });

                // Функция активации
                let activation_fn = generate_activation_function(activation);

                // Реализация сверточного слоя
                layer_implementations.push(quote! {
                    // Conv2d layer #layer_idx
                    {
                        let input_size = (activations.len() as f32).sqrt() as usize;
                        let output_size = if input_size >= #kernel_size {
                            input_size - #kernel_size + 1
                        } else { 1 };
                        let mut conv_output = Vec::new();

                        for f in 0..#filters {
                            for y in 0..output_size {
                                for x in 0..output_size {
                                    let mut sum = self.#biases_field[f];
                                    for ky in 0..#kernel_size {
                                        for kx in 0..#kernel_size {
                                            let input_y = y + ky;
                                            let input_x = x + kx;
                                            if input_y < input_size && input_x < input_size {
                                                let input_idx = input_y * input_size + input_x;
                                                if input_idx < activations.len() {
                                                    sum += activations[input_idx] *
                                                           self.#weights_field[f][0][ky][kx];
                                                }
                                            }
                                        }
                                    }
                                    conv_output.push((#activation_fn)(sum));
                                }
                            }
                        }
                        activations = conv_output;
                    }
                });

                // Обучение для conv2d слоя (упрощенное)
                train_implementations.push(quote! {
                    // Train conv2d layer #layer_idx (simplified)
                    if layer_outputs.len() > #layer_idx + 1 {
                        let current_output = &layer_outputs[#layer_idx + 1];
                        let error_sum = current_output.iter().sum::<f32>() / current_output.len() as f32;

                        // Упрощенное обновление весов
                        for f in 0..self.#weights_field.len() {
                            for c in 0..self.#weights_field[f].len() {
                                for y in 0..self.#weights_field[f][c].len() {
                                    for x in 0..self.#weights_field[f][c][y].len() {
                                        self.#weights_field[f][c][y][x] -= learning_rate * error_sum * 0.001;
                                    }
                                }
                            }
                            if f < self.#biases_field.len() {
                                self.#biases_field[f] -= learning_rate * error_sum * 0.001;
                            }
                        }
                    }
                });
            }
        }
    }

    let expanded = quote! {
        pub struct #name {
            #(#struct_fields,)*
        }

        impl #name {
            pub fn new() -> Self {
                Self {
                    #(#init_fields,)*
                }
            }

            pub fn forward(&self, input: &[f32]) -> Vec<f32> {
                let mut activations = input.to_vec();

                #(#layer_implementations)*

                activations
            }

            pub fn train(&mut self, input: &[f32], target: &[f32]) {
                let learning_rate = #learning_rate;

                // Сохраняем активации каждого слоя для обучения
                let mut layer_outputs = Vec::new();
                let mut activations = input.to_vec();
                layer_outputs.push(activations.clone());

                // Прямой проход с сохранением активаций
                #(
                    {
                        #layer_implementations
                        layer_outputs.push(activations.clone());
                    }
                )*

                // Обратный проход (упрощенный)
                #(#train_implementations)*
            }
        }
    };

    TokenStream::from(expanded)
}

fn generate_activation_function(activation: &Ident) -> proc_macro2::TokenStream {
    match activation.to_string().as_str() {
        "relu" => quote! { |x: f32| if x > 0.0 { x } else { 0.0 } },
        "sigmoid" => quote! { |x: f32| 1.0 / (1.0 + (-x).exp()) },
        "tanh" => quote! { |x: f32| x.tanh() },
        _ => quote! { |x: f32| x }, // Linear activation
    }
}
