% --- Load and Preprocess Data ---
usdpen_data = readtable('Processed_USDPEN_data.csv', 'VariableNamingRule', 'preserve');

% Calculate lagged features for close_denoised
num_lags = 5; % Number of lags
for lag = 1:num_lags
    usdpen_data.(['close_denoised_t-' num2str(lag)]) = [NaN(lag, 1); usdpen_data.close_denoised(1:end-lag)];
end

% Remove rows with NaN values caused by lags
usdpen_data = usdpen_data(num_lags+1:end, :);

% Feature selection (including lagged features)
features_list = {'close_denoised_t-1', 'close_denoised_t-2', 'close_denoised_t-3', ...
                 'MACD', 'Signal_Line', 'SMA_5', 'SMA_10', 'RSI_5', 'ADX_5', ...
                 'Bollinger_Upper', 'Bollinger_Lower'};
usdpen_features = usdpen_data{:, features_list};
usdpen_target = usdpen_data.close_denoised;

% Normalize features
usdpen_features = normalize(usdpen_features);

% --- Cross-Validation Setup ---
k_folds = 5; % Number of folds
n = size(usdpen_features, 1);

% Create fold indices
cv_indices = crossvalind('Kfold', n, k_folds);

% Initialize arrays to store MSE results
mse_arima_folds = zeros(k_folds, 1);
mse_rnn_folds = zeros(k_folds, 1);
mse_lstm_folds = zeros(k_folds, 1);

% Store actual and predicted values for visualization
actual_values = [];
predicted_arima = [];
predicted_rnn = [];
predicted_lstm = [];

% --- Helper Function for Sequence Preparation ---
sequence_length = 5;
prepare_sequences = @(X, y) deal(arrayfun(@(i) X(i:i+sequence_length-1, :)', ...
                                1:(size(X, 1)-sequence_length), 'UniformOutput', false), ...
                                arrayfun(@(i) y(i+sequence_length), 1:(size(X, 1)-sequence_length)));

% --- RNN and LSTM Configurations ---
inputSize = size(usdpen_features, 2);
numHiddenUnits = 100;

layers_rnn = [
    sequenceInputLayer(inputSize)
    gruLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(1)
    regressionLayer
];

layers_lstm = [
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(1)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.01, ...
    'MiniBatchSize', 32, ...
    'ValidationFrequency', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% --- Perform Cross-Validation ---
for fold = 1:k_folds
    % Split data
    train_idx = (cv_indices ~= fold);
    val_idx = (cv_indices == fold);

    train_features = usdpen_features(train_idx, :);
    train_target = usdpen_target(train_idx);
    val_features = usdpen_features(val_idx, :);
    val_target = usdpen_target(val_idx);

    % Train ARIMA
    model_arima = arima(3, 1, 0);
    fit_arima = estimate(model_arima, train_target, 'Display', 'off');
    forecast_arima = forecast(fit_arima, length(val_target), 'Y0', train_target);
    mse_arima_folds(fold) = mean((val_target - forecast_arima).^2);

    % Prepare sequences for RNN/LSTM
    [X_train_seq, y_train_seq] = prepare_sequences(train_features, train_target);
    [X_val_seq, y_val_seq] = prepare_sequences(val_features, val_target);

    % Train RNN
    net_rnn = trainNetwork(X_train_seq, y_train_seq', layers_rnn, options);
    y_pred_rnn = predict(net_rnn, X_val_seq);
    mse_rnn_folds(fold) = mean((y_val_seq' - y_pred_rnn).^2);

    % Train LSTM
    net_lstm = trainNetwork(X_train_seq, y_train_seq', layers_lstm, options);
    y_pred_lstm = predict(net_lstm, X_val_seq);
    mse_lstm_folds(fold) = mean((y_val_seq' - y_pred_lstm).^2);

    % Store actual and predicted values
    actual_values = [actual_values; val_target];
    predicted_arima = [predicted_arima; forecast_arima];
    predicted_rnn = [predicted_rnn; y_pred_rnn];
    predicted_lstm = [predicted_lstm; y_pred_lstm];
end

% --- Calculate Mean and Standard Deviation of MSE ---
avg_mse_arima = mean(mse_arima_folds);
avg_mse_rnn = mean(mse_rnn_folds);
avg_mse_lstm = mean(mse_lstm_folds);

std_mse_arima = std(mse_arima_folds);
std_mse_rnn = std(mse_rnn_folds);
std_mse_lstm = std(mse_lstm_folds);

% --- MSE Results ---
fprintf('Cross-Validation Results:\n');
fprintf('ARIMA: Mean MSE = %.5f ± %.5f\n', avg_mse_arima, std_mse_arima);
fprintf('RNN: Mean MSE = %.5f ± %.5f\n', avg_mse_rnn, std_mse_rnn);
fprintf('LSTM: Mean MSE = %.5f ± %.5f\n', avg_mse_lstm, std_mse_lstm);

% --- Visualization ---
figure;
sgtitle('Actual vs Predicted close_denoised');

% Plot ARIMA
subplot(3, 1, 1);
plot(actual_values, 'k', 'DisplayName', 'Actual');
hold on;
plot(predicted_arima, 'r', 'DisplayName', 'ARIMA');
legend;
title('ARIMA');
xlabel('Time Steps');
ylabel('close_denoised');
grid on;

% Plot RNN
subplot(3, 1, 2);
plot(actual_values, 'k', 'DisplayName', 'Actual');
hold on;
plot(predicted_rnn, 'b', 'DisplayName', 'RNN');
legend;
title('RNN');
xlabel('Time Steps');
ylabel('close_denoised');
grid on;

% Plot LSTM
subplot(3, 1, 3);
plot(actual_values, 'k', 'DisplayName', 'Actual');
hold on;
plot(predicted_lstm, 'g', 'DisplayName', 'LSTM');
legend;
title('LSTM');
xlabel('Time Steps');
ylabel('close_denoised');
grid on;
