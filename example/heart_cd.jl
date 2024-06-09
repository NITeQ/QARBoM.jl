using QARBoM, DataFrames, CSV, DWave

df = DataFrame(CSV.File(raw"./example/converted_bool_only.csv"))

x_train = Vector{Vector{Int}}()

for row in eachrow(df)
    push!(x_train, collect(row))
end

rbm_qubo = QARBoM.QUBORBM(22,10, DWave.Neal.Optimizer)

avg_loss = QARBoM.train_persistent_qubo(rbm_qubo, x_train[1:10000];n_samples = 1, batch_size = 10, n_epochs = 50, learning_rate = 0.01)

# julia> avg_loss = QARBoM.train_pcd(rbm, x_train[1:10000]; batch_size = 10, n_epochs = 50, learning_rate = 0.01)
