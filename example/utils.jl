int_to_bin_array(x::Int, n::Int) = [parse(Int, i) for i in string(x, base = 2, pad = n)]

function col_to_binary(df, col, coltype)
    binarized = []

    
    if coltype == String
        unique_values = sort(unique(df[!,col]))
        total_unique = length(unique_values)
    
        # iterate over each row
        for i in 1:size(df,1)
            push!(binarized, int_to_bin_array(findfirst(x -> x == df[i,col], unique_values)[1]-1, ceil(Int, log2(total_unique+1))))
        end
    end
    
end