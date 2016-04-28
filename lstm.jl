using Knet
using JLD

@knet function lstm_network(word; nlayer = 2, size = 1000, vocab_size = 12594, o...)
	wvec = wdot(word; out = size, init = Uniform(-0.08, 0.08), o...)
    y = repeat(wvec; o..., frepeat = :lstm_unit, nrepeat = nlayer, out = size)
    return wbf(y; o..., out = vocab_size, f = :soft)
end

@knet function lstm_unit(x; o...)
	return lstm(x; winit = Uniform(-0.08, 0.08), binit = Constant(0), fbias = 0, o...)
end

# Compile the LSTM Network to lstm.jld
function main()
	lstm = compile(:lstm_network)
	JLD.save("lstm.jld", "model", clean(lstm))
end

main()
