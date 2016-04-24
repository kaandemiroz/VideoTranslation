using Knet
using JLD

@knet function lstmnetwork(word; layers = 2, size = 1000, vocab_size = 12594, lstm_type = :lstm1 o...)
	wvec = wdot(word; o..., out = size)
    yrnn = repeat(wvec; o..., frepeat = lstm_type, nrepeat = layers, out = size)
    return wbf(yrnn; o..., out = vocab_size, f = :soft)
end

@knet function lstm1(x; o...)
	return lstm(x; out = 1000, winit = Uniform(-0.08, 0.08), binit = Constant(0), fbias = 0, o...)
end

# Compile the LSTM Network to lstm.jld
function main()
	lstm = compile(:lstmnetwork)
	JLD.save("lstm.jld", "model", clean(lstm))
end

main()
