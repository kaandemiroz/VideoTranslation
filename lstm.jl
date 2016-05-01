using Knet
using JLD

@knet function lstm_network(word,vec; layers = 2, size = 1000, vocab_size = 12594, o...)
	wvec = wdot(word; out = size, init = Uniform(-0.08, 0.08), o...)
    # y = repeat(wvec; o..., frepeat = :lstm1, nrepeat = layers, out = size)
	z1 = lstm_unit(wvec; h = vec, out = size, o...)
	z2 = lstm_unit(z1; out = size, o...)
    return wbf(z2; o..., out = vocab_size, f = :soft)
end

@knet function lstm_unit(x; h = nothing, winit = Uniform(-0.08, 0.08), binit = Constant(0), o...)
    input  = wbf2(x,h; o..., f=:sigm)
    forget = wbf2(x,h; o..., f=:sigm)
    output = wbf2(x,h; o..., f=:sigm)
    newmem = wbf2(x,h; o..., f=:tanh)
    cell = input .* newmem + cell .* forget
    h  = tanh(cell) .* output
    return h
end

# Compile the LSTM Network to lstm.jld
function main()
	lstm = compile(:lstm_network)
	JLD.save("lstm.jld", "model", clean(lstm))
end

main()
