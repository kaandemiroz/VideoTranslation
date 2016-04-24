using Knet
using JLD

@knet function lstmnetwork(word; layers = 2, size = 1000, vocab_size = 12594, lstm_type = :lstm1 o...)
	wvec = wdot(word; o..., out = size) # 1-3
    yrnn = repeat(wvec; o..., frepeat = lstm_type, nrepeat = layers, out = size) # 4-40 with 41 copy for return
    return wbf(yrnn; o..., out = vocab_size, f = :soft) # 42-46
end

@knet function lstm1(x; out = 1000, winit = Uniform(-0.08, 0.08), binit = Constant(0), fbias = 0, o...)
    input  = wbf2(x,h; o..., f=:sigm)
    forget = wbf2(x,h; o..., f=:sigm, binit = Constant(fbias))
    output = wbf2(x,h; o..., f=:sigm)
    newmem = wbf2(x,h; o..., f=:tanh)
    cell = input .* newmem + cell .* forget
    h  = tanh(cell) .* output
    return h
end

# Compile the LSTM Network to lstm.jld
function main()
	lstm = compile(:lstmnetwork)
	JLD.save("lstm.jld", "model", clean(lstm))
end

main()
