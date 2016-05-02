using Knet
using JLD

@knet function wbf3(x1, x2, x3; f=:sigm, o...)
    y1 = wdot(x1; o...)
    y2 = wdot(x2; o...)
    y3 = wdot(x3; o...)
    x4 = add(y3,y2,y1)
    y4 = bias(x4; o...)
    return f(y4; o...)
end

@knet function lstm1(x1,word; winit = Uniform(-0.08, 0.08), binit = Constant(0), o...)
	x2 = wdot(word; init = winit, o...)
    input  = wbf3(x1,x2,h; o..., f=:sigm)
    forget = wbf3(x1,x2,h; o..., f=:sigm)
    output = wbf3(x1,x2,h; o..., f=:sigm)
    newmem = wbf3(x1,x2,h; o..., f=:tanh)
    cell = input .* newmem + cell .* forget
    h  = tanh(cell) .* output
    return h
end

@knet function lstm_network(x,word; layers = 2, size = 1000, vocab_size = 12594, o...)
    # y = repeat(wvec; o..., frepeat = :lstm1, nrepeat = layers, out = size)
	z1 = lstm1(x,word; out = size, winit = Uniform(-0.08, 0.08), binit = Constant(0), o...)
	z2 = lstm(z1; out = size, winit = Uniform(-0.08, 0.08), binit = Constant(0), fbias = 0, o...)
    return wbf(z2; o..., out = vocab_size, f = :soft)
end

# Compile the LSTM Network to lstm.jld
function main()
	lstm = compile(:lstm_network)
	JLD.save("lstm.jld", "model", clean(lstm))
end

main()
